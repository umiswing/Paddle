// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/core/distributed/comm_context_manager.h"

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/slice_kernel_impl.h"
#include "paddle/phi/kernels/slice_kernel.h"

#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/kernels/gpu/comm_overlap_utils.h"
#include "paddle/phi/kernels/gpu/submatrix_parallel_utils.h"

#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {
template <typename T, typename Context>
void SMPGEMMReduceScatterKernel(const Context& dev_ctx,
                                const DenseTensor& a,
                                const DenseTensor& b,
                                const bool transpose_b,
                                const int32_t ring_id,
                                DenseTensor* out) {
  VLOG(10) << "SMPGEMMReduceScatterKernel";
  // get ProcessGroup and NCCLCommContext
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(
      pg, nullptr, common::errors::Unavailable("ProcessGroup is nullptr."));

  distributed::NCCLCommContext* comm_ctx =
      pg->GetOrCreateCommContext(a.place(), distributed::CommType::ALLGATHER);

  PADDLE_ENFORCE_NE(
      comm_ctx, nullptr, common::errors::Unavailable("comm_ctx is nullptr."));

  const int32_t a_rank = a.dims().size();
  const int32_t b_rank = b.dims().size();
  PADDLE_ENFORCE_EQ(
      a_rank,
      2,
      common::errors::InvalidArgument(
          "a must be 2-D tensor, but received a %d-D tensor", a_rank));
  PADDLE_ENFORCE_EQ(
      b_rank,
      2,
      common::errors::InvalidArgument(
          "b must be 2-D tensor, but received a %d-D tensor", b_rank));

  int32_t rank = pg->GetRank();
  int32_t world_size = pg->GetSize();

  // init workspace (tmp out)
  int64_t global_m = a.dims()[0];
  int64_t global_n = transpose_b ? b.dims()[0] : b.dims()[1];
  PADDLE_ENFORCE(global_m % world_size == 0,
                 "m needs to be divisible by world size");
  std::vector<DenseTensor> tmp_outs(2);
  for (int i = 0; i < 2; i++) {
    tmp_outs[i].Resize(common::make_dim(global_m / world_size, global_n));
    dev_ctx.template Alloc<T>(&(tmp_outs[i]));
  }

  size_t workspace_size_in_bytes = tmp_outs[0].numel() * sizeof(T);

  // init comm buffers
  std::vector<int64_t> comm_buffer_shape{global_m / world_size, global_n};
  static BuffersHolder<T> comm_buffers_holder{comm_buffer_shape, dev_ctx, pg};
  std::vector<DenseTensor> comm_buffers =
      comm_buffers_holder.get_buffers(comm_buffer_shape);

  // init barrier
  static BuffersHolder<int32_t> send_barrier_buffers_holder{
      {world_size}, dev_ctx, pg};
  std::vector<DenseTensor> send_barrier_buffers =
      send_barrier_buffers_holder.get_buffers({world_size});

  static BuffersHolder<int32_t> recv_barrier_buffers_holder{
      {world_size}, dev_ctx, pg};
  std::vector<DenseTensor> recv_barrier_buffers =
      recv_barrier_buffers_holder.get_buffers({world_size});

  static BuffersHolder<int32_t> gemm_barrier_buffers_holder{{2}, dev_ctx, pg};
  std::vector<DenseTensor> gemm_barrier_buffers =
      gemm_barrier_buffers_holder.get_buffers({2});

  // init sync
  static BuffersHolder<int32_t> sync_buffers_holder{{world_size}, dev_ctx, pg};
  std::vector<DenseTensor> sync_buffers =
      sync_buffers_holder.get_buffers({world_size});
  std::vector<int32_t*> sync_buffer_ptrs(world_size, nullptr);

  for (size_t i = 0; i < sync_buffers.size(); i++) {
    sync_buffer_ptrs[i] = static_cast<int32_t*>(sync_buffers[i].data());
  }

  // push-based comm gemm overlap
  DenseTensor sub_a;
  int workspace_idx = 0;
  for (int i = rank + world_size - 1; i >= rank; --i) {
    int id = i % world_size;
    // gemm
    if (!(i == rank + world_size - 1 || i == rank + world_size - 2)) {
      phi::smp::wait_empty(
          gemm_barrier_buffers[rank].data(), workspace_idx, dev_ctx.stream());
    }
    phi::smp::get_submatrix<T>(dev_ctx, a, world_size, id, &sub_a);
    phi::MatmulKernel<T>(
        dev_ctx, sub_a, b, false, transpose_b, &tmp_outs[workspace_idx]);
    if (i != rank + world_size - 1) {
      phi::smp::wait_full_reset(
          recv_barrier_buffers[rank].data(), id, dev_ctx.stream());
      phi::AddKernel<T>(dev_ctx,
                        tmp_outs[workspace_idx],
                        comm_buffers[rank],
                        &tmp_outs[workspace_idx]);
    }
    if (i != rank) {
      phi::smp::set_full(
          send_barrier_buffers[(rank + world_size - 1) % world_size].data(),
          (i + world_size - 1) % world_size,
          dev_ctx.stream());
      phi::smp::set_full(
          gemm_barrier_buffers[rank].data(), workspace_idx, dev_ctx.stream());
    }

    // comm
    if (i != rank) {
      phi::smp::wait_full(gemm_barrier_buffers[rank].data(),
                          workspace_idx,
                          comm_ctx->GetStream());
      phi::smp::wait_full_reset(
          send_barrier_buffers[rank].data(), id, comm_ctx->GetStream());
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyAsync(comm_buffers[(rank + 1) % world_size].data(),
                          tmp_outs[workspace_idx].data(),
                          workspace_size_in_bytes,
                          cudaMemcpyDefault,
                          comm_ctx->GetStream()));
      phi::smp::set_empty(gemm_barrier_buffers[rank].data(),
                          workspace_idx,
                          comm_ctx->GetStream());
      phi::smp::set_full(recv_barrier_buffers[(rank + 1) % world_size].data(),
                         id,
                         comm_ctx->GetStream());
      workspace_idx ^= 1;
    }
  }

  *out = tmp_outs[workspace_idx];

  // reset signals
  phi::smp::cudaipc_barrier_all_on_stream_impl(
      dev_ctx.stream(), sync_buffer_ptrs.data(), rank, world_size);
}

template <typename T, typename Context>
void SMPAllGatherGEMMKernel(const Context& dev_ctx,
                            const DenseTensor& a,
                            const DenseTensor& b,
                            const bool transpose_b,
                            const bool deepcopy_a,
                            const int32_t ring_id,
                            DenseTensor* out,
                            DenseTensor* global_a) {
  VLOG(10) << "SMPAllGatherGEMMKernel";
  // get ProcessGroup and NCCLCommContext
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(
      pg, nullptr, common::errors::Unavailable("ProcessGroup is nullptr."));

  distributed::NCCLCommContext* comm_ctx =
      pg->GetOrCreateCommContext(a.place(), distributed::CommType::ALLGATHER);

  PADDLE_ENFORCE_NE(
      out, nullptr, common::errors::Unavailable("out is nullptr."));

  PADDLE_ENFORCE_NE(
      comm_ctx, nullptr, common::errors::Unavailable("comm_ctx is nullptr."));

  PADDLE_ENFORCE_NE(deepcopy_a && global_a == nullptr,
                    true,
                    common::errors::InvalidArgument(
                        "can not return a when global_a is nullptr"));

  const int32_t a_rank = a.dims().size();
  const int32_t b_rank = b.dims().size();
  PADDLE_ENFORCE_EQ(
      a_rank,
      2,
      common::errors::InvalidArgument(
          "a must be 2-D tensor, but received a %d-D tensor", a_rank));
  PADDLE_ENFORCE_EQ(
      b_rank,
      2,
      common::errors::InvalidArgument(
          "b must be 2-D tensor, but received a %d-D tensor", b_rank));

  const int32_t rank = pg->GetRank();
  const int32_t world_size = pg->GetSize();

  // init comm buffers
  std::vector<int64_t> comm_buffer_shape(a_rank);

  for (int i = 0; i < a_rank; ++i) {
    comm_buffer_shape[i] = a.dims()[i];
  }

  comm_buffer_shape[a_rank - 2] *= world_size;

  // if not low_memory, no need to use double buffer
  static BuffersHolder<T> comm_buffers_holder{comm_buffer_shape, dev_ctx, pg};
  std::vector<DenseTensor> comm_buffers =
      comm_buffers_holder.get_buffers(comm_buffer_shape);

  // init barrier
  static BuffersHolder<int32_t> barrier_buffers_holder{
      {world_size}, dev_ctx, pg};
  std::vector<DenseTensor> barrier_buffers =
      barrier_buffers_holder.get_buffers({world_size});

  // init sync
  static BuffersHolder<int32_t> sync_buffers_holder{{world_size}, dev_ctx, pg};
  std::vector<DenseTensor> sync_buffers =
      sync_buffers_holder.get_buffers({world_size});
  std::vector<int32_t*> sync_buffer_ptrs(world_size, nullptr);

  for (size_t i = 0; i < sync_buffers.size(); i++) {
    sync_buffer_ptrs[i] = static_cast<int32_t*>(sync_buffers[i].data());
  }

  // init cuda event
  constexpr bool disable_timing = true;
  static CUDAEventHolder cp_event_holder{disable_timing};
  static CUDAEventHolder ready_event_holder{disable_timing};

  cudaEvent_t cp_event = cp_event_holder.event;
  cudaEvent_t ready_event = ready_event_holder.event;

  // init out
  const int64_t out_m = a.dims()[a_rank - 2] * world_size;
  const int64_t out_n =
      transpose_b ? b.dims()[b_rank - 2] : b.dims()[b_rank - 1];
  out->Resize(common::make_ddim({out_m, out_n}));
  dev_ctx.template Alloc<T>(out);

  // copy a locally to comm_buffer
  const size_t a_size_in_bytes = a.numel() * SizeOf(a.dtype());

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      ptr_offset(comm_buffers[rank].data(), rank * a_size_in_bytes),
      a.data(),
      a_size_in_bytes,
      cudaMemcpyDefault,
      dev_ctx.stream()));

  smp::cudaipc_barrier_all_on_stream_impl(
      dev_ctx.stream(), sync_buffer_ptrs.data(), rank, world_size);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(ready_event, dev_ctx.stream()));

  smp::set_full(barrier_buffers[rank].data(), rank, dev_ctx.stream());

  // calc out
  DenseTensor sub_out;
  phi::smp::get_submatrix<T>(dev_ctx, *out, world_size, rank, &sub_out);
  DenseTensor sub_a;
  phi::smp::get_submatrix<T>(
      dev_ctx, comm_buffers[rank], world_size, rank, &sub_a);
  phi::MatmulKernel<T>(dev_ctx, sub_a, b, false, transpose_b, &sub_out);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(comm_ctx->GetStream(), ready_event));
  // pull-based ring gemm-comm-overlap
  for (int i = rank + 1; i < (world_size + rank); ++i) {
    int id = i % world_size;
    // comm
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
        ptr_offset(comm_buffers[rank].data(), id * a_size_in_bytes),
        ptr_offset(comm_buffers[id].data(), id * a_size_in_bytes),
        a_size_in_bytes,
        cudaMemcpyDefault,
        comm_ctx->GetStream()));
    phi::smp::set_full(barrier_buffers[rank].data(), id, comm_ctx->GetStream());

    // gemm
    phi::smp::wait_full(barrier_buffers[rank].data(), id, dev_ctx.stream());

    phi::smp::get_submatrix<T>(dev_ctx, *out, world_size, id, &sub_out);
    phi::smp::get_submatrix<T>(
        dev_ctx, comm_buffers[rank], world_size, id, &sub_a);
    phi::MatmulKernel<T>(dev_ctx, sub_a, b, false, transpose_b, &sub_out);
  }

  if (global_a != nullptr) {
    if (deepcopy_a) {
      *global_a = phi::Empty<T>(
          dev_ctx, IntArray{comm_buffer_shape[0], comm_buffer_shape[1]});
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
          global_a->data(),
          comm_buffers[rank].data(),
          sizeof(T) * comm_buffer_shape[0] * comm_buffer_shape[1],
          cudaMemcpyDefault,
          comm_ctx->GetStream()));
    } else {
      *global_a = comm_buffers[rank];
    }
  }

  /// reset signals
  phi::smp::cudaipc_barrier_all_on_stream_impl(
      dev_ctx.stream(), sync_buffer_ptrs.data(), rank, world_size);

  phi::funcs::SetConstant<GPUContext, int32_t> set_zero_int32;
  set_zero_int32(dev_ctx, &(barrier_buffers[rank]), int32_t{0});
}

// SMPColRowRow means x split in col, w split in row, o split in row
// transpose_weight means weight with shape [n, k]
template <typename T, typename Context>
void SMPColRowRowLinear(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& weight,
                        const paddle::optional<DenseTensor>& bias,
                        const bool transpose_weight,
                        const bool low_memory,
                        const int32_t ring_id,
                        DenseTensor* out) {
  VLOG(10) << "SMPColRowRowLinear";
  SMPGEMMReduceScatterKernel<T>(
      dev_ctx, x, weight, transpose_weight, ring_id, out);
  if (bias) {
    phi::AddKernel<T>(dev_ctx, *out, bias.get(), out);
  }
}

template <typename T, typename Context>
void SMPColRowRowLinearGrad(const Context& dev_ctx,
                            const DenseTensor& dy,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const bool low_memory,
                            const bool require_dx,
                            const bool require_dw,
                            const bool require_db,
                            const int32_t ring_id,
                            DenseTensor* dx,
                            DenseTensor* dw,
                            DenseTensor* db) {
  VLOG(10) << "SMPColRowRowLinearGrad";
  PADDLE_ENFORCE_EQ(dy.dims().size(),
                    2,
                    common::errors::InvalidArgument(
                        "dy must be 2-D tensor, but received a %d-D tensor",
                        dy.dims().size()));

  DenseTensor global_dy;
  // calc dx
  if (require_dx) {
    SMPAllGatherGEMMKernel<T>(
        dev_ctx, dy, weight, true, false, ring_id, dx, &global_dy);
  }

  // get comm ctx, prepare for dw and db
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(
      pg, nullptr, common::errors::Unavailable("ProcessGroup is nullptr."));

  distributed::NCCLCommContext* comm_ctx =
      pg->GetOrCreateCommContext(dy.place(), distributed::CommType::ALLGATHER);

  PADDLE_ENFORCE_NE(
      comm_ctx, nullptr, common::errors::Unavailable("comm_ctx is nullptr."));

  // calc dw
  if (require_dw) {
    if (!require_dx) {
      // all gather
      comm_ctx->AllGather(&global_dy, dy, dev_ctx.stream());
    }
    *dw = phi::Matmul<T>(dev_ctx, x, global_dy, true, false);
  }

  // calc db
  if (require_db) {
    DenseTensor local_db = phi::Sum<T>(dev_ctx, dy, {0, 1}, dy.dtype(), false);
    // TODO(umiswing): try overlap
    comm_ctx->AllReduce(db, local_db, ncclSum, dev_ctx.stream());
  } else {
    db->Resize(common::make_ddim({1}));
    dev_ctx.template Alloc<T>(db);
  }
}

template <typename T, typename Context>
void SMPRowColColLinear(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& weight,
                        const paddle::optional<DenseTensor>& bias,
                        const bool transpose_weight,
                        const bool return_x,
                        const int32_t ring_id,
                        DenseTensor* out,
                        DenseTensor* global_x) {
  VLOG(10) << "SMPRowColColLinear";

  // prevent overlap
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  paddle::distributed::ProcessGroup* pg = map->get(ring_id);
  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  distributed::BarrierOptions opts{};
  opts.device_id = device_id;
  pg->Barrier(opts)->Wait();

  SMPAllGatherGEMMKernel<T>(
      dev_ctx, x, weight, transpose_weight, return_x, ring_id, out, global_x);
  if (bias) {
    phi::AddKernel<T>(dev_ctx, *out, bias.get(), out);
  }
}

// Here, we assume always receive global x to simplify
template <typename T, typename Context>
void SMPRowColColLinearGrad(const Context& dev_ctx,
                            const DenseTensor& dy,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const bool require_dx,
                            const bool require_dw,
                            const bool require_db,
                            const int32_t ring_id,
                            DenseTensor* dx,
                            DenseTensor* dw,
                            DenseTensor* db) {
  VLOG(10) << "SMPRowColColLinearGrad";
  // make sure receive global x
  PADDLE_ENFORCE_EQ(
      x.dims()[0],
      dy.dims()[0],
      common::errors::InvalidArgument(
          "x must be global, but received x with %d rows", x.dims()[0]));
  // calc dx
  // dx = reduce_scatter(dy * w^T)
  if (require_dx) {
    SMPGEMMReduceScatterKernel<T>(dev_ctx, dy, weight, true, ring_id, dx);
  }
  // calc dw
  // dw = g_x^T * dy
  if (require_dw) {
    phi::MatmulKernel<T>(dev_ctx, x, dy, true, false, dw);
  }

  // calc db
  if (require_db) {
    phi::SumKernel<T>(dev_ctx, dy, {0, 1}, dy.dtype(), false, db);
  } else {
    db->Resize(common::make_ddim({1}));
    dev_ctx.template Alloc<T>(db);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(smp_col_row_row_linear,
                   GPU,
                   ALL_LAYOUT,
                   phi::SMPColRowRowLinear,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(smp_col_row_row_linear_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SMPColRowRowLinearGrad,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(smp_row_col_col_linear,
                   GPU,
                   ALL_LAYOUT,
                   phi::SMPRowColColLinear,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(smp_row_col_col_linear_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SMPRowColColLinearGrad,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
