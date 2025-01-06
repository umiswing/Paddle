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

#include "paddle/phi/kernels/matmul_kernel.h"

#include "paddle/phi/kernels/elementwise_add_kernel.h"

namespace phi {

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
                        DenseTensor* output) {
  VLOG(10) << "SMPColRowRowLinear";
  PADDLE_ENFORCE(x.dims().size() == 2, "x must be 2-D tensor");
  PADDLE_ENFORCE(weight.dims().size() == 2, "weight must be 2-D tensor");
  // get ProcessGroup and NCCLCommContext
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(
      pg, nullptr, common::errors::Unavailable("ProcessGroup is nullptr."));

  distributed::NCCLCommContext* comm_ctx =
      pg->GetOrCreateCommContext(x.place(), distributed::CommType::ALLGATHER);

  PADDLE_ENFORCE_NE(
      comm_ctx, nullptr, common::errors::Unavailable("comm_ctx is nullptr."));

  int32_t rank = pg->GetRank();
  int32_t world_size = pg->GetSize();

  // init workspace (tmp output)
  int64_t global_m = x.dims()[0];
  int64_t global_n = transpose_weight ? weight.dims()[0] : weight.dims()[1];
  PADDLE_ENFORCE(global_m % world_size == 0,
                 "m needs to be divisible by world size");
  std::vector<DenseTensor> tmp_outputs(2);
  for (int i = 0; i < 2; i++) {
    tmp_outputs[i].Resize(common::make_dim(global_m / world_size, global_n));
    dev_ctx.template Alloc<T>(&(tmp_outputs[i]));
  }

  size_t workspace_size_in_bytes = tmp_outputs[0].numel() * sizeof(T);

  // init comm buffers
  std::vector<int64_t> comm_buffer_shape{global_m / world_size, global_n};
  // std::vector<int64_t> comm_buffer_shape{global_m, global_n};
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
  DenseTensor sub_x;
  int workspace_idx = 0;
  for (int i = rank + world_size - 1; i >= rank; --i) {
    int id = i % world_size;
    // gemm
    if (!(i == rank + world_size - 1 || i == rank + world_size - 2)) {
      phi::smp::wait_empty(
          gemm_barrier_buffers[rank].data(), workspace_idx, dev_ctx.stream());
    }
    phi::smp::get_submatrix<T>(dev_ctx, x, world_size, id, &sub_x);
    phi::MatmulKernel<T>(dev_ctx,
                         sub_x,
                         weight,
                         false,
                         transpose_weight,
                         &tmp_outputs[workspace_idx]);
    if (i != rank + world_size - 1) {
      phi::smp::wait_full_reset(
          recv_barrier_buffers[rank].data(), id, dev_ctx.stream());
      phi::AddKernel<T>(dev_ctx,
                        tmp_outputs[workspace_idx],
                        comm_buffers[rank],
                        &tmp_outputs[workspace_idx]);
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
                          tmp_outputs[workspace_idx].data(),
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

  *output = tmp_outputs[workspace_idx];

  // reset signals
  phi::smp::cudaipc_barrier_all_on_stream_impl(
      dev_ctx.stream(), sync_buffer_ptrs.data(), rank, world_size);
}

template <typename T, typename Context>
void SMPColRowRowLinearGrad(const Context& dev_ctx,
                            const DenseTensor& dy,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const paddle::optional<DenseTensor>& bias,
                            const bool low_memory,
                            const int32_t ring_id,
                            DenseTensor* dw,
                            DenseTensor* dx) {
  VLOG(10) << "SMPColRowRowLinearGrad";
  // get ProcessGroup and NCCLCommContext
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(
      pg, nullptr, common::errors::Unavailable("ProcessGroup is nullptr."));

  distributed::NCCLCommContext* comm_ctx =
      pg->GetOrCreateCommContext(dy.place(), distributed::CommType::ALLGATHER);

  PADDLE_ENFORCE_NE(dx, nullptr, common::errors::Unavailable("dx is nullptr."));

  PADDLE_ENFORCE_NE(
      comm_ctx, nullptr, common::errors::Unavailable("comm_ctx is nullptr."));

  int32_t rank = pg->GetRank();
  int32_t world_size = pg->GetSize();

  // init comm buffers
  int32_t dy_rank = dy.dims().size();
  std::vector<int64_t> comm_buffer_shape(dy_rank);

  for (int i = 0; i < dy_rank; ++i) {
    comm_buffer_shape[i] = dy.dims()[i];
  }

  if (!low_memory) {
    comm_buffer_shape[dy_rank - 2] *= world_size;
  }

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

  // init dx
  dx->Resize(x.dims());
  dev_ctx.template Alloc<T>(dx);

  // copy dy locally to comm_buffer0
  const size_t dy_size_in_bytes = dy.numel() * SizeOf(dy.dtype());

  int32_t double_buffer_idx = 0;

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      ptr_offset(comm_buffers[rank].data(), rank * dy_size_in_bytes),
      dy.data(),
      dy_size_in_bytes,
      cudaMemcpyDefault,
      dev_ctx.stream()));

  smp::cudaipc_barrier_all_on_stream_impl(
      dev_ctx.stream(), sync_buffer_ptrs.data(), rank, world_size);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(ready_event, dev_ctx.stream()));

  smp::set_full(barrier_buffers[rank].data(), rank, dev_ctx.stream());

  // calc dx
  DenseTensor sub_dx;
  phi::smp::get_submatrix<T>(dev_ctx, *dx, world_size, rank, &sub_dx);
  DenseTensor sub_dy;
  phi::smp::get_submatrix<T>(
      dev_ctx, comm_buffers[rank], world_size, rank, &sub_dy);
  phi::MatmulKernel<T>(dev_ctx, sub_dy, weight, false, true, &sub_dx);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(comm_ctx->GetStream(), ready_event));
  // pull-based ring gemm-comm-overlap
  for (int i = rank + 1; i < (world_size + rank); ++i) {
    int id = i % world_size;
    // comm
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
        ptr_offset(comm_buffers[rank].data(), id * dy_size_in_bytes),
        ptr_offset(comm_buffers[id].data(), id * dy_size_in_bytes),
        dy_size_in_bytes,
        cudaMemcpyDefault,
        comm_ctx->GetStream()));
    phi::smp::set_full(barrier_buffers[rank].data(), id, comm_ctx->GetStream());

    // gemm
    phi::smp::wait_full(barrier_buffers[rank].data(), id, dev_ctx.stream());

    phi::smp::get_submatrix<T>(dev_ctx, *dx, world_size, rank, &sub_dx);
    phi::smp::get_submatrix<T>(
        dev_ctx, comm_buffers[rank], world_size, rank, &sub_dy);
    phi::MatmulKernel<T>(dev_ctx, sub_dy, weight, false, true, &sub_dx);
  }

  // calc dw
  if (dw != nullptr) {
    *dw = phi::Matmul<T>(dev_ctx, x, comm_buffers[rank], true, false);
  }

  /// reset signals
  phi::smp::cudaipc_barrier_all_on_stream_impl(
      dev_ctx.stream(), sync_buffer_ptrs.data(), rank, world_size);

  phi::funcs::SetConstant<GPUContext, int32_t> set_zero_int32;
  set_zero_int32(dev_ctx, &(barrier_buffers[rank]), int32_t{0});
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
