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

#ifdef PADDLE_WITH_FLUX
#include "paddle/phi/backends/dynload/flux.h"
#endif
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

namespace phi {

// Integration of Flux, a gemm-comm-overlap library as described in the paper
// https://arxiv.org/pdf/2406.06858

#ifdef PADDLE_WITH_FLUX
template <typename InT, typename OutT>
class AGGemmHelper {
 public:
  static constexpr int MAX_NUM_SIGNAL = 64;

  using FlagType = int32_t;
  const phi::GPUContext& dev_ctx;
  distributed::NCCLCommContext* comm_ctx;
  paddle::distributed::ProcessGroup* tp_group;
  int32_t nnodes;
  int32_t full_m;
  int32_t n_dim;
  int32_t k_dim;
  const phi::DataType input_dtype =
      std::is_same<InT, phi::dtype::float16>::value ? phi::DataType::FLOAT16
                                                    : phi::DataType::BFLOAT16;
  const phi::DataType output_dtype =
      std::is_same<OutT, phi::dtype::float16>::value ? phi::DataType::FLOAT16
                                                     : phi::DataType::BFLOAT16;
  const bool transpose_weight;
  const bool is_fp8_gemm{false};
  int32_t rank;
  int32_t world_size;
  int32_t local_world_size;
  int32_t local_rank;
  // used for the cuda-ipc-barrier
  std::vector<DenseTensor> sync_buffers;  // int32_t
  std::vector<int32_t*> sync_buffer_ptrs;
  std::vector<DenseTensor> input_buffers;    // InT
  std::vector<DenseTensor> output_buffers;   // OutT
  std::vector<DenseTensor> barrier_buffers;  // int32_t

  std::vector<void*> input_buffer_ptrs;
  std::vector<void*> output_buffer_ptrs;
  std::vector<void*> barrier_buffer_ptrs;

  DenseTensor input_buffer;
  DenseTensor output_buffer;
  DenseTensor barrier_buffer;

  std::vector<void*> input_ptrs;
  std::vector<FlagType*> barrier_ptrs;
  AGRingMode ring_mode;
  DenseTensor gemm_buffer;
  size_t chunk_size;
  size_t split_chunk_size;

  int num_cp_streams;
  std::vector<cudaStream_t> cp_streams;

  cudaEvent_t cp_event;
  cudaEvent_t ready_event;

  AGGemmHelper(const phi::GPUContext& dev_ctx_,
               paddle::distributed::ProcessGroup* tp_group_,
               distributed::NCCLCommContext* comm_ctx_,
               int32_t nnodes,
               int32_t full_m,
               int32_t n_dim,
               int32_t k_dim,
               bool transpose_weight = true,
               AGRingMode ring_mode_ = AGRingMode::Auto)
      : dev_ctx(dev_ctx_),
        comm_ctx(comm_ctx_),
        tp_group(tp_group_),
        nnodes(nnodes),
        full_m(full_m),
        n_dim(n_dim),
        k_dim(k_dim),
        transpose_weight(transpose_weight),
        rank(tp_group->GetRank()),
        world_size(tp_group->GetSize()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        input_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr),
        input_buffer_ptrs(world_size, nullptr),
        output_buffer_ptrs(world_size, nullptr),
        barrier_buffer_ptrs(world_size, nullptr),
        sync_buffer_ptrs(world_size, nullptr) {
    PADDLE_ENFORCE(rank >= 0 && rank < world_size,
                   "invalid rank: %d and world_size: %d",
                   rank,
                   world_size);
    PADDLE_ENFORCE(world_size % nnodes == 0,
                   "invalid nnodes: world_size[%d] % nnodes[%d] !=0",
                   world_size,
                   nnodes);
    PADDLE_ENFORCE(!(transpose_weight == true && is_fp8_gemm == true),
                   "FP8 GEMM does not support transpose weight");
    this->ring_mode = get_ring_mode(ring_mode_);

    // copy stream
    this->num_cp_streams = 1;
    for (int i = 0; i < this->num_cp_streams; ++i) {
      this->cp_streams.push_back(this->comm_ctx->GetStream());
    }
    // create events
    constexpr bool disable_timing = true;
    static CUDAEventHolder cp_event_holder{disable_timing};
    static CUDAEventHolder ready_event_holder{disable_timing};

    this->cp_event = cp_event_holder.event;
    this->ready_event = ready_event_holder.event;
  }

  void init_buffer() {
    // input buffer
    static BuffersHolder<InT> input_buffers_holder{
        {full_m, k_dim}, dev_ctx, tp_group};
    this->input_buffers = input_buffers_holder.get_buffers({full_m, k_dim});
    this->input_buffer = this->input_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        // on the same node
        this->input_ptrs[i] = this->input_buffers[i].data();
      } else {
        this->input_ptrs[i] = nullptr;
      }
    }

    int num_signals = MAX_NUM_SIGNAL;
    static BuffersHolder<int32_t> barrier_buffers_holder{
        {num_signals}, dev_ctx, tp_group};
    this->barrier_buffers = barrier_buffers_holder.get_buffers({num_signals});
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        // on the same node
        this->barrier_ptrs[i] =
            static_cast<FlagType*>(this->barrier_buffers[i].data());
      } else {
        this->barrier_ptrs[i] = static_cast<FlagType*>(nullptr);
      }
    }

    static BuffersHolder<int32_t> sync_buffers_holder{
        {this->world_size}, dev_ctx, tp_group};
    this->sync_buffers = sync_buffers_holder.get_buffers({this->world_size});
    for (size_t i = 0; i < this->sync_buffers.size(); i++) {
      this->sync_buffer_ptrs[i] =
          static_cast<int32_t*>(this->sync_buffers[i].data());
    }
  }

  void lazy_init_gemm_buffer(int64_t buffer_size) {
    if (buffer_size <= 0) return;
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.initialized() ||
        buffer_size > this->gemm_buffer.numel()) {
      this->gemm_buffer = phi::Empty<uint8_t>(dev_ctx, {buffer_size});
    }
  }

  ~AGGemmHelper() {}
};
#endif

template <typename T, typename Context>
void AllGatherGemmKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& weight,
                         const paddle::optional<DenseTensor>& bias,
                         const paddle::optional<DenseTensor>& input_scale,
                         const paddle::optional<DenseTensor>& weight_scale,
                         const paddle::optional<DenseTensor>& output_scale,
                         const int32_t nnodes,
                         const int32_t full_m,
                         const int32_t n_dim,
                         const int32_t k_dim,
                         const int32_t ring_id,
                         const bool fast_accum,
                         const bool deepcopy_input_parallel,
                         const bool transpose_weight,
                         const bool check_can_implement,
                         DenseTensor* output,
                         DenseTensor* input_parallel) {
#ifdef PADDLE_WITH_FLUX
  int sm_version =
      backends::gpu::GetGPUComputeCapability(dev_ctx.GetPlace().GetDeviceId());

  if (sm_version != 90 && sm_version != 80) {
    flux::RaiseNotSupportedError();
  }

  constexpr int SPLIT = 1;

  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(
      pg, nullptr, common::errors::Unavailable("ProcessGroup is nullptr."));

  distributed::NCCLCommContext* comm_ctx = pg->GetOrCreateCommContext(
      input.place(), distributed::CommType::ALLGATHER);

  PADDLE_ENFORCE_NE(
      comm_ctx, nullptr, common::errors::Unavailable("comm_ctx is nullptr."));

  const int32_t n = transpose_weight ? weight.dims()[1] : weight.dims()[0];
  const int32_t k = transpose_weight ? weight.dims()[0] : weight.dims()[1];
  AGGemmHelper<T, T> helper{dev_ctx,
                            pg,
                            comm_ctx,
                            nnodes,
                            full_m,
                            n_dim,
                            k_dim,
                            transpose_weight,
                            AGRingMode::Auto};

  helper.n_dim = n_dim;

  helper.chunk_size = input.numel() * SizeOf(input.dtype());
  helper.split_chunk_size = helper.chunk_size / SPLIT;

  static bool kDebugRunGemm = true;
  static bool kPushMode = true;
  DenseTensor output_buffer = phi::Empty<T>(dev_ctx, IntArray{full_m, n_dim});

  auto launcher = [&](const bool return_workspace_size) -> size_t {
    return phi::dynload::ag_gemm(
        const_cast<void*>(input.data()),
        helper.input_buffer.initialized() ? helper.input_buffer.data()
                                          : nullptr,
        const_cast<void*>(weight.data()),
        bias.is_initialized() ? const_cast<void*>(bias->data()) : nullptr,
        output_buffer.data(),
        helper.barrier_buffer.initialized() ? helper.barrier_buffer.data()
                                            : nullptr,
        helper.gemm_buffer.initialized() ? helper.gemm_buffer.data() : nullptr,
        dev_ctx.stream(),
        helper.ready_event,
        n,
        k,
        helper.n_dim,
        helper.k_dim,
        input.dims()[0],
        helper.rank,
        helper.world_size,
        helper.nnodes,
        static_cast<int>(helper.ring_mode),
        std::is_same<T, phi::dtype::bfloat16>::value,
        kDebugRunGemm,
        helper.transpose_weight,
        fast_accum,
        return_workspace_size,
        check_can_implement);
  };

  if (check_can_implement) {
    auto can_implement = [&]() -> bool {
      size_t result = launcher(false);
      if (result == 1)
        return true;
      else
        return false;
    };
    bool result = can_implement();
    bool* out_data = dev_ctx.template HostAlloc<bool>(output);
    out_data[0] = result;
    return;
  }

  helper.init_buffer();

  auto get_workspace_size = [&]() -> size_t { return launcher(true); };
  int64_t workspace_size = get_workspace_size();
  helper.lazy_init_gemm_buffer(workspace_size);

  auto ag_gemm = [&]() { launcher(false); };

  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  distributed::BarrierOptions opts{};
  opts.device_id = device_id;
  pg->Barrier(opts)->Wait();

  /// AG GEMM
  cudaStream_t current_stream = dev_ctx.stream();

  // copy_local
  helper.chunk_size = input.numel() * SizeOf(input.dtype());
  helper.split_chunk_size = helper.chunk_size / SPLIT;
  const void* input_ptr = input.data();
  void* input_buffer_ptr = helper.input_buffer.data();

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      ptr_offset(input_buffer_ptr, helper.rank * helper.chunk_size),
      input_ptr,
      helper.chunk_size,
      cudaMemcpyDefault,
      current_stream));

  for (int j = 0; j < SPLIT; ++j) {
    phi::dynload::set_ready(
        helper.barrier_ptrs[helper.rank], helper.rank, j, dev_ctx.stream());
  }
  phi::dynload::cudaipc_barrier_all_on_stream_impl_capi(
      current_stream,
      helper.sync_buffer_ptrs.data(),
      helper.rank,
      helper.world_size);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(helper.ready_event, current_stream));

  if (helper.ring_mode == AGRingMode::All2All) {
    /// All2All algorithm
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamWaitEvent(helper.cp_streams[0], helper.ready_event));

    for (int i = helper.rank + 1; i < (helper.world_size + helper.rank); ++i) {
      auto id = i % helper.world_size;
      for (int j = 0; j < SPLIT; ++j) {
        auto split_offset = j * helper.split_chunk_size;
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyAsync(ptr_offset(helper.input_ptrs[helper.rank],
                                       id * helper.chunk_size + split_offset),
                            ptr_offset(helper.input_ptrs[id],
                                       id * helper.chunk_size + split_offset),
                            helper.split_chunk_size,
                            cudaMemcpyDefault,
                            helper.cp_streams[0]));
        phi::dynload::set_ready(
            helper.barrier_ptrs[helper.rank], id, j, helper.cp_streams[0]);
      }
    }
  }

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(helper.cp_event, helper.cp_streams[0]));

  ag_gemm();

  PADDLE_ENFORCE(output_buffer.meta().is_contiguous(),
                 "output_buffer is not contiguous");
  MetaTensor meta_output(output);
  std::vector<int64_t> infer_flags = {1};
  std::vector<int64_t> decrease_axis = {};
  SliceRawInferMeta(output_buffer,
                    {0},
                    {0},
                    {static_cast<int32_t>(input.dims()[0] * helper.world_size)},
                    infer_flags,
                    decrease_axis,
                    &meta_output);
  phi::SliceStridedKernel<Context>(
      dev_ctx,
      output_buffer,
      {0},
      {0},
      {static_cast<int32_t>(input.dims()[0] * helper.world_size)},
      infer_flags,
      decrease_axis,
      output);
  if (!output->meta().is_contiguous()) {
    phi::SliceKernel<T, Context>(
        dev_ctx,
        output_buffer,
        {0},
        {0},
        {static_cast<int32_t>(input.dims()[0] * helper.world_size)},
        infer_flags,
        decrease_axis,
        output);
  }

  if (!deepcopy_input_parallel) {
    *input_parallel = helper.input_buffer;
  } else {
    *input_parallel = phi::Empty<T>(dev_ctx, IntArray{full_m, k_dim});
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(input_parallel->data(),
                                               helper.input_buffer.data(),
                                               sizeof(T) * full_m * k_dim,
                                               cudaMemcpyDefault,
                                               helper.cp_streams[0]));
  }

  /// reset signals
  cudaStreamWaitEvent(current_stream, helper.cp_event);
  phi::dynload::cudaipc_barrier_all_on_stream_impl_capi(
      current_stream,
      helper.sync_buffer_ptrs.data(),
      helper.rank,
      helper.world_size);

  phi::funcs::SetConstant<GPUContext, int32_t> set_functor;
  set_functor(dev_ctx, &(helper.barrier_buffer), int32_t{0});
#else
  flux::RaiseNotSupportedError();
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(all_gather_gemm,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllGatherGemmKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
