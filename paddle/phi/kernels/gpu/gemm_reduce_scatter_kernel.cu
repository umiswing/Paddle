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
#include "paddle/phi/core/utils/intrusive_ptr.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/impl/slice_kernel_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"
#include "paddle/phi/kernels/view_kernel.h"

#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/kernels/gpu/comm_overlap_utils.h"
namespace phi {

// Integration of Flux, a gemm-comm-overlap library as described in the paper
// https://arxiv.org/pdf/2406.06858

#ifdef PADDLE_WITH_FLUX
template <typename InT, typename OutT>
class GemmRSHelper {
 public:
  const phi::GPUContext& dev_ctx;
  paddle::distributed::ProcessGroup* tp_group;
  distributed::NCCLCommContext* comm_ctx;
  const int32_t nnodes;
  const int32_t max_m;
  const int32_t n_dim;
  const phi::DataType input_dtype =
      std::is_same<InT, phi::dtype::float16>::value ? phi::DataType::FLOAT16
                                                    : phi::DataType::BFLOAT16;
  const phi::DataType output_dtype =
      std::is_same<OutT, phi::dtype::float16>::value ? phi::DataType::FLOAT16
                                                     : phi::DataType::BFLOAT16;
  const bool transpose_weight;
  const bool fuse_reduction;

  const int32_t rank;
  const int32_t world_size;
  const int32_t local_world_size;
  const int32_t local_rank;
  const int32_t node_idx;

  // Symmetrically distributed tensor
  std::vector<DenseTensor> output_buffers;   // OutT
  std::vector<DenseTensor> reduce_buffers;   // OutT
  std::vector<DenseTensor> barrier_buffers;  // uint8_t
  // used for the cuda-ipc-barrier
  std::vector<DenseTensor> sync_buffers;  // int32_t
  DenseTensor output_buffer;
  DenseTensor reduce_buffer;
  DenseTensor barrier_buffer;
  DenseTensor gemm_buffer;
  std::vector<void*> output_scatter_ptrs;
  std::vector<void*> barrier_ptrs;
  std::vector<void*> output_buffer_ptrs;
  std::vector<void*> reduce_buffer_ptrs;
  std::vector<void*> barrier_buffer_ptrs;
  std::vector<int32_t*> sync_buffer_ptrs;
  bool no_nvlink;
  int sub_world_size;
  cudaStream_t rs_stream_;
  cudaEvent_t event_;
  bool use_1d_ring;
  bool use_p2p_read;
  const bool is_fp8_gemm{false};

  GemmRSHelper(const phi::GPUContext& dev_ctx,
               paddle::distributed::ProcessGroup* tp_group_,
               distributed::NCCLCommContext* comm_ctx,
               int32_t nnodes,
               int32_t max_m,
               int32_t n_dim,
               bool transpose_weight,
               bool fuse_reduction)
      : dev_ctx(dev_ctx),
        comm_ctx(comm_ctx),
        tp_group(tp_group_),
        nnodes(nnodes),
        max_m(max_m),
        n_dim(n_dim),
        transpose_weight(transpose_weight),
        fuse_reduction(fuse_reduction),
        rank(tp_group->GetRank()),
        world_size(tp_group->GetSize()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        node_idx(rank / local_world_size),
        output_scatter_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr),
        output_buffer_ptrs(world_size, nullptr),
        reduce_buffer_ptrs(world_size, nullptr),
        barrier_buffer_ptrs(world_size, nullptr),
        sync_buffer_ptrs(world_size, nullptr),
        no_nvlink(!has_nvlink()),
        rs_stream_(CreateReduceScatterStream()),  // private stream. never dup
                                                  // with gemm stream
        use_1d_ring(use_1d_ring_or_not()),
        use_p2p_read(use_p2p_read_or_not()) {
    PADDLE_ENFORCE(rank >= 0 && rank < world_size,
                   "invalid rank: %d and world_size: %d",
                   rank,
                   world_size);
    PADDLE_ENFORCE(world_size % nnodes == 0,
                   "invalid nnodes: world_size[%d] % nnodes[%d] !=0",
                   world_size,
                   nnodes);
    PADDLE_ENFORCE(
        !fuse_reduction || this->input_dtype == phi::DataType::FLOAT16,
        "Fuse reduction only support float16 type on SM80 due to instruction "
        "limitation.");
    static CUDAEventHolder event_holder{};
    this->event_ = event_holder.event;
  }

  ~GemmRSHelper() {}

  bool has_nvlink() { return true; }

  bool use_1d_ring_or_not() {
    phi::dynload::ensure_nvml_init_capi();
    int devid = phi::backends::gpu::GetCurrentDeviceId();
    std::string devname(phi::dynload::get_gpu_device_name_capi(devid));
    if (devname != "NVIDIA L20" && world_size == 8) {
      return false;
    }
    return true;
  }

  bool use_p2p_read_or_not() {
    phi::dynload::ensure_nvml_init_capi();
    int devid = phi::backends::gpu::GetCurrentDeviceId();
    std::string devname(phi::dynload::get_gpu_device_name_capi(devid));
    if (devname != "NVIDIA L20") {
      return true;
    }
    return false;
  }

  void init_output_buffer() {
    // update max_m and allocate buffer
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    int sm_version = backends::gpu::GetGPUComputeCapability(device_id);
    if (sm_version == 90 || no_nvlink || (sm_version == 80 && nnodes > 1)) {
      int reduce_m_dim = (sm_version == 90) ? (max_m + world_size - 1) /
                                                  world_size * nnodes * nnodes
                                            : max_m;
      static BuffersHolder<OutT> reduce_buffers_holder{
          {max_m, n_dim}, dev_ctx, tp_group};

      reduce_buffers = reduce_buffers_holder.get_buffers({max_m, n_dim});
      reduce_buffer = reduce_buffers[local_rank];
    }
    static BuffersHolder<OutT> output_buffers_holder{
        {max_m, n_dim}, dev_ctx, tp_group};
    if (sm_version == 80 && nnodes > 1 &&
        input_dtype == phi::DataType::BFLOAT16) {
      // SM80 does not support the fuse reduction for the bfloat16 data type
      // we have to use the float32 global_red instruction when SM80 && nnodes>1
      // && input_type=bf16 Therefore, in this case, here double the size of the
      // output_buffer.
      output_buffers = output_buffers_holder.get_buffers({max_m * 2, n_dim});
    } else {
      output_buffers = output_buffers_holder.get_buffers({max_m, n_dim});
    }
    output_buffer = output_buffers[local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / local_world_size == rank / local_world_size) {
        output_scatter_ptrs[i] = output_buffers[i % local_world_size].data();
        // only check for ranks on the same node
        PADDLE_ENFORCE_NOT_NULL(
            output_scatter_ptrs[i],
            common::errors::InvalidArgument("nullptr buffr of rank " +
                                            std::to_string(i)));
      } else {
        output_scatter_ptrs[i] = nullptr;
      }
    }
    for (size_t i = 0; i < reduce_buffers.size(); i++) {
      reduce_buffer_ptrs[i] = reduce_buffers[i].data();
    }
    for (size_t i = 0; i < output_buffers.size(); i++) {
      output_buffer_ptrs[i] = output_buffers[i].data();
    }
    static BuffersHolder<int32_t> sync_buffers_holder{
        {this->world_size}, dev_ctx, tp_group};
    this->sync_buffers = sync_buffers_holder.get_buffers({this->world_size});
    for (size_t i = 0; i < sync_buffers.size(); i++) {
      sync_buffer_ptrs[i] = static_cast<int32_t*>(sync_buffers[i].data());
    }
  }

  void lazy_init_barrier_buffer(int64_t buffer_size) {
    if (buffer_size == 0) {
      return;
    }

    static BuffersHolder<uint8_t> barrier_buffers_holder{
        {buffer_size}, dev_ctx, tp_group};
    barrier_buffers = barrier_buffers_holder.get_buffers({buffer_size});

    for (size_t i = 0; i < barrier_buffers.size(); i++) {
      barrier_buffer_ptrs[i] = barrier_buffers[i].data();
    }
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        barrier_ptrs[i] = barrier_buffers[i % this->local_world_size].data();
        // only check for ranks on the same node
        PADDLE_ENFORCE_NOT_NULL(
            barrier_ptrs[i],
            common::errors::InvalidArgument("nullptr buffr of rank " +
                                            std::to_string(i)));
      } else {
        barrier_ptrs[i] = nullptr;
      }
    }
  }

  void lazy_init_gemm_buffer(int64_t buffer_size) {
    if (buffer_size <= 0) {
      return;
    }
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!gemm_buffer.initialized() || buffer_size > gemm_buffer.numel()) {
      gemm_buffer = phi::Empty<uint8_t>(dev_ctx, {buffer_size});
    }
  }

  cudaStream_t CreateReduceScatterStream() {
    return this->comm_ctx->GetStream();
  }

  void flux_barrier_all_on_stream() {
    std::vector<int32_t*> sync_buffer_ptrs;

    int world_size = sync_buffers.size();
    for (size_t i = 0; i < sync_buffers.size(); i++) {
      sync_buffer_ptrs.push_back(
          reinterpret_cast<int32_t*>(sync_buffers[i].data()));
    }
    phi::dynload::cudaipc_barrier_all_on_stream_impl_capi(
        dev_ctx.stream(), sync_buffer_ptrs.data(), rank, world_size);
  }
};
#endif

template <typename T, typename Context>
void GemmReduceScatterKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const DenseTensor& weight,
                             const paddle::optional<DenseTensor>& bias,
                             const paddle::optional<DenseTensor>& input_scale,
                             const paddle::optional<DenseTensor>& weight_scale,
                             const paddle::optional<DenseTensor>& output_scale,
                             const int32_t nnodes,
                             const int32_t max_m,
                             const int32_t n_dim,
                             const bool transpose_weight,
                             const bool fuse_reduction,
                             const bool check_can_implement,
                             const int ring_id,
                             const int nranks,
                             DenseTensor* output) {
#ifdef PADDLE_WITH_FLUX
  int sm_version =
      backends::gpu::GetGPUComputeCapability(dev_ctx.GetPlace().GetDeviceId());

  if (sm_version != 90 && sm_version != 80) {
    flux::RaiseNotSupportedError();
  }

  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      common::errors::InvalidArgument(
          "The ring_id (%d) for c_scatter_op must be non-negative.", ring_id));

  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(
      pg, nullptr, common::errors::Unavailable("ProcessGroup is nullptr."));

  distributed::NCCLCommContext* comm_ctx = pg->GetOrCreateCommContext(
      input.place(), distributed::CommType::REDUCE_SCATTER);

  PADDLE_ENFORCE_NE(
      comm_ctx, nullptr, common::errors::Unavailable("comm_ctx is nullptr."));

  static int num_blocks = 12;
  static bool use_barrier_queue = false;
  static bool use_gemmk = false;
  static bool use_cudaMemcpyAsync = false;
  static int n_split = 1;
  static bool per_tile_flags = false;
  const bool fast_accum = false;

  GemmRSHelper<T, T> helper{dev_ctx,
                            pg,
                            comm_ctx,
                            nnodes,
                            max_m,
                            n_dim,
                            transpose_weight,
                            fuse_reduction};

  const int32_t m = input.dims()[0];
  const int32_t k = input.dims()[1];
  const int32_t n = transpose_weight ? weight.dims()[1] : weight.dims()[0];
  const int32_t wk = transpose_weight ? weight.dims()[0] : weight.dims()[1];

  auto launcher = [&](const bool get_workspace_size_flag,
                      const bool get_barrier_workspace_size) -> size_t {
    return phi::dynload::gemm_rs(
        input.data(),
        weight.data(),
        bias.is_initialized() ? bias->data() : nullptr,
        input_scale.is_initialized() ? input_scale->data() : nullptr,
        weight_scale.is_initialized() ? weight_scale->data() : nullptr,
        output_scale.is_initialized() ? output_scale->data() : nullptr,
        helper.gemm_buffer.initialized() ? helper.gemm_buffer.data() : nullptr,
        helper.reduce_buffer_ptrs.data(),
        helper.output_scatter_ptrs.data(),
        helper.barrier_ptrs.data(),
        helper.output_buffer_ptrs.data(),
        helper.barrier_buffer_ptrs.data(),
        helper.sync_buffer_ptrs.data(),
        m,
        n,
        k,
        wk,
        helper.nnodes,
        helper.max_m,
        helper.n_dim,
        helper.rank,
        helper.world_size,
        helper.local_world_size,
        helper.local_rank,
        helper.node_idx,
        num_blocks,
        n_split,
        fast_accum,
        std::is_same<T, phi::dtype::bfloat16>::value,
        helper.transpose_weight,
        helper.fuse_reduction,
        helper.use_1d_ring,
        helper.use_p2p_read,
        helper.is_fp8_gemm,
        use_barrier_queue,
        use_gemmk,
        use_cudaMemcpyAsync,
        per_tile_flags,
        helper.no_nvlink,
        get_workspace_size_flag,
        get_barrier_workspace_size,
        check_can_implement,
        dev_ctx.stream(),
        helper.rs_stream_,
        helper.event_);
  };

  if (check_can_implement) {
    auto can_implement = [&]() -> bool {
      size_t result = launcher(false, false);
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

  helper.init_output_buffer();

  auto get_workspace_size = [&]() -> size_t { return launcher(true, false); };

  auto get_barrier_workspace_size = [&]() -> size_t {
    return launcher(false, true);
  };

  size_t workspace_size = get_workspace_size();
  size_t barrier_workspace_size = get_barrier_workspace_size();

  helper.lazy_init_gemm_buffer(workspace_size);
  helper.lazy_init_barrier_buffer(barrier_workspace_size);

  auto gemm_rs = [&]() { launcher(false, false); };

  gemm_rs();

  // reduce impl
  if (sm_version == 90) {
    DenseTensor output_3d;
    phi::ViewShapeKernel(dev_ctx,
                         helper.reduce_buffer,
                         {helper.world_size, m / helper.world_size, n},
                         &output_3d);

    output->Resize(common::make_dim(m / helper.world_size, n));
    dev_ctx.template Alloc<T>(output);
    phi::SumKernel<T>(
        dev_ctx, output_3d, {0}, helper.output_dtype, false, output);
  } else if (sm_version == 80) {
    helper.flux_barrier_all_on_stream();
    DenseTensor output_3d;
    phi::ViewShapeKernel(dev_ctx,
                         helper.output_buffer,
                         {helper.world_size, m / helper.world_size, n},
                         &output_3d);

    output->Resize(common::make_dim(m / helper.world_size, n));
    dev_ctx.template Alloc<T>(output);
    phi::SumKernel<T>(
        dev_ctx, output_3d, {0}, helper.output_dtype, false, output);
  }
#else
  flux::RaiseNotSupportedError();
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(gemm_reduce_scatter,
                   GPU,
                   ALL_LAYOUT,
                   phi::GemmReduceScatterKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
