#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/common/data_type.h"

#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/backends/dynload/flux.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
// #include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/impl/slice_kernel_impl.h"

#include "paddle/phi/kernels/gpu/flux_utils.h"

namespace phi {

template<typename BufferT>
void print_address(std::vector<DenseTensor>&sync_buffers, std::vector<int32_t*>&sync_buffer_ptrs,
                   BuffersHolder<BufferT> holder, int32_t rank) {
  std::cout << "\nprint_address" << std::endl;
  std::cout << sync_buffers[rank].data() << " ,"
            << sync_buffer_ptrs[rank] << " ,"
            << holder.ptr << " ,"
            << holder.local_buffer.data() << " ,"
            << holder.ptrs[rank] << std::endl;

  std::cout << "\nprint sync_buffer_ptrs" << std::endl;
  for(auto& d_ptr : sync_buffer_ptrs) {
    std::cout << d_ptr << " ,";
  }
  std::cout << std::endl;
}

template<typename BufferT>
void write_ptrs(std::vector<DenseTensor>& buffers, const phi::GPUContext& dev_ctx) {
  phi::funcs::SetConstant<GPUContext, BufferT> set_func;

  for(auto& x : buffers) {
    set_func(dev_ctx, &(x), static_cast<BufferT>(67));
  }

}

template<typename BufferT>
void print_ptrs(const std::string& s, std::vector<BufferT*>& ptrs, int64_t numel) {
  BufferT x;

  std::cout << "\nprint_ptrs: " << s << "\n" << std::endl;

  bool all_zero = true;

  for(auto& d_ptr : ptrs) {
    for(int i=0;i<numel;i++) {
      cudaMemcpy(&x,static_cast<BufferT*>(d_ptr)+i,sizeof(BufferT),cudaMemcpyDeviceToHost);
      // if(x!=static_cast<BufferT>(0)) {
      //  all_zero = false;
        std::cout << " ," << x;
      // }
    }
  }
#if 0
  if(all_zero) {
    std::cout << "all zero!";
  }
#endif
  std::cout << std::endl;

}

template<typename InT, typename OutT>
class AGGemmHelper {
public:
  static constexpr int MAX_NUM_SIGNAL = 64;

  using FlagType = int32_t;
  const phi::GPUContext& dev_ctx;
  const phi::GPUContext* comm_ctx;
  paddle::distributed::ProcessGroup* tp_group;
  int32_t nnodes;
  int32_t full_m;
  int32_t n_dim;
int32_t k_dim;
  const phi::DataType input_dtype = std::is_same<InT, phi::dtype::float16>::value ?
                                    phi::DataType::FLOAT16 :
                                    phi::DataType::BFLOAT16;
  const phi::DataType output_dtype = std::is_same<OutT, phi::dtype::float16>::value ?
                                     phi::DataType::FLOAT16 :
                                     phi::DataType::BFLOAT16;
  const bool transpose_weight;
  const bool local_copy;
const bool is_fp8_gemm{false};
int32_t rank;
int32_t world_size;
int32_t local_world_size;
int32_t local_rank;
#ifndef FLUX_SHM_USE_NVSHMEM
  // used for the cuda-ipc-barrier
  std::vector<DenseTensor> sync_buffers; // int32_t
  std::vector<int32_t *> sync_buffer_ptrs;
#endif
  std::vector<DenseTensor> input_buffers; // InT
  std::vector<DenseTensor> output_buffers; // OutT
  std::vector<DenseTensor> barrier_buffers; //UINT8 (c10::ScalarType::Byte)

  std::vector<void *> input_buffer_ptrs;
  std::vector<void *> output_buffer_ptrs;
  std::vector<void *> barrier_buffer_ptrs;

  DenseTensor input_buffer;
  DenseTensor output_buffer;
  DenseTensor barrier_buffer;

  std::vector<void *> input_ptrs;
  std::vector<FlagType *> barrier_ptrs;
  AGRingMode ring_mode;
  // std::any gemm_args; // .so side
  DenseTensor gemm_buffer;
  // void *workspace;
  //std::unique_ptr<GemmOperatorBase> cutlass_op; // .so side
  size_t chunk_size;
  size_t split_chunk_size;

  int num_cp_streams;
  std::vector<cudaStream_t> cp_streams;
  std::vector<cudaStream_t> reset_streams;

  cudaEvent_t cp_event;
  cudaEvent_t ready_event;
  cudaEvent_t all_gather_event;

  AGGemmHelper(
    const phi::GPUContext& dev_ctx_,
    paddle::distributed::ProcessGroup* tp_group_,
    const phi::GPUContext* comm_ctx_,
    int32_t nnodes,
    int32_t full_m,
    int32_t n_dim,
    int32_t k_dim,
    bool transpose_weight = true,
    bool local_copy = false,
    AGRingMode ring_mode_ = AGRingMode::Auto)
    : dev_ctx(dev_ctx_),
      comm_ctx(comm_ctx_),
      tp_group(tp_group_),
      nnodes(nnodes),
      full_m(full_m),
      n_dim(n_dim),
      k_dim(k_dim),
      transpose_weight(transpose_weight),
      local_copy(local_copy),
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
    PADDLE_ENFORCE(
        rank >= 0 && rank < world_size,
        "invalid rank: %d and world_size: %d",
        rank,
        world_size);
    PADDLE_ENFORCE(
        world_size % nnodes == 0,
        "invalid nnodes: world_size[%d] % nnodes[%d] !=0",
        world_size,
        nnodes);
    PADDLE_ENFORCE(
        !(transpose_weight == true && is_fp8_gemm == true),
        "FP8 GEMM does not support transpose weight");
    this->ring_mode = get_ring_mode(ring_mode_);
    // input buffer
    static BuffersHolder<InT> input_buffers_holder{{full_m, k_dim}, dev_ctx, tp_group};
    // BuffersHolder<InT> input_buffers_holder{dev_ctx, tp_group};
    // BuffersHolder<InT> input_buffers_holder{{full_m, k_dim}, dev_ctx, tp_group};
    this->input_buffers = input_buffers_holder.get_buffers({full_m, k_dim});
    // this->input_buffers = cudaipc_create_tensor_list<InT>({full_m, k_dim});
    this->input_buffer = this->input_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      // this->input_buffer_ptrs[i] = this->input_buffers[i].data();
      if (i / this->local_world_size == rank / this->local_world_size) {
        // on the same node
        this->input_ptrs[i] = this->input_buffers[i].data();
      } else {
        this->input_ptrs[i] = nullptr;
      }
    }

    // this->output_buffer = phi::Empty<OutT>(dev_ctx, IntArray{full_m, n_dim});

    int num_signals = MAX_NUM_SIGNAL;
    static BuffersHolder<int32_t> barrier_buffers_holder{{num_signals}, dev_ctx, tp_group};
    this->barrier_buffers = barrier_buffers_holder.get_buffers({num_signals});
    barrier_buffers_holder.print_ptrs({"barrier_buffers"});
    // this->barrier_buffers = cudaipc_create_tensor_list<int32_t>({num_signals});
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      // this->barrier_buffer_ptrs[i] = this->barrier_buffers[i].data();
      if (i / this->local_world_size == rank / this->local_world_size) {
        // on the same node
        this->barrier_ptrs[i] = (FlagType *)this->barrier_buffers[i].data();
      } else {
        this->barrier_ptrs[i] = (FlagType *)nullptr;
      }
    }
#if 0
  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  distributed::BarrierOptions opts{};
  opts.device_id = device_id;
  this->tp_group->Barrier(opts)->Wait();
#endif

    // copy stream
    this->num_cp_streams = 1;
    for (int i = 0; i < this->num_cp_streams; ++i) {
      // umiswing: unfortunately, paddle has no such function
      // this->cp_streams.push_back(at::cuda::getStreamFromPool());
      cudaStream_t cp_stream = nullptr;
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&cp_stream));
      this->cp_streams.push_back(cp_stream);
    }
    // create events
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(&this->cp_event, cudaEventDisableTiming));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(&this->all_gather_event, cudaEventDisableTiming));
#if 0
    // reset stream
    for (int i = 0; i < 1; ++i) {
      cudaStream_t stream = nullptr;
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream));
      this->reset_streams.push_back(stream);
    }
#endif
#ifndef FLUX_SHM_USE_NVSHMEM
    static BuffersHolder<int32_t> sync_buffers_holder{{this->world_size}, dev_ctx, tp_group};
    this->sync_buffers = sync_buffers_holder.get_buffers({this->world_size});
    sync_buffers_holder.print_ptrs({"sync_buffers"});
    // this->sync_buffers =
    //     cudaipc_create_tensor_list<int32_t>({this->world_size});
    phi::funcs::SetConstant<GPUContext, int32_t> set_functor;
#if 0
    set_functor(this->dev_ctx, &this->sync_buffers[this->rank], 0);
#endif
    for(size_t i=0;i<this->sync_buffers.size();i++) {
      this->sync_buffer_ptrs[i] = static_cast<int32_t *>(this->sync_buffers[i].data());
    }
    print_address(this->sync_buffers, this->sync_buffer_ptrs, sync_buffers_holder, this->rank);
#endif
  }

  void
  lazy_init_gemm_buffer(int64_t buffer_size) {
    if (buffer_size <= 0)
      return;
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.initialized() || buffer_size > this->gemm_buffer.numel()) {
      this->gemm_buffer = phi::Empty<uint8_t>(dev_ctx,{buffer_size});
    }
  }

  ~AGGemmHelper() {
    cudaEventDestroy(cp_event);
    cudaEventDestroy(ready_event);
    cudaEventDestroy(all_gather_event);
    cudaStreamDestroy(this->cp_streams[0]);

#if 0
    for (int i = 0; i < world_size; ++i) {
      if(i != this->rank) {
        cudaIpcCloseMemHandle(this->input_buffers[i].data());
        cudaIpcCloseMemHandle(this->barrier_buffers[i].data());
        cudaIpcCloseMemHandle(this->sync_buffers[i].data());
      }
    }
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    distributed::BarrierOptions opts{};
    opts.device_id = device_id;
    this->tp_group->Barrier(opts)->Wait();
    cudaFree(this->input_buffers[this->rank].data());
    cudaFree(this->barrier_buffers[this->rank].data());
    cudaFree(this->sync_buffers[this->rank].data());
#endif
  }

#if 0
using Deleter = void (*)(phi::Allocation*);
using AllocationDeleter = void (*)(phi::Allocation*);
DenseTensor from_blob(void *data,
                      const std::vector<int64_t>& shape,
                      phi::DataType dtype,
                      phi::Place place,
                      const Deleter& deleter,
                      phi::DataLayout layout = phi::DataLayout::NCHW ) {
  PADDLE_ENFORCE_NOT_NULL(
      data, common::errors::InvalidArgument("data can not be nullptr."));

  // TODO(umiswing): this check looks nice
  // auto data_place = GetPlaceFromPtr(data);
  // phi::is_gpu_place(place);
#if 0
  PADDLE_ENFORCE(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU,
      "gemm_rs not on GPU");
#endif

  auto meta =
      phi::DenseTensorMeta(dtype, common::make_ddim(shape), layout);

  size_t size = SizeOf(dtype) * (meta.is_scalar ? 1 : product(meta.dims));

  auto alloc =
      // std::make_shared<phi::Allocation>(data, size, alloc_deleter, place/*data_place*/);
      std::make_shared<phi::Allocation>(data, size, deleter, place/*data_place*/);

  return DenseTensor(alloc, meta);
}

template<typename BufferT>
std::vector<DenseTensor>
cudaipc_create_tensor_list(
    const std::vector<int64_t> &shape) {

  PADDLE_ENFORCE_GE(
      phi::backends::gpu::GetGPUDeviceCount(),
      tp_group->GetSize(),
      common::errors::InvalidArgument("create_ipc_tensors should only be used intra node"));

  size_t size = sizeof(BufferT) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
#if 0
  size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
#endif
  PADDLE_ENFORCE_NE(size, 0);

  void *ptr = nullptr;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(ptr, 0, size)); // memset the allocated buffer

#if 0
  DenseTensor local_tensor = phi::Empty<BufferT>(dev_ctx, IntArray{shape});
  phi::funcs::SetConstant<GPUContext, BufferT> set_functor;
  set_functor(dev_ctx, &local_tensor, BufferT{0});
  void *ptr = local_tensor.data();
#endif

  cudaIpcMemHandle_t handle;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcGetMemHandle(&handle, ptr));

  DenseTensor handle_d = phi::Empty<uint8_t>(dev_ctx, {sizeof(cudaIpcMemHandle_t)});
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
      handle_d.data(), &handle, sizeof(cudaIpcMemHandle_t), cudaMemcpyHostToDevice));
  long int handles_shape = sizeof(cudaIpcMemHandle_t) * tp_group->GetSize();
  DenseTensor handles_d = phi::Empty<uint8_t>(dev_ctx, {handles_shape});
  // TODO(umiswing): find a better way to wrap func params
  tp_group->AllGather(&handles_d, handle_d, 0, -1, true, true)->Wait();

  std::vector<cudaIpcMemHandle_t> handles_h(tp_group->GetSize());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
      handles_h.data(),
      handles_d.data(),
      sizeof(cudaIpcMemHandle_t) * tp_group->GetSize(),
      cudaMemcpyDeviceToHost));

  std::vector<void *> ptrs(tp_group->GetSize());
  for (int i = 0; i < tp_group->GetSize(); ++i) {
    if (i != tp_group->GetRank()) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(&ptrs[i], handles_h[i], cudaIpcMemLazyEnablePeerAccess));
    } else {
      ptrs[i] = ptr;
    }
  }

 phi::DataType dtype;
 if (std::is_same<BufferT, phi::dtype::float16>::value) dtype = phi::DataType::FLOAT16;
 else if(std::is_same<BufferT, phi::dtype::bfloat16>::value) dtype = phi::DataType::BFLOAT16;
 else if(std::is_same<BufferT, uint8_t>::value) dtype = phi::DataType::UINT8;
 else if(std::is_same<BufferT, int32_t>::value) dtype = phi::DataType::INT32;
 else throw std::runtime_error("cudaipc_create_tensor_list unexpected BufferT");

  std::vector<DenseTensor> tensors;
  for (int i = 0; i < tp_group->GetSize(); ++i) {
    if (i == tp_group->GetRank()) {
      DenseTensor local_tensor;
      local_tensor =
          // from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { cudaFree(allocation->ptr()); });
          from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { });
      tensors.emplace_back(local_tensor);
    } else {
      DenseTensor tensor;
      tensor =
          // from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { cudaIpcCloseMemHandle(allocation->ptr()); });
          from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { });
      tensors.emplace_back(tensor);
    }
  }

  return tensors;
}
#endif
};

template<typename T, typename Context>
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
                         const bool transpose_weight,
                         const bool local_copy,
                         DenseTensor* output,
                         DenseTensor* input_parallel) {
  constexpr int SPLIT = 1;

  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(pg,
                    nullptr,
                    common::errors::Unavailable(
                        "ProcessGroup is nullptr."));

  const phi::GPUContext* comm_ctx = static_cast<phi::GPUContext*>(pg->GetGemmRSContext(input.place()));

  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "comm_ctx is nullptr."));

  const int32_t n = transpose_weight ? weight.dims()[1] : weight.dims()[0];
  const int32_t k = transpose_weight ? weight.dims()[0] : weight.dims()[1];
  AGGemmHelper<T,T> helper{dev_ctx,
                             pg,
                             comm_ctx,
                             nnodes,
                             full_m,
                             n_dim,
                             // 2304,
                             k_dim,
                             transpose_weight,
                             local_copy,
                             AGRingMode::Auto};

  helper.n_dim = n_dim;

  if(local_copy) {
    PADDLE_ENFORCE_EQ(
        input.numel() * SizeOf(input.dtype()), // or tensor.capacity()?
        helper.chunk_size,
        common::errors::InvalidArgument(
            "helper.chunk_size should be equal to input.numel() * SizeOf(input.dtype())"));
  }
  helper.chunk_size = input.numel() * SizeOf(input.dtype());
  helper.split_chunk_size = helper.chunk_size / SPLIT;

#if defined(FLUX_DEBUG)
  static bool kDebugRunGemm = get_bool_from_env("FLUX_AG_RUN_GEMM", true);
  static bool kPushMode = get_bool_from_env("FLUX_AG_CPY_PUSH", true);
#else
  static bool kDebugRunGemm = true;
  static bool kPushMode = true;
#endif
  DenseTensor output_buffer = phi::Empty<T>(dev_ctx, IntArray{full_m, n_dim});
  // *output = phi::Empty<T>(dev_ctx, IntArray{full_m, n_dim});

  // static int32_t first_flag = 0;
  auto launcher = [&](const bool return_workspace_size) -> size_t {
    return phi::dynload::ag_gemm(
        const_cast<void*>(input.data()),
        helper.input_buffer.data(),
        const_cast<void*>(weight.data()),
        bias.is_initialized() ? const_cast<void*>(bias->data()) : nullptr,
        // helper.output_buffer.data(),
        output_buffer.data(),
        // output->data(),
        helper.barrier_buffer.data(),
        helper.gemm_buffer.initialized() ? helper.gemm_buffer.data() : nullptr,
        dev_ctx.stream(),
        helper.ready_event,
        // first_flag == 0 ? 2304 : n, k, first_flag == 0 ? 2304 : helper.n_dim, helper.k_dim,
        n, k, helper.n_dim, helper.k_dim,
        input.dims()[0],
        helper.rank,
        helper.world_size,
        helper.nnodes,
        static_cast<int>(helper.ring_mode),
        std::is_same<T, phi::dtype::bfloat16>::value,
        kDebugRunGemm,
        helper.transpose_weight,
        fast_accum,
        return_workspace_size);
  };

  // if (first_flag == 0) {
    auto get_workspace_size = [&]() -> size_t {
      return launcher(true);
    };
    int64_t workspace_size = get_workspace_size();
    helper.lazy_init_gemm_buffer(workspace_size);
  // }
  // first_flag = 1;

  auto ag_gemm = [&]() {
    launcher(false);
  };

  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  distributed::BarrierOptions opts{};
  opts.device_id = device_id;
  pg->Barrier(opts)->Wait();

#if 0
  cudaEvent_t all_to_all_event;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(&all_to_all_event, cudaEventDisableTiming));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(all_to_all_event, comm_ctx->stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(dev_ctx.stream(), all_to_all_event));
  cudaEventDestroy(all_to_all_event);
#endif

  /// AG GEMM
  cudaStream_t current_stream = dev_ctx.stream();

  if (!local_copy) {
    print_ptrs<int32_t>({"sync_buffer_ptrs.data()"}, helper.sync_buffer_ptrs, helper.world_size);
    // copy_local
    helper.chunk_size = input.numel() * SizeOf(input.dtype());
    helper.split_chunk_size = helper.chunk_size / SPLIT;
    const void *input_ptr = input.data();
    void *input_buffer_ptr = helper.input_buffer.data();

    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
        ptr_offset(input_buffer_ptr, helper.rank * helper.chunk_size),
        input_ptr,
        helper.chunk_size,
        cudaMemcpyDefault,
        current_stream));

  if(helper.rank == 1) {
    write_ptrs<int32_t>(helper.sync_buffers, dev_ctx);
  }

  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  distributed::BarrierOptions opts{};
  opts.device_id = device_id;
  pg->Barrier(opts)->Wait();

  cudaDeviceSynchronize();

  print_ptrs<int32_t>({"sync_buffer_ptrs.data()"}, helper.sync_buffer_ptrs, helper.world_size);
    for (int j = 0; j < SPLIT; ++j) {
      phi::dynload::set_ready(helper.barrier_ptrs[helper.rank], helper.rank, j, dev_ctx.stream());
    }
  print_ptrs<int32_t>({"sync_buffer_ptrs.data()"}, helper.sync_buffer_ptrs, helper.world_size);
    phi::dynload::cudaipc_barrier_all_on_stream_impl_capi(
        current_stream,
        helper.sync_buffer_ptrs.data(),
        helper.rank,
        helper.world_size);
  }

  print_ptrs<int32_t>({"sync_buffer_ptrs.data()"}, helper.sync_buffer_ptrs, helper.world_size);


  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(helper.ready_event, current_stream));

  if (helper.ring_mode == AGRingMode::All2All) {
    /// All2All algorithm
    // copy_all_to_all(input, helper.cp_streams[0]);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(helper.cp_streams[0], helper.ready_event));

    for (int i = helper.rank + 1; i < (helper.world_size + helper.rank); ++i) {
      auto id = i % helper.world_size;
      for (int j = 0; j < SPLIT; ++j) {
        auto split_offset = j * helper.split_chunk_size;
        PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
            ptr_offset(helper.input_ptrs[helper.rank], id * helper.chunk_size + split_offset),
            ptr_offset(helper.input_ptrs[id], id * helper.chunk_size + split_offset),
            helper.split_chunk_size,
            cudaMemcpyDefault,
            helper.cp_streams[0]));
        phi::dynload::set_ready(helper.barrier_ptrs[helper.rank], id, j, helper.cp_streams[0]);
      }
    }
  }

  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(helper.cp_event, helper.cp_streams[0]));

  ag_gemm();

#if 0
  *output = phi::funcs::Slice<T>(dev_ctx,
                                 // helper.output_buffer,
                                 output_buffer,
                                 {0},
                                 {0},
                                 {static_cast<int32_t>(input.dims()[0] * helper.world_size)});
#endif

  // *output = phi::Slice<T>(dev_ctx, output_buffer, {0}, {0}, {static_cast<int32_t>(input.dims()[0] * helper.world_size)});
  // phi::SliceKernel<T>(dev_ctx, output, {0}, {0}, {static_cast<int32_t>(input.dims()[0] * helper.world_size)});
#if 0
  std::vector<int64_t> infer_flags = {1};
  std::vector<int64_t> decrease_axis = {};
  phi::SliceKernel<T, Context>(
    dev_ctx, *output, {0}, {0}, {static_cast<int32_t>(input.dims()[0] * helper.world_size)}, infer_flags, decrease_axis, output);
#endif
  PADDLE_ENFORCE(
    output_buffer.meta().is_contiguous(),
    "output_buffer is not contiguous");
  MetaTensor meta_output(output);
  std::vector<int64_t> infer_flags = {1};
  std::vector<int64_t> decrease_axis = {};
  SliceRawInferMeta(
    output_buffer, {0}, {0}, {static_cast<int32_t>(input.dims()[0] * helper.world_size)}, infer_flags, decrease_axis, &meta_output);
  phi::SliceStridedKernel<Context>(dev_ctx, output_buffer, {0}, {0}, {static_cast<int32_t>(input.dims()[0] * helper.world_size)}, infer_flags, decrease_axis, output);
  if(!output->meta().is_contiguous()) {
    phi::SliceKernel<T, Context>(
      dev_ctx, output_buffer, {0}, {0}, {static_cast<int32_t>(input.dims()[0] * helper.world_size)}, infer_flags, decrease_axis, output);
  }

  if (fast_accum) {
    *input_parallel = helper.input_buffer;
  } else {
    *input_parallel = phi::Empty<T>(dev_ctx, IntArray{full_m, k_dim});
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      input_parallel->data(),
      helper.input_buffer.data(),
      sizeof(T) * full_m * k_dim,
      cudaMemcpyDefault,
      helper.cp_streams[0]));
  }
#if 0
  phi::Copy(dev_ctx, helper.input_buffer, dev_ctx.GetPlace(), false, input_parallel);
#endif

  /// reset signals
  cudaStreamWaitEvent(current_stream, helper.cp_event);
  phi::dynload::cudaipc_barrier_all_on_stream_impl_capi(
    current_stream,
    helper.sync_buffer_ptrs.data(),
    helper.rank,
    helper.world_size); 

  phi::funcs::SetConstant<GPUContext, int32_t> set_functor;
  set_functor(dev_ctx, &(helper.barrier_buffer), int32_t{0});
  // gather();
}

} // namespace phi

PD_REGISTER_KERNEL(all_gather_gemm,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllGatherGemmKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
}
