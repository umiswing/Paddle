namespace phi {
inline void *
ptr_offset(void *ptr, ptrdiff_t offset) {
  return static_cast<char *>(ptr) + offset;
}

// All2All for nvlink mode. for NVLINK machine, default is 0
// Ring1D for 1d-ring. for PCI-e machine without GPUs cross NUMA nodes use ring 1d
// Ring2D for 2d-ring. for PCI-e machine with GPUs cross NUMA nodes defaults to ring_2d
// RingCustom for custom ring. for defining arbitrary ring at compile time
enum class AGRingMode {
  All2All = 0,
  Ring1D = 1,
  Ring2D = 2,
  RingCustom = 3,
  Auto = -1,
};

static AGRingMode
get_ring_mode(AGRingMode ring_mode) {
      return AGRingMode::All2All;
}

template<typename BufferT>
class BuffersHolder {
public:
  const GPUContext& dev_ctx;
  paddle::distributed::ProcessGroup* tp_group;
  size_t world_size;
  std::vector<void*> ptrs;
  size_t size_in_bytes;
  void * ptr;
  phi::DataType dtype;
  DenseTensor local_buffer;
  int64_t numel;
public:

#if 0
  BuffersHolder(
                const GPUContext& dev_ctx_,
                paddle::distributed::ProcessGroup* tp_group_) :
    dev_ctx(dev_ctx_),
    tp_group(tp_group_),
    world_size(tp_group->GetSize()),
    ptrs(world_size, nullptr) {}
#endif

  BuffersHolder(const std::vector<int64_t>&shape,
                const GPUContext& dev_ctx_,
                paddle::distributed::ProcessGroup* tp_group_) :
    dev_ctx(dev_ctx_),
    tp_group(tp_group_),
    world_size(tp_group->GetSize()),
    ptrs(world_size, nullptr) {

    if (std::is_same<BufferT, phi::dtype::float16>::value) dtype = phi::DataType::FLOAT16;
    else if(std::is_same<BufferT, phi::dtype::bfloat16>::value) dtype = phi::DataType::BFLOAT16;
    else if(std::is_same<BufferT, uint8_t>::value) dtype = phi::DataType::UINT8;
    else if(std::is_same<BufferT, int32_t>::value) dtype = phi::DataType::INT32;
    else throw std::runtime_error("cudaipc_create_tensor_list unexpected BufferT");

    this->size_in_bytes = calc_size(shape);
    this->numel = calc_numel(shape);
    alloc();
  }

  // TODO(umiswing): need to find a way to destruct BuffersHolder

  std::vector<DenseTensor> get_buffers(const std::vector<int64_t>& shape) {
    reserve(shape);
    std::vector<DenseTensor> tensors;
    for (int i = 0; i < tp_group->GetSize(); ++i) {
      if (false && i == tp_group->GetRank()) {
        tensors.emplace_back(this->local_buffer);
      } else {
        DenseTensor tensor;
        tensor =
            // from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { cudaIpcCloseMemHandle(allocation->ptr()); });
            from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) {});
        tensors.emplace_back(tensor);
      }
    }

    return tensors;

  }

  void print_ptrs(const std::string& s) {
    BufferT x;

    std::cout << "\nprint_ptrs: " << s << "\n" << std::endl;

    bool all_zero = true;

    for(auto& d_ptr : ptrs) {
      for(int i=0;i<this->numel;i++) {
        cudaMemcpy(&x,static_cast<BufferT*>(d_ptr)+i,sizeof(BufferT),cudaMemcpyDeviceToHost);
        if(x!=static_cast<BufferT>(0)) {
          all_zero = false;
          std::cout << " ," << x;
        }
      }
    }
    if(all_zero) {
      std::cout << "all zero!";
    }
    std::cout << std::endl;

  }

private:
  void alloc() {
#if 0
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size_in_bytes));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(ptr, 0, size_in_bytes));
#endif
    this->local_buffer = phi::Empty<BufferT>(dev_ctx, {this->numel});
    phi::funcs::SetConstant<GPUContext, BufferT> set_zero;
    set_zero(this->dev_ctx, &(this->local_buffer), static_cast<BufferT>(0));
    ptr = this->local_buffer.data();
#if 0
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(ptr, 0, this->local_buffer.numel()));
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

    for (int i = 0; i < tp_group->GetSize(); ++i) {
      if (i != tp_group->GetRank()) {
          PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(&ptrs[i], handles_h[i], cudaIpcMemLazyEnablePeerAccess));
      } else {
        ptrs[i] = ptr;
      }
    }
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    distributed::BarrierOptions opts{};
    opts.device_id = device_id;
    this->tp_group->Barrier(opts)->Wait();
  }

  int64_t calc_numel(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  }

  size_t calc_size(const std::vector<int64_t>& shape) {
    return sizeof(BufferT) * this->calc_numel(shape);
  }

  void release() {
    for(int i=0; i<world_size; ++i) {
      if(i != this->tp_group->GetRank()) {
        cudaIpcCloseMemHandle(this->ptrs[i]);
      }
    }
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    distributed::BarrierOptions opts{};
    opts.device_id = device_id;
    this->tp_group->Barrier(opts)->Wait();
    // cudaFree(this->ptr);
  }

  void reserve(const std::vector<int64_t>& shape) {
    size_t require_size = calc_size(shape);
    if(require_size > this->size_in_bytes) {
      this->size_in_bytes = require_size;
      this->numel = this->calc_numel(shape);
      release();
      alloc();
    }
  }

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

    auto meta = this->local_buffer.meta();
  
    size_t size = SizeOf(dtype) * (meta.is_scalar ? 1 : product(meta.dims));
  
    auto alloc =
        // std::make_shared<phi::Allocation>(data, size, alloc_deleter, place/*data_place*/);
        std::make_shared<phi::Allocation>(data, size, deleter, place/*data_place*/);
  
    return DenseTensor(alloc, meta);
  }
};
} // namespace phi
