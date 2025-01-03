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

#include "paddle/phi/backends/dynload/cuda_driver.h"

#pragma once

namespace phi {

inline void* ptr_offset(void* ptr, ptrdiff_t offset) {
  return static_cast<char*>(ptr) + offset;
}

// Integration of Flux, a gemm-comm-overlap library as described in the paper
// https://arxiv.org/pdf/2406.06858

#ifdef PADDLE_WITH_FLUX

// All2All for nvlink mode. for NVLINK machine, default is 0
// Ring1D for 1d-ring. for PCI-e machine without GPUs cross NUMA nodes use ring
// 1d Ring2D for 2d-ring. for PCI-e machine with GPUs cross NUMA nodes defaults
// to ring_2d RingCustom for custom ring. for defining arbitrary ring at compile
// time
enum class AGRingMode {
  All2All = 0,
  Ring1D = 1,
  Ring2D = 2,
  RingCustom = 3,
  Auto = -1,
};

static AGRingMode get_ring_mode(AGRingMode ring_mode) {
  return AGRingMode::All2All;
}

class CUDAEventHolder {
 public:
  explicit CUDAEventHolder(const bool disable_timing = false) {
    if (disable_timing) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventCreateWithFlags(&this->event, cudaEventDisableTiming));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&event));
    }
  }

  ~CUDAEventHolder() { PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event)); }
  cudaEvent_t event;
};

template <typename BufferT>
class BuffersHolder {
 private:
  const GPUContext& dev_ctx;
  paddle::distributed::ProcessGroup* tp_group;
  size_t world_size;
  std::vector<void*> ptrs;
  size_t size_in_bytes;
  void* ptr;
  phi::DataType dtype;

 public:
  BuffersHolder(const std::vector<int64_t>& shape,
                const GPUContext& dev_ctx_,
                paddle::distributed::ProcessGroup* tp_group_)
      : dev_ctx(dev_ctx_),
        tp_group(tp_group_),
        world_size(tp_group->GetSize()),
        ptrs(world_size, nullptr) {
    if (std::is_same<BufferT, phi::dtype::float16>::value) {
      dtype = phi::DataType::FLOAT16;
    } else if (std::is_same<BufferT, phi::dtype::bfloat16>::value) {
      dtype = phi::DataType::BFLOAT16;
    } else if (std::is_same<BufferT, uint8_t>::value) {
      dtype = phi::DataType::UINT8;
    } else if (std::is_same<BufferT, int32_t>::value) {
      dtype = phi::DataType::INT32;
    } else {
      PADDLE_THROW(
          common::errors::Unimplemented("BuffersHolder unexpected BufferT"));
    }

    this->size_in_bytes = calc_size(shape);
    alloc();
  }

  // TODO(umiswing): although BuffersHolder object is static, it's better to
  // find a way to destruct it.

  std::vector<DenseTensor> get_buffers(const std::vector<int64_t>& shape) {
    reserve(shape);
    std::vector<DenseTensor> tensors;
    for (int i = 0; i < tp_group->GetSize(); ++i) {
      if (i == tp_group->GetRank()) {
        DenseTensor local_tensor;
        local_tensor = from_blob(ptrs[i],
                                 shape,
                                 dtype,
                                 dev_ctx.GetPlace(),
                                 [](phi::Allocation* allocation) {});
        tensors.emplace_back(local_tensor);
      } else {
        DenseTensor tensor;
        tensor = from_blob(ptrs[i],
                           shape,
                           dtype,
                           dev_ctx.GetPlace(),
                           [](phi::Allocation* allocation) {});
        tensors.emplace_back(tensor);
      }
    }

    return tensors;
  }

 private:
  void alloc() {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size_in_bytes));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(ptr, 0, size_in_bytes));

    cudaIpcMemHandle_t handle;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcGetMemHandle(&handle, ptr));

    DenseTensor handle_d =
        phi::Empty<uint8_t>(dev_ctx, {sizeof(cudaIpcMemHandle_t)});
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(handle_d.data(),
                                          &handle,
                                          sizeof(cudaIpcMemHandle_t),
                                          cudaMemcpyHostToDevice));
    int64_t handles_shape = sizeof(cudaIpcMemHandle_t) * tp_group->GetSize();
    DenseTensor handles_d = phi::Empty<uint8_t>(dev_ctx, {handles_shape});
    tp_group->AllGather(&handles_d, handle_d, 0, -1, true, true)->Wait();

    std::vector<cudaIpcMemHandle_t> handles_h(tp_group->GetSize());
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpy(handles_h.data(),
                   handles_d.data(),
                   sizeof(cudaIpcMemHandle_t) * tp_group->GetSize(),
                   cudaMemcpyDeviceToHost));

    for (int i = 0; i < tp_group->GetSize(); ++i) {
      if (i != tp_group->GetRank()) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(
            &ptrs[i], handles_h[i], cudaIpcMemLazyEnablePeerAccess));
      } else {
        ptrs[i] = ptr;
      }
    }
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    distributed::BarrierOptions opts{};
    opts.device_id = device_id;
    this->tp_group->Barrier(opts)->Wait();
  }

  size_t calc_size(const std::vector<int64_t>& shape) {
    return sizeof(BufferT) *
           std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  }

  void release() {
    for (int i = 0; i < world_size; ++i) {
      if (i != this->tp_group->GetRank()) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcCloseMemHandle(this->ptrs[i]));
      }
    }
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    distributed::BarrierOptions opts{};
    opts.device_id = device_id;
    this->tp_group->Barrier(opts)->Wait();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(this->ptr));

    for (int i = 0; i < world_size; ++i) {
      this->ptrs[i] = nullptr;
    }
  }

  void reserve(const std::vector<int64_t>& shape) {
    size_t require_size = calc_size(shape);
    if (require_size > this->size_in_bytes) {
      this->size_in_bytes = require_size;
      release();
      alloc();
    }
  }

  using Deleter = void (*)(phi::Allocation*);
  DenseTensor from_blob(void* data,
                        const std::vector<int64_t>& shape,
                        phi::DataType dtype,
                        phi::Place place,
                        const Deleter& deleter,
                        phi::DataLayout layout = phi::DataLayout::NCHW) {
    PADDLE_ENFORCE_NOT_NULL(
        data, common::errors::InvalidArgument("data can not be nullptr."));

    auto meta = phi::DenseTensorMeta(dtype, common::make_ddim(shape), layout);

    size_t size = SizeOf(dtype) * (meta.is_scalar ? 1 : product(meta.dims));

    auto alloc = std::make_shared<phi::Allocation>(
        data, size, deleter, place /*data_place*/);

    return DenseTensor(alloc, meta);
  }
};
#endif

namespace flux {
static void RaiseNotSupportedError() {
  PADDLE_THROW(
      common::errors::Unimplemented("Flux is unsupported, please check "
                                    "the GPU compability and CUDA Version."));
}
}  // namespace flux
}  // namespace phi
