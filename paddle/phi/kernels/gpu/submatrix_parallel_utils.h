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

namespace smp {

template <typename T>
void get_submatrix(const GPUContext& dev_ctx,
                   const DenseTensor& x,
                   int32_t world_size,
                   int32_t id,
                   DenseTensor* sub_x) {
  int32_t x_rank = x.dims().size();
  int32_t micro_hidden_size = x.dims()[x_rank - 2] / world_size;
  int32_t start = id * micro_hidden_size;
  int32_t end = start + micro_hidden_size;
  MetaTensor meta_sub_x(sub_x);
  std::vector<int64_t> infer_flags = {1};
  std::vector<int64_t> decrease_axis = {};
  SliceRawInferMeta(
      x, {x_rank - 2}, {start}, {end}, infer_flags, decrease_axis, &meta_sub_x);

  phi::SliceStridedKernel<GPUContext>(dev_ctx,
                                      x,
                                      {x_rank - 2},
                                      {start},
                                      {end},
                                      infer_flags,
                                      decrease_axis,
                                      sub_x);

  if (!sub_x->meta().is_contiguous()) {
    // raise error?
    phi::SliceKernel<T, GPUContext>(dev_ctx,
                                    x,
                                    {x_rank - 2},
                                    {start},
                                    {end},
                                    infer_flags,
                                    decrease_axis,
                                    sub_x);
  }
}

/// Load flag, as a strong acquire operation (int specialization)
__forceinline__ __device__ static int32_t ld_acquire(int32_t* ptr) {
  int32_t state = 0;

#if (__CUDA_ARCH__ >= 700)
  /// SM70 and newer use memory consistency qualifiers

  // Acquire pattern using acquire modifier
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(state)
               : "l"(ptr));

#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif  // (__CUDA_ARCH__ >= 700)

  return state;
}

__forceinline__ __global__ static void wait_eq(void* barrier_ptr,
                                               int32_t barrier_idx,
                                               int32_t val) {
  int32_t* flag_ptr = static_cast<int32_t*>(barrier_ptr) + barrier_idx;
  // clang-format off
    if (threadIdx.x == 0) {
      #pragma unroll 1
      while (ld_acquire(flag_ptr) != val) {}
    }
    __syncthreads();
    // // clang-format on
  }

void wait_full(void* barrier_ptr, int32_t barrier_idx, cudaStream_t stream) {
  wait_eq<<<1, 32, 0, stream>>>(barrier_ptr, barrier_idx, 1);
}

void wait_empty(void* barrier_ptr, int32_t barrier_idx, cudaStream_t stream) {
  wait_eq<<<1, 32, 0, stream>>>(barrier_ptr, barrier_idx, 0);
}

void set_val(void* barrier_ptr,
             int32_t barrier_idx,
             cuuint32_t val,
             cudaStream_t stream) {
  int32_t* flag_ptr = static_cast<int32_t *>(barrier_ptr) + barrier_idx;
  phi::dynload::cuStreamWriteValue32_v2(
      stream,
      (CUdeviceptr)(flag_ptr),
      val,
      CU_STREAM_WRITE_VALUE_DEFAULT);
}

void set_full(void* barrier_ptr, int32_t barrier_idx, cudaStream_t stream) {
  set_val(barrier_ptr, barrier_idx, 1, stream);
}

void set_empty(void* barrier_ptr, int32_t barrier_idx, cudaStream_t stream) {
  set_val(barrier_ptr, barrier_idx, 0, stream);
}

__forceinline__ __global__ static void wait_eq_reset(void* barrier_ptr,
                                                     int32_t barrier_idx,
                                                     int32_t val,
                                                     int reset_val) {
  int *flag_ptr = static_cast<int32_t *>(barrier_ptr) + barrier_idx;
  // clang-format off
  if (threadIdx.x == 0) {
    #pragma unroll 1
    while(atomicCAS(flag_ptr, val, reset_val) != val) {}
  }
  // clang-format on
  __syncthreads();
}

void wait_full_reset(void* barrier_ptr,
                     int32_t barrier_idx,
                     cudaStream_t stream) {
  wait_eq_reset<<<1, 32, 0, stream>>>(barrier_ptr, barrier_idx, 1, 0);
}

void wait_empty_reset(void* barrier_ptr,
                      int32_t barrier_idx,
                      cudaStream_t stream) {
  wait_eq_reset<<<1, 32, 0, stream>>>(barrier_ptr, barrier_idx, 0, 1);
}

constexpr int32_t kMaxWorldSize = 32;

struct CudaIpcBarrierAllArgs {
  int32_t* sync_buffers[kMaxWorldSize];
  int32_t rank;
  int32_t world_size;
};

__global__ void CudaIpcBarrierAllKernel(CudaIpcBarrierAllArgs args) {
  int32_t** sync_buffers = args.sync_buffers;
  int32_t world_size = args.world_size;
  int32_t cur_rank = args.rank;
  if (threadIdx.x < world_size) {
    // set achieved flag for others
    int32_t* sync_buffer_dst = sync_buffers[threadIdx.x] + cur_rank;
    uint32_t const data = 1;
#pragma unroll 1
    while (atomicCAS_system(sync_buffer_dst, 0, 1) != 0) {
    }
    int32_t* wait_ptr = sync_buffers[cur_rank] + threadIdx.x;
#pragma unroll 1
    while (atomicCAS_system(wait_ptr, 1, 0) != 1) {
    }
  }
}

void cudaipc_barrier_all_on_stream_impl(cudaStream_t stream,
                                        int32_t** sync_buffer_ptr,
                                        int32_t rank,
                                        int32_t world_size) {
  dim3 grid_dim(1);
  dim3 block_dim(kMaxWorldSize);
  CudaIpcBarrierAllArgs args;
  args.world_size = world_size;
  args.rank = rank;
  assert(world_size < kMaxWorldSize);
  for (int32_t i = 0; i < world_size; i++) {
    args.sync_buffers[i] = sync_buffer_ptr[i];
  }
  CudaIpcBarrierAllKernel<<<grid_dim, block_dim, 0, stream>>>(args);
}

}  // namespace smp
}  // namespace phi
