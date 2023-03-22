// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/backends/dynload/rocsparse.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename IntT>
rocsparse_indextype GetGpuIndexType() {
  if (std::is_same<IntT, int32_t>::value) {
    return rocsparse_indextype_i32;
  } else if (std::is_same<IntT, int64_t>::value) {
    return rocsparse_indextype_i64;
  }
}

template <typename T>
rocsparse_datatype GetGpuDataType() {
  if (std::is_same<T, float>::value) {
    return rocsparse_datatype_f32_r;
  } else if (std::is_same<T, double>::value) {
    return rocsparse_datatype_f64_r;
  }
}

inline rocsparse_operation GetTransposeOperation(const bool trans) {
  if (trans) {
    return rocsparse_operation_transpose;
  } else {
    return rocsparse_operation_none;
  }
}

inline rocsparse_spmm_alg GetSpMMAlgorithm(const SparseCooTensor& x) {
  return rocsparse_spmm_alg_default;
}

/************* SPARSE MATRIX DESCRIPTOR (COO/CSR) ************/
template <typename T, typename IntT>
inline void CreateCooDescriptor(const phi::SparseCooTensor& x,
                                const phi::GPUContext& dev_ctx,
                                rocsparse_spmat_descr* descriptor) {
  std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
  auto x_ndims = xdim_vec.size();
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      phi::errors::InvalidArgument("the dim size of SparseCooTensor must be "
                                   "greater than or eaqual to 2."));

  int64_t M = xdim_vec[x_ndims - 2];
  int64_t N = xdim_vec[x_ndims - 1];
  int batch_size = 1;
  for (int i = 0; i < x_ndims - 2; i++) {
    batch_size *= xdim_vec[i];
  }
  int64_t nnz = x.nnz();

  const IntT* indices_data = x.non_zero_indices().data<IntT>();
  const T* values_data = x.non_zero_elements().data<T>();
  auto rows_data = indices_data + (x_ndims - 2) * nnz;
  auto cols_data = indices_data + (x_ndims - 1) * nnz;

  int64_t batch_nnz = nnz / batch_size;
  rocsparse_indextype itype = GetGpuIndexType<IntT>();
  rocsparse_indextype jtype = GetGpuIndexType<IntT>();
  rocsparse_datatype ttype = GetGpuDataType<T>();
  dev_ctx.CusparseCall([&](rocsparse_handle handle) {
    phi::dynload::rocsparse_create_coo_descr(descriptor,
                                             M,
                                             N,
                                             batch_nnz,
                                             const_cast<IntT*>(rows_data),
                                             const_cast<IntT*>(cols_data),
                                             const_cast<T*>(values_data),
                                             itype,
                                             jtype,
                                             rocsparse_index_base_zero,
                                             ttype);
  });

  if (batch_size > 1) {
#if CUDA_VERSION >= 11080
#if 0
    dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCooSetStridedBatch(
          *descriptor, batch_size, batch_nnz);
    });
#endif
#else
    PADDLE_THROW(phi::errors::Unimplemented(
        "Batch Sparse matmul use 'cusparseCooSetStridedBatch', which is "
        "supported from CUDA 11.8"));
#endif
  }
}

template <typename T>
class RocSparseSpMatDescriptor {
 public:
  explicit RocSparseSpMatDescriptor(const phi::SparseCooTensor& x,
                                    const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    PD_VISIT_BASE_INTEGRAL_TYPES(
        x.non_zero_indices().dtype(), "Coo RocSparseSpMatDescriptor", ([&] {
          CreateCooDescriptor<T, data_t>(x, dev_ctx_, &descriptor_);
        }));
    VLOG(6) << "Create coo rocsparse_spmat_descr " << &descriptor_;
  }

  ~RocSparseSpMatDescriptor() {
    dev_ctx_.CusparseCall([&](rocsparse_handle handle) {
      phi::dynload::rocsparse_destroy_spmat_descr(descriptor_);
    });
    VLOG(6) << "Destroy roscparse_spmat_descr " << &descriptor_;
  }

  const rocsparse_spmat_descr& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  rocsparse_spmat_descr descriptor_;
};

/************* DENSE MATRIX DESCRIPTOR ************/
template <typename T>
class RocSparseDnMatDescriptor {
 public:
  explicit RocSparseDnMatDescriptor(const phi::DenseTensor& x,
                                    const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();
    PADDLE_ENFORCE_GE(
        x_ndims,
        2,
        phi::errors::InvalidArgument("the dim size of SparseCsrTensor must be "
                                     "greater than or eaqual to 2."));

    int64_t M = xdim_vec[x_ndims - 2];
    int64_t N = xdim_vec[x_ndims - 1];
    int batch_size = 1;
    for (int i = 0; i < x_ndims - 2; i++) {
      batch_size *= xdim_vec[i];
    }
    int64_t ld = N;
    rocsparse_datatype ttype = GetGpuDataType<T>();
    dev_ctx.CusparseCall([&](rocsparse_handle handle) {
      phi::dynload::rocsparse_create_dnmat_descr(
          descriptor_, M, N, ld, x.data(), ttype, rocsparse_order_row);
    });
  }

  ~RocSparseDnMatDescriptor() {
    dev_ctx_.CusparseCall([&](rocsparse_handle handle) {
      phi::dynload::rocsparse_destroy_dnmat_descr(descriptor_);
    });
    VLOG(6) << "Destroy rocsparse_dnmat_descr " << &descriptor_;
  }

  const rocsparse_dnmat_descr& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  rocsparse_dnmat_descr descriptor_;
};

/************* SPARSE*DENSE->DENSE MATMUL ************/
template <>
template <typename T, typename TensorType>
void SparseBlas<phi::GPUContext>::SPMM(bool transa,
                                       bool transb,
                                       T alpha,
                                       const TensorType& mat_a,
                                       const phi::DenseTensor& mat_b,
                                       T beta,
                                       phi::DenseTensor* mat_out) const {
  auto a_descriptor = RocSparseSpMatDescriptor<T>(mat_a, dev_ctx_);
  auto b_descriptor = RocSparseDnMatDescriptor<T>(mat_b, dev_ctx_);
  auto out_descriptor = RocSparseDnMatDescriptor<T>(*mat_out, dev_ctx_);

  rocsparse_datatype ttype = GetGpuDataType<T>();
  size_t buffer_size = 0;
  // Query SpMM buffer
  dev_ctx_.CusparseCall([&](rocsparse_handle handle) {
    phi::dynload::rocsparse_spmm(handle,
                                 GetTransposeOperation(transa),
                                 GetTransposeOperation(transb),
                                 &alpha,
                                 a_descriptor.descriptor(),
                                 b_descriptor.descriptor(),
                                 &beta,
                                 out_descriptor.descriptor(),
                                 ttype,
                                 GetSpMMAlgorithm(mat_a),
                                 rocsparse_spmm_stage_buffer_size,
                                 &buffer_size,
                                 nullptr);
  });

  // Allocate buffer

  phi::Allocator::AllocationPtr tmp_buffer = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_ptr = tmp_buffer->ptr();
  // Preprocess data
  dev_ctx_.CusparseCall([&](rocsparse_handle handle) {
    phi::dynload::rocsparse_spmm(handle,
                                 GetTransposeOperation(transa),
                                 GetTransposeOperation(transb),
                                 &alpha,
                                 a_descriptor.descriptor(),
                                 b_descriptor.descriptor(),
                                 &beta,
                                 out_descriptor.descriptor(),
                                 ttype,
                                 GetSpMMAlgorithm(mat_a),
                                 rocsparse_spmm_stage_preprocess,
                                 tmp_buffer_ptr);
  });

  // Performs the actual SpMM computation
  dev_ctx_.CusparseCall([&](rocsparse_handle handle) {
    phi::dynload::rocsparse_spmm(handle,
                                 GetTransposeOperation(transa),
                                 GetTransposeOperation(transb),
                                 &alpha,
                                 a_descriptor.descriptor(),
                                 b_descriptor.descriptor(),
                                 &beta,
                                 out_descriptor.descriptor(),
                                 ttype,
                                 GetSpMMAlgorithm(mat_a),
                                 rocsparse_spmm_stage_compute,
                                 tmp_buffer_ptr);
  });
}
}  // namespace sparse
}  // namespace funcs
}  // namespace phi
