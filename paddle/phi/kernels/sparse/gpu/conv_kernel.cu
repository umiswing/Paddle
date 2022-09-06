/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/conv_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/funcs/sparse/scatter.cu.h"
#include "paddle/phi/kernels/sparse/gpu/conv.cu.h"
#ifdef PADDLE_WITH_CUTLASS
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/funcs/slice.h"
#endif

#include "glog/logging.h"

namespace phi {
namespace sparse {

#ifdef PADDLE_WITH_CUTLASS
template <typename T>
void AlignFeaturesChannel(const GPUContext& dev_ctx,
                          const SparseCooTensor& x,
                          SparseCooTensor& aligned_x,
                          int channel_idx,
                          int padding_num) {
  if (padding_num <= 0) {
    aligned_x = x;
    return;
  }
  if (padding_num > 0) {
    int features_channel_idx = -1;
    phi::DenseTensor features = x.non_zero_elements();
    phi::DenseTensor paddings = phi::Empty<T>(dev_ctx, {x.nnz(), padding_num});
    phi::DenseTensor aligned_features =
        phi::Empty<T>(dev_ctx, {x.nnz(), x.dims()[channel_idx] + padding_num});
    ConcatKernel<T, GPUContext>(
        dev_ctx,
        std::vector<const DenseTensor*>{&features, &paddings},
        features_channel_idx,
        &aligned_features);
    DDim aligned_dims(x.dims());
    aligned_dims[channel_idx] += padding_num;
    aligned_x =
        SparseCooTensor(x.non_zero_indices(), aligned_features, aligned_dims);
  }
}

template <typename T>
void AlignKernelChannel(const GPUContext& dev_ctx,
                        const DenseTensor& kernel,
                        DenseTensor& aligned_kernel,
                        int channel_idx,
                        int padding_num) {
  if (padding_num <= 0) {
    aligned_kernel = kernel;
    return;
  }
  if (padding_num > 0) {
    DDim padding_dims = kernel.dims();
    padding_dims[channel_idx] = padding_num;
    DenseTensor paddings = phi::Empty<T>(dev_ctx,
                                         {
                                             padding_dims[0],
                                             padding_dims[1],
                                             padding_dims[2],
                                             padding_dims[3],
                                             padding_dims[4],
                                         });
    ConcatKernel<T, GPUContext>(
        dev_ctx,
        std::vector<const DenseTensor*>{&kernel, &paddings},
        channel_idx,
        &aligned_kernel);
  }
}

template <typename T>
void AlignKernelChannels(const GPUContext& dev_ctx,
                         const DenseTensor& kernel,
                         DenseTensor& aligned_kernel,
                         int channel_in_idx,
                         int channel_out_idx,
                         int in_padding_num,
                         int out_padding_num) {
  DenseTensor aligned_in_kernel;
  AlignKernelChannel<T>(
      dev_ctx, kernel, aligned_in_kernel, channel_in_idx, in_padding_num);
  AlignKernelChannel<T>(dev_ctx,
                     aligned_in_kernel,
                     aligned_kernel,
                     channel_out_idx,
                     out_padding_num);
}
#endif
template <typename T, typename IntT>
void Conv3dCooGPUKernel(const GPUContext& dev_ctx,
                        const SparseCooTensor& x,
                        const DenseTensor& kernel,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        const int groups,
                        const bool subm,
                        const std::string& key,
                        SparseCooTensor* out,
                        DenseTensor* rulebook,
                        DenseTensor* counter) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)
  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  DDim out_dims = {1, 1, 1, 1, 1};
  std::vector<int> kernel_sizes(kernel_dims.size());
  for (int i = 0; i < kernel_dims.size(); i++) {
    kernel_sizes[i] = kernel_dims[i];
  }

  std::vector<int> subm_paddings(paddings), subm_strides(strides);
  if (subm) {
    // the out shape of subm_conv is same as input shape
    // reset the padding=kernel_size/2 and strides=1
    phi::funcs::sparse::ResetSubmKernelSizeAndStrides(
        kernel.dims(), &subm_paddings, &subm_strides);
  }

  phi::funcs::sparse::GetOutShape(
      x_dims, kernel_sizes, subm_paddings, dilations, subm_strides, &out_dims);
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];
  DenseTensor h_counter, h_offsets;
  h_counter.Resize({kernel_size + 1});
  h_offsets.Resize({kernel_size + 1});
  int* h_counter_ptr = dev_ctx.template HostAlloc<int>(&h_counter);
  int* h_offsets_ptr = dev_ctx.template HostAlloc<int>(&h_offsets);

  // Second algorithm:
  // https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf
  // 1. product rulebook
  DenseTensor counter_per_kernel = phi::Empty<int>(dev_ctx, {kernel_size + 1});
  DenseTensor offsets_per_kernel = phi::Empty<int>(dev_ctx, {kernel_size});
  DenseTensor out_index = phi::Empty<int>(dev_ctx, {1});
  DenseTensor unique_value = phi::Empty<int>(dev_ctx, {1});

  VLOG(6) << "call SubmConv3D or Conv3D " << subm << " and the key is " << key;
  int rulebook_len = 0;
  const IntT* rulebook_ptr = nullptr;
  bool need_product_rulebook = true;
  if (subm && !key.empty()) {
    rulebook_ptr = phi::funcs::sparse::PrepareSubm<T, IntT, GPUContext>(
        dev_ctx,
        x,
        key,
        out_dims,
        out,
        h_counter.data<int>(),
        h_offsets.data<int>(),
        &rulebook_len,
        &need_product_rulebook);
  }

  if (need_product_rulebook) {
    DenseTensor tmp_rulebook;
    rulebook_len = ProductRuleBook<T, GPUContext, IntT>(dev_ctx,
                                                        x,
                                                        kernel_sizes,
                                                        subm_paddings,
                                                        dilations,
                                                        subm_strides,
                                                        out_dims,
                                                        subm,
                                                        &tmp_rulebook,
                                                        &counter_per_kernel,
                                                        &offsets_per_kernel,
                                                        &out_index,
                                                        &unique_value,
                                                        out,
                                                        h_counter_ptr,
                                                        h_offsets_ptr);
    rulebook_ptr = tmp_rulebook.data<IntT>();

    phi::funcs::sparse::SaveToTable(
        dev_ctx, x, key, tmp_rulebook, h_counter, out, rulebook, counter);
  }
#ifdef PADDLE_WITH_CUTLASS
  // padding to make leading dimension of x, kernel and out 128 bits aligned.
  // kernel.layout is x,y,z,c_in,c_out.
  // x.layout and out.layout are NDHWC.

  int kernel_channel_in_idx = 3;
  int kernel_channel_out_idx = 4;
  int channel_idx = 4;
  int align_num = 128 / cutlass::sizeof_bits<T>::value;
  int channel_out_dim = kernel.dims()[kernel_channel_out_idx];
  int64_t in_padding_num = 0;
  int64_t out_padding_num = 0;

  if (kernel.dims()[kernel_channel_in_idx] % align_num != 0) {
    in_padding_num =
        align_num - kernel.dims()[kernel_channel_in_idx] % align_num;
  }

  if (kernel.dims()[kernel_channel_out_idx] % align_num != 0) {
    out_padding_num =
        align_num - kernel.dims()[kernel_channel_out_idx] % align_num;
  }

  if(in_padding_num==0 && out_padding_num==0) {
    auto* out_values = out->mutable_non_zero_elements();
    T* out_values_ptr = out_values->data<T>();
    phi::funcs::SetConstant<GPUContext, T> set_zero;
    set_zero(dev_ctx, out_values, static_cast<T>(0.0f));

    const T* kernel_ptr = kernel.data<T>();
    for (int i = 0; i < kernel_size; i++) {
      if (h_counter_ptr[i] <= 0) {
        continue;
      }

      const int M = h_counter_ptr[i];
      const int K = in_channels;
      const int N = out_channels;
      const T* tmp_kernel_ptr = kernel_ptr + i * K * N;
      const IntT* gather_indices = rulebook_ptr + h_offsets_ptr[i];
      const IntT* scatter_indices =
          rulebook_ptr + rulebook_len + h_offsets_ptr[i];

      gather_gemm_scatter<T, T, T, T, T>(x.non_zero_elements().data<T>(),
                                         tmp_kernel_ptr,
                                         out_values_ptr,
                                         out_values_ptr,
                                         M,
                                         N,
                                         K,
                                         gather_indices,
                                         scatter_indices,
                                         h_counter_ptr[i],
                                         static_cast<T>(1),
                                         static_cast<T>(1));
    }
  } else {
    SparseCooTensor aligned_x;
    DenseTensor aligned_features =
        phi::Empty<T>(dev_ctx, {out->nnz(), channel_out_dim + out_padding_num});
    DenseTensor aligned_kernel;
    AlignFeaturesChannel<T>(dev_ctx, x, aligned_x, channel_idx, in_padding_num);
    AlignKernelChannels<T>(dev_ctx,
                           kernel,
                           aligned_kernel,
                           kernel_channel_in_idx,
                           kernel_channel_out_idx,
                           in_padding_num,
                           out_padding_num);

    T* aligned_out_values_ptr = aligned_features.data<T>();

    const T* aligned_kernel_ptr = aligned_kernel.data<T>();
    for (int i = 0; i < kernel_size; i++) {
      if (h_counter_ptr[i] <= 0) {
        continue;
      }

      const int M = h_counter_ptr[i];
      const int K = in_channels + in_padding_num;
      const int N = out_channels + out_padding_num;
      const T* tmp_kernel_ptr = aligned_kernel_ptr + i * K * N;
      const IntT* gather_indices = rulebook_ptr + h_offsets_ptr[i];
      const IntT* scatter_indices =
          rulebook_ptr + rulebook_len + h_offsets_ptr[i];

      gather_gemm_scatter<T, T, T, T, T>(
          aligned_x.non_zero_elements().data<T>(),
          tmp_kernel_ptr,
          aligned_out_values_ptr,
          aligned_out_values_ptr,
          M,
          N,
          K,
          gather_indices,
          scatter_indices,
          h_counter_ptr[i],
          static_cast<T>(1),
          static_cast<T>(1));
    }
    DenseTensor out_features =
        phi::funcs::Slice<T, GPUContext>(dev_ctx,
                                         aligned_features,
                                         std::vector<int>{1},
                                         std::vector<int>{0},
                                         std::vector<int>{channel_out_dim});
    out->SetMember(
        out->non_zero_indices(), out_features, out->dims(), out->coalesced());
  }
#else
  // 2. gather
  phi::DenseTensor in_features =
      phi::Empty<T>(dev_ctx, {rulebook_len, in_channels});
  phi::DenseTensor out_features =
      phi::Empty<T>(dev_ctx, {rulebook_len, out_channels});
  T* in_features_ptr = in_features.data<T>();
  T* out_features_ptr = out_features.data<T>();
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  set_zero(dev_ctx, &out_features, static_cast<T>(0.0f));

  Gather<T, IntT>(dev_ctx,
                  x.values().data<T>(),
                  rulebook_ptr,
                  rulebook_len,
                  in_channels,
                  in_features_ptr);

  // 3. call gemm for every werght
  auto blas = phi::funcs::GetBlas<GPUContext, T>(dev_ctx);
  auto* out_values = out->mutable_values();
  T* out_values_ptr = out_values->data<T>();
  set_zero(dev_ctx, out_values, static_cast<T>(0.0f));

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (h_counter_ptr[i] <= 0) {
      continue;
    }

    // call gemm: (n, in_channels) * (in_channels, out_channels)
    const int M = h_counter_ptr[i];
    const int K = in_channels;
    const int N = out_channels;
    T* tmp_in_ptr = in_features_ptr + h_offsets_ptr[i] * in_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * K * N;
    T* tmp_out_ptr = out_features_ptr + h_offsets_ptr[i] * out_channels;

    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              tmp_in_ptr,
              tmp_kernel_ptr,
              static_cast<T>(0),
              tmp_out_ptr);
  }

  // 4. scatter
  phi::funcs::sparse::ScatterV2<T>(dev_ctx,
                                   out_features_ptr,
                                   out_index.data<int>(),
                                   unique_value.data<int>(),
                                   out->nnz(),
                                   kernel_size,
                                   h_counter_ptr[kernel_size],
                                   out_channels,
                                   1,
                                   out_values_ptr);
#endif
}

/**
 * x: the input SparseCooTensor, shape is (N, D, H, W, C)
 * kernel: the weight data, shape is (D, H, W, C, OC)
 * out: the output SparseCooTensor, shape is (N, D, H, W, OC)
 * rulebook: return rulebook if key is not vailed else return nullptr
 * counter: return counter if key is not vailed else return nullptr
 **/
template <typename T, typename Context>
void Conv3dCooKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const DenseTensor& kernel,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const int groups,
                     const bool subm,
                     const std::string& key,
                     SparseCooTensor* out,
                     DenseTensor* rulebook,
                     DenseTensor* counter) {
#ifdef PADDLE_WITH_CUTLASS
  Conv3dCooGPUKernel<T, int32_t>(dev_ctx,
                                 x,
                                 kernel,
                                 paddings,
                                 dilations,
                                 strides,
                                 groups,
                                 subm,
                                 key,
                                 out,
                                 rulebook,
                                 counter);
#else
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "Conv3dCooGPUKernel", ([&] {
                                 Conv3dCooGPUKernel<T, data_t>(dev_ctx,
                                                               x,
                                                               kernel,
                                                               paddings,
                                                               dilations,
                                                               strides,
                                                               groups,
                                                               subm,
                                                               key,
                                                               out,
                                                               rulebook,
                                                               counter);
                               }));
#endif
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(conv3d_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dCooKernel,
#ifdef PADDLE_WITH_CUTLASS
                   float) {
#else
                   float,
                   double,
                   phi::dtype::float16) {
#endif
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
