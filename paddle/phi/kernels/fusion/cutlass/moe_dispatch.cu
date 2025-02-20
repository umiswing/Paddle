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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "paddle/phi/backends/gpu/gpu_info.h"

#include "paddle/phi/kernels/fusion/cutlass/moe/fused_moe_helper.h"
static inline size_t zkk_AlignTo16(const size_t& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

#pragma GCC diagnostic pop

namespace phi {

namespace fusion {

template <typename T, typename Context>
void MoeDispatchKernel(const Context& ctx,
                       const DenseTensor& X,
                       const DenseTensor& gating_output,
                       const int moe_topk,

                       DenseTensor* out_feed_to_ffn,
                       DenseTensor* token_nums_per_expert,
                       DenseTensor* scatter_index,
                       DenseTensor* expert_scales_float,
                       // 下面这个就是返回每个token的专家的索引！结果很清晰！
                       DenseTensor* topk_expert_indices) {
  const int num_rows = X.dims()[0];
  const int hidden_size = X.dims()[1];
  const int expert_num = gating_output.dims()[1];

  // 这个变量表示每个token，含有moe_topk个数字，分别表示每个专家计算结果的加权系数！
  expert_scales_float->Resize({num_rows, moe_topk});

  DenseTensor finished_tensor = Empty<bool>(ctx, {num_rows});
  bool* finished = finished_tensor.data<bool>();
  // set false
  funcs::SetConstant<GPUContext, bool> zero;
  zero(ctx, &finished_tensor, false);

  const int num_moe_inputs = zkk_AlignTo16(num_rows * moe_topk);
  const int bytes = num_moe_inputs * sizeof(int);
  DenseTensor ws_ptr_tensor = Empty<int8_t>(ctx, {bytes});
  int8_t* ws_ptr = ws_ptr_tensor.data<int8_t>();
  int* source_rows_ = reinterpret_cast<int*>(ws_ptr);

  topk_expert_indices->Resize({num_rows, moe_topk});
  int* expert_for_source_row = ctx.template Alloc<int>(topk_expert_indices);

  DenseTensor tmp1 = Empty<float>(ctx, {num_rows * expert_num});
  float* softmax_out_ = tmp1.data<float>();

  // std::cout << num_rows <<" "<<  expert_num << "  " << moe_topk << std::endl;

  // comment: _, expert_for_source_row = paddle.topk(gating_output, moe_topk,
  // axis=-1)
  topk_gating_softmax_kernelLauncher<float>(
      gating_output.data<float>(),
      finished,
      ctx.template Alloc<float>(
          expert_scales_float),  // 每个token相对于每个专家的权重！
      softmax_out_,  // 这个是个临时空间，我先在不给他！
      // 每个token的专家的索引！我要把这么多数据按照专家来排个序！
      expert_for_source_row,
      // 这个是个输出，我理解就是每个专家的需要取的数据吧！,我觉得这个就是0，0，0，0,
      // 0,1,1,1,1,。。512,吧！
      source_rows_,

      nullptr,
      num_rows,
      expert_num,
      moe_topk,
      false,
      ctx.stream());

  CubKeyValueSorter sorter_(expert_num);

  const int sorter_ws_size_bytes =
      zkk_AlignTo16(sorter_.getWorkspaceSize(moe_topk * num_rows));
  DenseTensor sorter_ws = Empty<int8_t>(ctx, {sorter_ws_size_bytes});
  int8_t* sorter_ws_ptr = sorter_ws.data<int8_t>();

  DenseTensor tmp = Empty<int32_t>(ctx, {num_moe_inputs * 2});
  int* permuted_experts_ = tmp.data<int32_t>();
  int* permuted_rows_ = permuted_experts_ + num_moe_inputs;

  sorter_.run(reinterpret_cast<void*>(sorter_ws_ptr),
              sorter_ws_size_bytes,
              expert_for_source_row,  // in key【0，3，2】
              permuted_experts_,  // out 每个专家也被重新排序了【0，0,0,
                                  // 1,1,1,2,2,2,2，3,3,3,3】这样的！
              source_rows_,
              permuted_rows_,
              moe_topk * num_rows,
              false,
              ctx.stream());

  out_feed_to_ffn->Resize({moe_topk * num_rows, hidden_size});
  scatter_index->Resize({moe_topk, num_rows});

  initialize_moe_routing_kernelLauncher(
      X.data<T>(),
      ctx.template Alloc<T>(out_feed_to_ffn),
      permuted_rows_,  // 每个专家需要收集的数据吧！
      ctx.template Alloc<int32_t>(
          scatter_index),  // output 这个就是scatter_index ba !
      num_rows,
      num_rows,
      hidden_size,
      moe_topk,
      ctx.stream());

  token_nums_per_expert->Resize({expert_num});

  compute_total_rows_before_expert<T>(
      permuted_experts_,
      X.data<T>(),
      moe_topk * num_rows,
      expert_num,
      ctx.template Alloc<int64_t>(token_nums_per_expert),
      ctx.stream());
}

}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_CUDA_BF16
PD_REGISTER_KERNEL(moe_dispatch,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeDispatchKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(moe_dispatch,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeDispatchKernel,
                   phi::dtype::float16) {}
#endif
