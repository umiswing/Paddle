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

// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

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

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/numeric_conversion.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_cutlass_kernel.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_gemm_kernels.h"
#include "paddle/phi/kernels/fusion/cutlass/moe/fused_moe_helper.h"

#pragma GCC diagnostic pop

namespace phi {

namespace fusion {

template <typename T, typename Context>
void MoeReduceKernel(
    const Context& ctx,
    const DenseTensor& fc2_result,  // ffn output [num_rows * topk, hidden_dim]
    const paddle::optional<DenseTensor>& fc2_expert_biases,
    const DenseTensor&
        expert_scales_float,  // 对于每个token来说，不同专家对他的weight
    const DenseTensor& expanded_source_row_to_expanded_dest_row,
    const DenseTensor& topk_indices,
    const bool norm_topk_prob,
    DenseTensor* output) {
  const int topk = topk_indices.dims()[1];
  const int num_rows = fc2_result.dims()[0] / topk;
  const int hidden_size = fc2_result.dims()[1];
  output->Resize({num_rows, hidden_size});

  finalize_moe_routing_kernelLauncher(
      fc2_result.data<T>(),
      ctx.template Alloc<T>(output),
      fc2_expert_biases ? fc2_expert_biases->data<T>() : nullptr,
      expert_scales_float.data<float>(),
      expanded_source_row_to_expanded_dest_row.data<int32_t>(),
      topk_indices.data<int>(),
      num_rows,
      hidden_size,
      topk,
      static_cast<int>(1),
      norm_topk_prob,
      ctx.stream());
}

}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_CUDA_BF16
PD_REGISTER_KERNEL(moe_reduce,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeReduceKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(moe_reduce,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeReduceKernel,
                   phi::dtype::float16) {}
#endif
