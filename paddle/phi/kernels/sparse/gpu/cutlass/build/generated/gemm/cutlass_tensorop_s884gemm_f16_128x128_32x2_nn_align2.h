
/*
  Generated by gemm_operation.py - Do not edit.
*/
#pragma once
#ifdef PADDLE_WITH_CUTLASS

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "library_internal.h"
#include "gemm_operation.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


namespace phi {
namespace sparse {

// Gemm operator cutlass_tensorop_s884gemm_f16_128x128_32x2_nn_align2
struct cutlass_tensorop_s884gemm_f16_128x128_32x2_nn_align2 {
  using Gemm =
    cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm70,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<8, 8, 4>,
      
    cutlass::epilogue::thread::LinearCombination<
      float,
      2,
      float,
      float
    >
,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      2,
      2,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true, // gather a
      false, // gather b
      true // scatter d
    >;
};

}  // namespace sparse
}  // namespace phi
#endif
