/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/raw_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace framework {
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

const framework::FeedType& CheckAndGetFeedItem(const phi::ExtendedTensor& x,
                                               int col) {
  PADDLE_ENFORCE_GE(col,
                    0,
                    platform::errors::InvalidArgument(
                        "Expected the column index (the attribute 'col' of "
                        "operator 'Feed') of current feeding variable to be "
                        "no less than 0. But received column index = %d.",
                        col));
  auto feed_list = static_cast<const paddle::framework::FeedList*>(&x);
  PADDLE_ENFORCE_LT(
      static_cast<size_t>(col),
      feed_list->size(),
      platform::errors::InvalidArgument(
          "The column index of current feeding variable is expected to be "
          "less than the length of feeding list. But received column index = "
          "%d, the length of feeding list = %d",
          col,
          feed_list->size()));

  return feed_list->at(static_cast<size_t>(col));
}

template <typename Context>
void FeedDenseTensorKernel(const Context& dev_ctx,
                           const phi::ExtendedTensor& x,
                           int col,
                           phi::DenseTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      platform::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item = CheckAndGetFeedItem(x, col);
  const auto& in_tensor = paddle::get<phi::DenseTensor>(feed_item);
  const auto& place = dev_ctx.GetPlace();
  if (platform::is_same_place(in_tensor.place(), place)) {
    out->ShareDataWith(in_tensor);
  } else {
    framework::TensorCopy(in_tensor, place, dev_ctx, out);
  }

  out->set_lod(in_tensor.lod());
}

template <typename Context>
void FeedSparseCooTensorKernel(const Context& dev_ctx,
                               const phi::ExtendedTensor& x,
                               int col,
                               phi::SparseCooTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      platform::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item = CheckAndGetFeedItem(x, col);
  const auto& in_tensor = paddle::get<phi::SparseCooTensor>(feed_item);
  const auto& place = dev_ctx.GetPlace();
  if (platform::is_same_place(in_tensor.place(), place)) {
    *out = in_tensor;
  } else {
    phi::DenseTensor indices, values;
    framework::TensorCopy(in_tensor.indices(), place, dev_ctx, &indices);
    framework::TensorCopy(in_tensor.values(), place, dev_ctx, &values);
    out->SetMember(indices, values, in_tensor.meta());
  }
}

template <typename Context>
void FeedStringsKernel(const Context& dev_ctx,
                       const phi::ExtendedTensor& x,
                       int col,
                       phi::ExtendedTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      platform::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item = CheckAndGetFeedItem(x, col);
  auto strs_out = static_cast<framework::Strings*>(out);
  const auto& in_str = paddle::get<framework::Strings>(feed_item);
  strs_out->resize(in_str.size());
  *strs_out = in_str;
}

class FeedOp : public framework::OperatorWithKernel {
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "feed");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "feed");
    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("X")[0]);
      auto& x = x_var->Get<framework::FeedList>();
      int col = ctx->Attrs().Get<int>("col");
      auto& feed_item = x[col];
      if (feed_item.index() == 0) {
        const auto& feed_item = CheckAndGetFeedItem(x, col);
        auto& feed_tensor = PADDLE_GET_CONST(phi::DenseTensor, feed_item);
        ctx->SetOutputDim("Out", feed_tensor.dims());
      } else if (feed_item.index() == 1) {
        auto& feed_str = PADDLE_GET_CONST(framework::Strings, feed_item);
        framework::Variable* out_var =
            PADDLE_GET(framework::Variable*, ctx->GetOutputVarPtrs("Out")[0]);
        out_var->GetMutable<framework::Strings>()->resize(feed_str.size());
      } else {
        auto& feed_sparse_tensor =
            PADDLE_GET_CONST(phi::SparseCooTensor, feed_item);
        framework::Variable* out_var =
            PADDLE_GET(framework::Variable*, ctx->GetOutputVarPtrs("Out")[0]);
        out_var->GetMutable<phi::SparseCooTensor>()->set_meta(
            feed_sparse_tensor.meta());
        out_var->GetMutable<phi::SparseCooTensor>()->SetCoalesced(
            feed_sparse_tensor.coalesced());
        out_var->GetMutable<phi::SparseCooTensor>()->SetIndicesDict(
            feed_sparse_tensor.GetIndicesDict());
      }
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }
};

class FeedOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<phi::DenseTensor>) "
             "A feeding list of phi::DenseTensor, which may have "
             "different dimension and data type.");
    AddOutput("Out",
              "(phi::DenseTensor) The phi::DenseTensor which is a copy "
              "of the col-th feeding "
              "object.");
    AddAttr<int>("col", "(int) The column index of current feeding object.");
    AddComment(R"DOC(
Feed Operator.
It should not be configured by users directly.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

// TODO(YuanRisheng): Maybe we need design a new registry macro for
// registering device independent kernels.

REGISTER_OPERATOR(
    feed,
    paddle::operators::FeedOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::FeedOpInfoMaker);

PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    CPU,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::CPUContext>,
    ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    CPU,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::CPUContext>,
    ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    CPU,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::CPUContext>,
    ALL_DTYPE) {}

#if defined(PADDLE_WITH_MKLDNN)
PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    OneDNN,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::OneDNNContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    OneDNN,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::OneDNNContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    OneDNN,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::OneDNNContext>,
    ALL_DTYPE) {}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    GPU,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::GPUContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    GPU,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::GPUContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    GPU,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::GPUContext>,
    ALL_DTYPE) {}
#elif defined(PADDLE_WITH_XPU)
PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    XPU,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::XPUContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    XPU,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::XPUContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    XPU,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::XPUContext>,
    ALL_DTYPE) {}
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace paddle {
namespace operators {
template void FeedDenseTensorKernel<phi::CustomContext>(
    const phi::CustomContext& dev_ctx,
    const phi::ExtendedTensor& x,
    int col,
    phi::DenseTensor* out);
}  // namespace operators
}  // namespace paddle
#endif
