/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "lib_mentha/core/framework/op_kernel.h"
#include "lib_mentha/core/kernels/global_avepool_op_avx.h"
#include "lib_mentha/core/kernels/global_avepool_op_internal.h"

namespace lib_mentha {

class GlobalAvePoolGradOp : public core::OpKernel {
 public:
  explicit GlobalAvePoolGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->global_avepool();

    // incoming/outcoming data
    tensor_t &prev_delta = context.input_grad(0);
    tensor_t &curr_delta = context.output_grad(0);

    // initialize outputs
    fill_tensor(prev_delta, float_t{0});

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::avx) {
#ifdef CNN_USE_AVX
      kernels::global_avepool_grad_op_avx(prev_delta, curr_delta, params,
                                          context.parallelize());
#endif
    } else {
      kernels::global_avepool_grad_op_internal(prev_delta, curr_delta, params,
                                               context.parallelize());
    }
  }
};

}  // namespace lib_mentha
