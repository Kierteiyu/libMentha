/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "lib_mentha/core/kernels/tiny_deconv2d_kernel.h"
#include "lib_mentha/core/params/deconv_params.h"

namespace lib_mentha {
namespace core {
namespace kernels {

inline void avx_deconv2d_kernel(const deconv_params &params,
                                const tensor_t &in,
                                const vec_t &W,
                                const vec_t &bias,
                                tensor_t &out,
                                const bool layer_parallelize) {
  // fallback to non-avx version
  tiny_deconv2d_kernel(params, in, W, bias, out, layer_parallelize);
}

}  // namespace kernels
}  // namespace core
}  // namespace lib_mentha
