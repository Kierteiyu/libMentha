/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include "lib_mentha/activations/sigmoid_layer.h"
#include "lib_mentha/activations/tanh_layer.h"
#include "lib_mentha/core/params/params.h"

namespace lib_mentha {
namespace core {

class gru_cell_params : public Params {
 public:
  size_t in_size_;
  size_t out_size_;
  std::shared_ptr<tanh_layer> tanh_;
  std::shared_ptr<sigmoid_layer> sigmoid_;
  bool has_bias_;
};

inline gru_cell_params &Params::gru_cell() {
  return *(static_cast<gru_cell_params *>(this));
}

}  // namespace core
}  // namespace lib_mentha
