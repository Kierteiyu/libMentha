/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>

#include "lib_mentha/activations/activation_layer.h"
#include "lib_mentha/core/params/params.h"

namespace lib_mentha {
namespace core {

class rnn_cell_params : public Params {
 public:
  size_t in_size_;
  size_t out_size_;
  std::shared_ptr<activation_layer> activation_{};
  bool has_bias_;
};

inline rnn_cell_params &Params::rnn_cell() {
  return *(static_cast<rnn_cell_params *>(this));
}

}  // namespace core
}  // namespace lib_mentha
