/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "lib_mentha/core/params/params.h"

namespace lib_mentha {
namespace core {

class global_avepool_params : public Params {
 public:
  shape3d in;
  shape3d out;
};

inline global_avepool_params &Params::global_avepool() {
  return *(static_cast<global_avepool_params *>(this));
}

}  // namespace core
}  // namespace lib_mentha
