/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "lib_mentha/config.h"
#include "lib_mentha/network.h"
#include "lib_mentha/nodes.h"

#include "lib_mentha/core/framework/tensor.h"

#include "lib_mentha/core/framework/device.h"
#include "lib_mentha/core/framework/program_manager.h"

#include "lib_mentha/activations/asinh_layer.h"
#include "lib_mentha/activations/elu_layer.h"
#include "lib_mentha/activations/leaky_relu_layer.h"
#include "lib_mentha/activations/relu_layer.h"
#include "lib_mentha/activations/selu_layer.h"
#include "lib_mentha/activations/sigmoid_layer.h"
#include "lib_mentha/activations/softmax_layer.h"
#include "lib_mentha/activations/softplus_layer.h"
#include "lib_mentha/activations/softsign_layer.h"
#include "lib_mentha/activations/tanh_layer.h"
#include "lib_mentha/activations/tanh_p1m2_layer.h"
#include "lib_mentha/layers/arithmetic_layer.h"
#include "lib_mentha/layers/average_pooling_layer.h"
#include "lib_mentha/layers/average_unpooling_layer.h"
#include "lib_mentha/layers/batch_normalization_layer.h"
#include "lib_mentha/layers/cell.h"
#include "lib_mentha/layers/cells.h"
#include "lib_mentha/layers/concat_layer.h"
#include "lib_mentha/layers/convolutional_layer.h"
#include "lib_mentha/layers/deconvolutional_layer.h"
#include "lib_mentha/layers/dropout_layer.h"
#include "lib_mentha/layers/fully_connected_layer.h"
#include "lib_mentha/layers/global_average_pooling_layer.h"
#include "lib_mentha/layers/input_layer.h"
#include "lib_mentha/layers/l2_normalization_layer.h"
#include "lib_mentha/layers/linear_layer.h"
#include "lib_mentha/layers/lrn_layer.h"
#include "lib_mentha/layers/max_pooling_layer.h"
#include "lib_mentha/layers/max_unpooling_layer.h"
#include "lib_mentha/layers/power_layer.h"
#include "lib_mentha/layers/quantized_convolutional_layer.h"
#include "lib_mentha/layers/quantized_deconvolutional_layer.h"
#include "lib_mentha/layers/recurrent_layer.h"
#include "lib_mentha/layers/slice_layer.h"
#include "lib_mentha/layers/zero_pad_layer.h"

#ifdef CNN_USE_GEMMLOWP
#include "lib_mentha/layers/quantized_fully_connected_layer.h"
#endif  // CNN_USE_GEMMLOWP

#include "lib_mentha/lossfunctions/loss_function.h"
#include "lib_mentha/optimizers/optimizer.h"

#include "lib_mentha/util/deform.h"
#include "lib_mentha/util/graph_visualizer.h"
#include "lib_mentha/util/product.h"
#include "lib_mentha/util/weight_init.h"
#include "lib_mentha/util/nms.h"

#include "lib_mentha/io/cifar10_parser.h"
#include "lib_mentha/io/display.h"
#include "lib_mentha/io/layer_factory.h"
#include "lib_mentha/io/mnist_parser.h"

#ifndef CNN_NO_SERIALIZATION
#include "lib_mentha/util/deserialization_helper.h"
#include "lib_mentha/util/serialization_helper.h"
// to allow upcasting
CEREAL_REGISTER_TYPE(lib_mentha::elu_layer)
CEREAL_REGISTER_TYPE(lib_mentha::leaky_relu_layer)
CEREAL_REGISTER_TYPE(lib_mentha::relu_layer)
CEREAL_REGISTER_TYPE(lib_mentha::sigmoid_layer)
CEREAL_REGISTER_TYPE(lib_mentha::softmax_layer)
CEREAL_REGISTER_TYPE(lib_mentha::softplus_layer)
CEREAL_REGISTER_TYPE(lib_mentha::softsign_layer)
CEREAL_REGISTER_TYPE(lib_mentha::tanh_layer)
CEREAL_REGISTER_TYPE(lib_mentha::tanh_p1m2_layer)
#endif  // CNN_NO_SERIALIZATION

// shortcut version of layer names
namespace lib_mentha {
namespace layers {

using conv = lib_mentha::convolutional_layer;

using q_conv = lib_mentha::quantized_convolutional_layer;

using max_pool = lib_mentha::max_pooling_layer;

using ave_pool = lib_mentha::average_pooling_layer;

using fc = lib_mentha::fully_connected_layer;

using dense = lib_mentha::fully_connected_layer;

using zero_pad = lib_mentha::zero_pad_layer;

// using rnn_cell = lib_mentha::rnn_cell_layer;

#ifdef CNN_USE_GEMMLOWP
using q_fc = lib_mentha::quantized_fully_connected_layer;
#endif

using add = lib_mentha::elementwise_add_layer;

using dropout = lib_mentha::dropout_layer;

using input = lib_mentha::input_layer;

using linear = lib_mentha::linear_layer;

using lrn = lib_mentha::lrn_layer;

using concat = lib_mentha::concat_layer;

using deconv = lib_mentha::deconvolutional_layer;

using max_unpool = lib_mentha::max_unpooling_layer;

using ave_unpool = lib_mentha::average_unpooling_layer;

}  // namespace layers

namespace activation {

using sigmoid = lib_mentha::sigmoid_layer;

using asinh = lib_mentha::asinh_layer;

using tanh = lib_mentha::tanh_layer;

using relu = lib_mentha::relu_layer;

using rectified_linear = lib_mentha::relu_layer;

using softmax = lib_mentha::softmax_layer;

using leaky_relu = lib_mentha::leaky_relu_layer;

using elu = lib_mentha::elu_layer;

using selu = lib_mentha::selu_layer;

using tanh_p1m2 = lib_mentha::tanh_p1m2_layer;

using softplus = lib_mentha::softplus_layer;

using softsign = lib_mentha::softsign_layer;

}  // namespace activation

#include "lib_mentha/models/alexnet.h"

using batch_norm = lib_mentha::batch_normalization_layer;

using l2_norm = lib_mentha::l2_normalization_layer;

using slice = lib_mentha::slice_layer;

using power = lib_mentha::power_layer;

}  // namespace lib_mentha

#ifdef CNN_USE_CAFFE_CONVERTER
// experimental / require google protobuf
#include "lib_mentha/io/caffe/layer_factory.h"
#endif
