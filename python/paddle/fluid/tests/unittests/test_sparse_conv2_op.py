# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
import paddle
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard
import random
import spconv.pytorch as spconv
import torch
import logging


def generate_data(config):
    values = []
    indices = []
    for i in range(config['nnz']):
        value = []
        idx = []
        for j in range(config['in_channels']):
            value.append(random.uniform(-1, -0.0001)*random.choice([-1, 1]))
        values.append(value)

        idx.append(random.randrange(0, config['batch_size']))
        idx.append(random.randrange(0, config['x']))
        idx.append(random.randrange(0, config['y']))
        idx.append(random.randrange(0, config['z']))
        indices.append(idx)
    return values, indices


class TestSparseConv(unittest.TestCase):
    def test_conv3d(self):
        with _test_eager_guard():
            config = {
                'batch_size': 8,
                'x': 41,
                'y': 1600,
                'z': 1408,
                'kx': 3,
                'ky': 3,
                'kz': 3,
                'kernel_dim': (3, 3, 3),
                'in_channels': 4,
                'out_channels': 16,
                'nnz': 136000,
                'paddings': (0, 0, 0),
                'strides': (1, 1, 1),
                'dilations': (1, 1, 1),
                'diff': 1e-3
            }

            values, indices = generate_data(config)

            spatial_shape = [config['x'], config['y'], config['z']]
            s_values = torch.tensor(values).cuda()
            s_indices = torch.tensor(indices).int().cuda()
            s_input = spconv.SparseConvTensor(
                s_values, s_indices, spatial_shape, config['batch_size'])
            s_kernel = spconv.SparseConv3d(config['in_channels'], config['out_channels'], kernel_size=(
                config['kx'], config['ky'], config['kz']), stride=config['strides'], padding=config['paddings'], dilation=config['dilations'], bias=False)
            s_kernel.weight = torch.nn.Parameter(s_kernel.weight.cuda())
            nd_kernel = s_kernel.weight.movedim(0, 4).cpu().detach().numpy()
            s_out = s_kernel(s_input)

            p_shape = [config['batch_size'], config['x'],
                       config['y'], config['z'], config['in_channels']]
            p_kernel = paddle.to_tensor(
                nd_kernel, dtype='float32', stop_gradient=True)
            p_indices = paddle.to_tensor(indices, dtype='int32')
            p_indices = paddle.transpose(p_indices, perm=[1, 0])
            p_values = paddle.to_tensor(values, dtype='float32')
            p_input = core.eager.sparse_coo_tensor(p_indices, p_values,
                                                   p_shape, False)
            p_out = _C_ops.final_state_sparse_conv3d(p_input, p_kernel,
                                                   config['paddings'], config['dilations'], config['strides'],
                                                   1, False)

            assert np.array_equal(
                s_out.indices.cpu().detach().numpy().transpose(1, 0), p_out.indices().numpy())

            s_out_features_nd = s_out.features.cpu().detach().numpy().flatten()
            p_out_features_nd = p_out.values().numpy().flatten()

            for i in range(s_out_features_nd.size):
                try:
                    assert abs(s_out_features_nd[i] -
                               p_out_features_nd[i]) < config['diff']
                except AssertionError:
                    logging.exception(
                        "fails here:", i, s_out_features_nd[i], p_out_features_nd[i])
                    raise