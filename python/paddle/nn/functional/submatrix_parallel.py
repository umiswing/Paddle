# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import _C_ops


def smp_col_row_row_linear(x, weight, bias, group, low_memory=False):
    ring_id = group.id

    transpose_weight = x.shape[1] != weight.shape[0]

    out = _C_ops.smp_col_row_row_linear(
        x, weight, bias, transpose_weight, low_memory, ring_id
    )
    return out


def smp_col_row_row_linear_grad(
    dy,
    x,
    weight,
    group,
    require_dx,
    require_dw,
    require_db,
    low_memory=True,
):
    ring_id = group.id
    # umiswing: luckily, weight is never transposed in tested case,
    # but it's necessary to support transposed weight

    dx, dw, db = _C_ops.smp_col_row_row_linear_grad(
        dy, x, weight, low_memory, require_dx, require_dw, require_db, ring_id
    )
    return dx, dw, db


def smp_row_col_col_linear(x, weight, bias, group, return_x):
    ring_id = group.id

    transpose_weight = x.shape[1] != weight.shape[0]

    out, global_x = _C_ops.smp_row_col_col_linear(
        x, weight, bias, transpose_weight, return_x, ring_id
    )
    if return_x:
        return out, global_x
    else:
        return out


def smp_row_col_col_linear_grad(
    dy, x, weight, group, require_dx, require_dw, require_db
):
    ring_id = group.id
    # umiswing: luckily, weight is never transposed in tested case,
    # but it's necessary to support transposed weight

    dx, dw, db = _C_ops.smp_row_col_col_linear_grad(
        dy, x, weight, require_dx, require_dw, require_db, ring_id
    )
    return dx, dw, db
