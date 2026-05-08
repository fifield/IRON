#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
import torch

from iron.common.test_utils import run_test
from iron.operators.strided_copy.op import StridedCopy


def _linear_indices(sizes, strides, offset):
    for idx in product(*(range(size) for size in sizes)):
        yield offset + sum(i * stride for i, stride in zip(idx, strides))


def _generate_golden_reference(
    input_sizes,
    input_strides,
    input_offset,
    input_buffer_size,
    output_sizes,
    output_strides,
    output_offset,
    output_buffer_size,
):
    source = torch.arange(input_buffer_size, dtype=torch.float32).to(torch.bfloat16)
    expected = torch.zeros(output_buffer_size, dtype=torch.bfloat16)

    input_indices = _linear_indices(input_sizes, input_strides, input_offset)
    output_indices = _linear_indices(output_sizes, output_strides, output_offset)
    for input_idx, output_idx in zip(input_indices, output_indices):
        expected[output_idx] = source[input_idx]

    return {"input": source, "output": expected}


def get_params():
    return [
        pytest.param(
            (4, 64),
            (64, 1),
            0,
            256,
            (1, 4, 64),
            (0, 64, 1),
            0,
            256,
            1,
            id="llama_kv_token",
        ),
        pytest.param(
            (2, 64),
            (128, 1),
            16,
            272,
            (2, 64),
            (64, 1),
            0,
            128,
            1,
            marks=pytest.mark.extensive,
            id="gapped_input_rows",
        ),
        pytest.param(
            (4, 32),
            (64, 1),
            8,
            256,
            (4, 32),
            (32, 1),
            0,
            128,
            2,
            marks=pytest.mark.extensive,
            id="two_channels",
        ),
    ]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "input_sizes,input_strides,input_offset,input_buffer_size,"
    "output_sizes,output_strides,output_offset,output_buffer_size,"
    "num_aie_channels",
    get_params(),
)
def test_strided_copy(
    input_sizes,
    input_strides,
    input_offset,
    input_buffer_size,
    output_sizes,
    output_strides,
    output_offset,
    output_buffer_size,
    num_aie_channels,
    aie_context,
):
    golden_ref = _generate_golden_reference(
        input_sizes=input_sizes,
        input_strides=input_strides,
        input_offset=input_offset,
        input_buffer_size=input_buffer_size,
        output_sizes=output_sizes,
        output_strides=output_strides,
        output_offset=output_offset,
        output_buffer_size=output_buffer_size,
    )

    operator = StridedCopy(
        input_sizes=input_sizes,
        input_strides=input_strides,
        input_offset=input_offset,
        output_sizes=output_sizes,
        output_strides=output_strides,
        output_offset=output_offset,
        input_buffer_size=input_buffer_size,
        output_buffer_size=output_buffer_size,
        num_aie_channels=num_aie_channels,
        context=aie_context,
    )

    input_buffers = {"input": golden_ref["input"]}
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.01, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"


@pytest.mark.parametrize(
    "input_sizes,input_strides",
    [pytest.param((2, 4), (4,), id="input_sizes_mismatch")],
)
def test_strided_copy_rejects_mismatched_input_shape(input_sizes, input_strides):
    with pytest.raises(ValueError, match="input_sizes and input_strides"):
        StridedCopy(
            input_sizes=input_sizes,
            input_strides=input_strides,
            input_offset=0,
            output_sizes=(2, 4),
            output_strides=(4, 1),
            output_offset=0,
            input_buffer_size=8,
            output_buffer_size=8,
        )
