#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.common.test_utils import run_test
from iron.operators.repeat.op import Repeat


def _generate_golden_reference(rows, cols, repeat):
    torch.manual_seed(42)
    val_range = 4
    input_tensor = torch.rand(rows, cols, dtype=torch.bfloat16) * val_range
    output_tensor = torch.repeat_interleave(input_tensor, repeats=repeat, dim=0)
    return {"input": input_tensor, "output": output_tensor}


def get_params():
    return [
        pytest.param(4, 64, 2, id="grouped_attention_small"),
        pytest.param(4, 128, 4, marks=pytest.mark.extensive, id="llama_kv_groups"),
        pytest.param(2, 256, 4, marks=pytest.mark.extensive, id="wide_rows"),
    ]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize("rows,cols,repeat", get_params())
def test_repeat(rows, cols, repeat, aie_context):
    golden_ref = _generate_golden_reference(rows=rows, cols=cols, repeat=repeat)

    operator = Repeat(
        rows=rows,
        cols=cols,
        repeat=repeat,
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
