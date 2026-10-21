# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""PR #61355 packed GDN shapes on T-Head: Graph + Events, warm cache.

https://github.com/vllm-project/vllm/pull/61355
Only the T-Head packed implementation is timed. This is not CUPTI kernel
timing or a reproduction of the upstream cold-L2 protocol. Reported times are amortized us per update (32 calls per Graph).
Input construction and correctness checks are local to this file.
"""

import math
import statistics

import pytest
import torch

from benchmark.recording import BenchmarkRecorder
from flag_attn import runtime
from flag_attn.testing import backend as backend

BATCHES = (1, 2, 4, 8, 16, 32, 64)
STATE_DTYPES = (torch.float32, torch.bfloat16)
H, HV, K, V = 4, 12, 128, 128
ROUNDS, REPEAT_ITERS, DRY_RUN_ITERS = 7, 40, 5
INITIAL_WARMUP, ROUND_WARMUP = 10, 30
CALLS_PER_GRAPH = 32
SEED = 1234


def _assert_close(name, actual, expected):
    atol, rtol = (1e-4, 1e-4) if actual.dtype == torch.float32 else (2e-2, 1e-2)
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert torch.isfinite(actual).all(), f"{name}: nonfinite actual"
    assert torch.isfinite(expected).all(), f"{name}: nonfinite reference"
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{name}: elementwise check failed\n{message}",
    )
    diff = actual - expected
    ratio = diff.square().mean().sqrt().item() / (
        expected.square().mean().sqrt().item() + 1e-8
    )
    assert diff.abs().max().item() <= 1e-6 or ratio < 0.01, (
        f"{name}: relative_rms={ratio:.7g}, limit=0.01"
    )


@pytest.fixture(scope="module")
def benchmark_inputs():
    if not backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    if not backend.graph_available() or not hasattr(backend.device_fn, "Event"):
        pytest.fail("this benchmark requires validated T-Head Graph and Event support")
    # One generator spans the full matrix, matching the PR's construction order
    # even when pytest selects a subset of cases.
    generator = torch.Generator(device=backend.device).manual_seed(SEED)
    cases = {}
    for state_dtype in STATE_DTYPES:
        for batch in BATCHES:
            cases[state_dtype, batch] = dict(
                mixed_qkv=torch.randn(
                    batch, 2 * H * K + HV * V, device=backend.device,
                    dtype=torch.bfloat16, generator=generator,
                ) * 0.1,
                a=torch.randn(batch, HV, device=backend.device, dtype=torch.bfloat16, generator=generator),
                b=torch.randn(batch, HV, device=backend.device, dtype=torch.bfloat16, generator=generator),
                A_log=torch.zeros(HV, device=backend.device, dtype=torch.float32),
                dt_bias=torch.zeros(HV, device=backend.device, dtype=torch.float32),
                ssm_state_indices=torch.arange(1, batch + 1, device=backend.device, dtype=torch.int32),
                initial_state=torch.randn(
                    batch + 1, HV, V, K, device=backend.device,
                    dtype=state_dtype, generator=generator,
                ) * 0.1,
            )
    return cases


def _make_calls(data):
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd as ordinary,
    )
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent_packed_decode import (
        fused_recurrent_gated_delta_rule_packed_decode as packed,
    )

    batch = data["mixed_qkv"].shape[0]
    state = data["initial_state"].clone()
    ref_state = data["initial_state"].clone()
    out = torch.empty(batch, 1, HV, V, device=backend.device, dtype=torch.bfloat16)
    q, k, v = torch.split(data["mixed_qkv"], (H * K, H * K, HV * V), dim=-1)
    q = q.view(batch, 1, H, K).contiguous()
    k = k.view(batch, 1, H, K).contiguous()
    v = v.view(batch, 1, HV, V).contiguous()
    x = data["a"].float() + data["dt_bias"].float()
    softplus_x = torch.where(
        x <= 20, torch.log1p(torch.exp(torch.clamp(x, max=20))), x,
    )
    g = (-data["A_log"].float().exp() * softplus_x).unsqueeze(1)
    beta = data["b"].float().sigmoid().unsqueeze(1)

    def launch():
        return packed(
            mixed_qkv=data["mixed_qkv"], a=data["a"], b=data["b"],
            A_log=data["A_log"], dt_bias=data["dt_bias"], scale=K**-0.5,
            initial_state=state, out=out, ssm_state_indices=data["ssm_state_indices"],
            use_qk_l2norm_in_kernel=True,
        )

    def reference():
        return ordinary(
            q=q, k=k, v=v, g=g, beta=beta, scale=K**-0.5,
            initial_state=ref_state, inplace_final_state=True,
            ssm_state_indices=data["ssm_state_indices"], use_qk_l2norm_in_kernel=True,
        )

    return launch, reference, state, ref_state, out


def _capture(launch, calls=1):
    device = backend.device_fn
    stream = device.Stream()
    stream.wait_stream(device.current_stream())
    with device.stream(stream):
        launch()
    device.synchronize()
    graph = device.CUDAGraph()
    with device.graph(graph, stream=stream):
        for _ in range(calls):
            launch()
    device.current_stream().wait_stream(stream)
    device.synchronize()
    return graph


def _check_updates(call, reference, state, ref_state, out, initial):
    state.copy_(initial)
    ref_state.copy_(initial)
    for step in range(10):
        call()
        expected_out, expected_state = reference()
        if step in (0, 9):
            backend.device_fn.synchronize()
            _assert_close(f"output after {step + 1} updates", out, expected_out)
            _assert_close(f"state after {step + 1} updates", state, expected_state)
    assert torch.equal(state[0], initial[0]), "null state slot changed"


@torch.inference_mode()
def _measure(data, record_property):
    launch, reference, state, ref_state, out = _make_calls(data)
    initial = data["initial_state"]
    for _ in range(INITIAL_WARMUP):
        launch()
        reference()
    _check_updates(launch, reference, state, ref_state, out, initial)
    graph = _capture(launch)
    _check_updates(graph.replay, reference, state, ref_state, out, initial)

    del graph
    graph = _capture(launch, calls=CALLS_PER_GRAPH)
    state.copy_(initial)
    ref_state.copy_(initial)
    graph.replay()
    graph_out, graph_state = out.clone(), state.clone()
    state.copy_(initial)
    for _ in range(CALLS_PER_GRAPH):
        launch()
        expected_out, expected_state = reference()
    backend.device_fn.synchronize()
    assert torch.equal(graph_out, out), "Graph/eager output mismatch"
    assert torch.equal(graph_state, state), "Graph/eager 32-update state mismatch"
    _assert_close("Graph output after 32 updates", graph_out, expected_out)
    _assert_close("Graph state after 32 updates", graph_state, expected_state)
    assert torch.equal(state[0], initial[0]), "null state slot changed"
    del graph_out, graph_state

    device = backend.device_fn
    events = [(device.Event(enable_timing=True), device.Event(enable_timing=True))
              for _ in range(REPEAT_ITERS)]
    # Initialize lazily allocated event resources outside measured rounds.
    for start, end in events:
        start.record()
        end.record()
    device.synchronize()
    samples_by_round = []
    for _ in range(ROUNDS):
        state.copy_(initial)
        for _ in range(ROUND_WARMUP):
            launch()
        for _ in range(DRY_RUN_ITERS):
            graph.replay()
        device.synchronize()
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        device.synchronize()
        # Event.elapsed_time returns milliseconds; convert exactly once.
        samples = [start.elapsed_time(end) * 1000.0 / CALLS_PER_GRAPH for start, end in events]
        assert all(math.isfinite(value) and value > 0 for value in samples), samples
        assert torch.isfinite(state).all(), "nonfinite state after timing"
        assert torch.isfinite(out).all(), "nonfinite output after timing"
        assert torch.equal(state[0], initial[0]), "null state slot changed after timing"
        samples_by_round.append(samples)
    medians = [statistics.median(samples) for samples in samples_by_round]
    latency_us = statistics.median(medians)
    batch = data["mixed_qkv"].shape[0]
    recorder = BenchmarkRecorder(
        record_property, op_name="fused_recurrent_gated_delta_rule_packed_decode",
        dtype="torch.bfloat16", phase="forward", baseline=None,
        timing_method="device_event", execution_kind="multi_call_graph_replay",
        latency_scope="amortized_graph_update", calls_per_graph=CALLS_PER_GRAPH,
        cache_protocol="warm_repeated_no_flush", latency_unit="us",
        upstream_protocol_reproduced=False, source="vllm-project/vllm#61355",
        device=backend.get_device_name(), torch_version=torch.__version__, seed=SEED,
    )
    recorder.add(
        shape_detail=dict(B=batch, T=1, H=H, HV=HV, K=K, V=V,
                          state_dtype=str(state.dtype), state_shape=list(state.shape),
                          mixed_qkv_shape=list(data["mixed_qkv"].shape),
                          mixed_qkv_stride=list(data["mixed_qkv"].stride()),
                          TP=4, qk_normalization=True, scale=K**-0.5),
        latency=latency_us, samples_us=samples_by_round, round_medians_us=medians,
        rounds=ROUNDS, samples_per_round=REPEAT_ITERS,
        dry_run_iters=DRY_RUN_ITERS, initial_warmup=INITIAL_WARMUP,
        round_warmup=ROUND_WARMUP, correctness="eager_and_graph_updates_1_10_and_32",
    )
    recorder.record()
    return batch, str(state.dtype).removeprefix("torch."), latency_us


@pytest.fixture(scope="module")
def benchmark_table(request):
    """Stream rows through pytest's terminal while keeping normal diagnostics."""
    prefix = request.node.nodeid + "::"

    class QuietPassedCases:
        @pytest.hookimpl(tryfirst=True)
        def pytest_report_teststatus(self, report, config):
            if report.nodeid.startswith(prefix) and report.when == "call" and report.passed:
                return "passed", "", ""

    plugin = QuietPassedCases()
    manager = request.config.pluginmanager
    manager.register(plugin)
    reporter = manager.get_plugin("terminalreporter")

    def write(line):
        with manager.get_plugin("capturemanager").global_and_fixture_disabled():
            if reporter is not None:
                reporter.write(line + "\n", flush=True)
            else:
                print(line, flush=True)

    write("")
    write(f"{'B':>4} {'H':>3} {'HV':>3} {'K':>4} {'V':>4} {'QKV':>8} "
          f"{'State':>8} {'packed_decode(us)':>18}")
    write("-" * 68)

    def emit(row):
        batch, state_dtype, us = row
        write(f"{batch:>4} {H:>3} {HV:>3} {K:>4} {V:>4} {'bfloat16':>8} "
              f"{state_dtype:>8} {us:>18.3f}")

    try:
        yield emit
    finally:
        manager.unregister(plugin)


@pytest.mark.benchmark
@pytest.mark.parametrize("batch", BATCHES, ids=[f"B{batch}" for batch in BATCHES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=["state-fp32", "state-bf16"])
def test_fused_recurrent_packed_decode_benchmark(
    benchmark_inputs, benchmark_table, record_property, batch, state_dtype,
):
    benchmark_table(_measure(benchmark_inputs[state_dtype, batch], record_property))
