# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""T-Head fused recurrent gated delta rule forward benchmark."""

import math
import statistics

import pytest
import torch

from flag_attn import runtime
from flag_attn.testing import backend as test_backend

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder

ASSERT_RATIO = 0.01
BATCHES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)
# (TP label, per-device query/key heads, per-device value heads)
HEAD_CONFIGS = ((2, 8, 16), (4, 4, 8))
HEAD_DIM = 128
DTYPE = torch.bfloat16
STATE_DTYPE = torch.bfloat16
ROUNDS, REPEAT_ITERS, DRY_RUN_ITERS = 7, 40, 5
INITIAL_WARMUP, ROUND_WARMUP = 10, 30
CALLS_PER_GRAPH = 32

def _load_fwd():
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )
    return fused_recurrent_gated_delta_rule_fwd


def _reference(inputs):
    """CPU FP32 recurrence; state ABI is [slots, HV, V, K]."""
    q, k, v, g, beta = [inputs[name].detach().float().cpu() for name in ("q", "k", "v", "g", "beta")]
    state0 = inputs["initial_state"].detach().float().cpu()
    indices = inputs["ssm_state_indices"].detach().cpu()
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    cu = inputs["cu_seqlens"]
    spans = list(zip(cu.cpu().tolist()[:-1], cu.cpu().tolist()[1:])) if cu is not None else [
        (b * T, (b + 1) * T) for b in range(B)
    ]
    q, k = q.reshape(-1, H, K), k.reshape(-1, H, K)
    v, g = v.reshape(-1, HV, V), g.reshape(-1, HV)
    beta = beta.reshape(-1, HV, V) if beta.ndim == 4 else beta.reshape(-1, HV, 1)
    out = torch.zeros_like(v)
    final = state0.clone() if inputs["inplace_final_state"] else torch.empty(B * T, HV, V, K)
    valid = torch.zeros(B * T, dtype=torch.bool)
    written = set()
    accepted = inputs.get("num_accepted_tokens")
    accepted = None if accepted is None else accepted.detach().cpu()
    for n, (start, end) in enumerate(spans):
        if start == end:
            continue
        row = indices[n].reshape(-1)
        read_pos = int(accepted[n]) - 1 if accepted is not None else 0
        slot = int(row[read_pos])
        if slot <= 0:
            continue
        state = state0[slot].clone()
        for t, pos in enumerate(range(start, end)):
            qt = q[pos].repeat_interleave(HV // H, dim=0)
            kt = k[pos].repeat_interleave(HV // H, dim=0)
            if inputs["use_qk_l2norm_in_kernel"]:
                qt = qt / torch.sqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            state = state * torch.exp(g[pos])[:, None, None]
            residual = (v[pos] - (state * kt[:, None, :]).sum(-1)) * beta[pos]
            state = state + residual[:, :, None] * kt[:, None, :]
            out[pos] = (state * (qt * inputs["scale"])[:, None, :]).sum(-1)
            valid[pos] = True
            dest = int(row[t]) if inputs["inplace_final_state"] else pos
            if not inputs["inplace_final_state"] or dest > 0:
                final[dest] = state
                written.add(dest)
    return out.reshape(B, T, HV, V), final, valid, sorted(written)


def _assert_close(name, actual, expected):
    # Select tolerances before converting: output and state may differ in dtype.
    atol, rtol = (1e-4, 1e-4) if actual.dtype == torch.float32 else (2e-2, 1e-2)
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert torch.isfinite(actual).all(), f"{name}: nonfinite actual"
    assert torch.isfinite(expected).all(), f"{name}: nonfinite reference"
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{name}: elementwise check failed\n{message}",
    )
    max_abs = (actual - expected).abs().max().item()
    ratio = (actual - expected).square().mean().sqrt().item() / (
        expected.square().mean().sqrt().item() + 1e-8
    )
    assert max_abs <= 1e-6 or ratio < ASSERT_RATIO, (
        f"{name}: relative_rms={ratio:.7g}, max_abs={max_abs:.7g}, limit={ASSERT_RATIO}"
    )
    return ratio


def _check_case_impl(inputs):
    fwd = _load_fwd()
    expected_o, expected_state, valid, written = _reference(inputs)
    original = inputs["initial_state"].clone()
    actual_o, actual_state = fwd(**inputs)
    test_backend.device_fn.synchronize()
    actual_flat = actual_o.reshape(-1, *actual_o.shape[-2:])
    expected_flat = expected_o.reshape(actual_flat.shape)
    output_ratio = 0.0
    for pos in valid.nonzero().flatten().tolist():
        output_ratio = max(output_ratio, _assert_close(f"output[{pos}]", actual_flat[pos], expected_flat[pos]))
    state_ratio = 0.0
    for slot in written:
        state_ratio = max(state_ratio, _assert_close(f"state[{slot}]", actual_state[slot], expected_state[slot]))
    if inputs["inplace_final_state"]:
        assert actual_state.data_ptr() == inputs["initial_state"].data_ptr()
        untouched = [slot for slot in range(len(original)) if slot not in written]
        assert torch.equal(actual_state[untouched], original[untouched]), "untouched state slots changed"
    else:
        assert torch.equal(inputs["initial_state"], original), "non-inplace call modified initial state"
        assert actual_state.data_ptr() != inputs["initial_state"].data_ptr()
    return output_ratio, state_ratio



def _check_case(inputs):
    threads = torch.get_num_threads()
    try:
        # Bound CPU reference and assertion overhead; restore before timing.
        torch.set_num_threads(1)
        return _check_case_impl(inputs)
    finally:
        torch.set_num_threads(threads)


def _make_inputs(batch, H, HV):
    torch.manual_seed(0)
    device = test_backend.device
    K = V = HEAD_DIM
    q = torch.randn(batch, 1, H, K, device=device, dtype=DTYPE)
    k = torch.randn(batch, 1, H, K, device=device, dtype=DTYPE)
    v = torch.randn(batch, 1, HV, V, device=device, dtype=DTYPE)
    g = -torch.rand(batch, 1, HV, device=device, dtype=torch.float32)
    beta = torch.rand(batch, 1, HV, device=device, dtype=DTYPE)
    indices = torch.randperm(batch, device=device, dtype=torch.int32) + 1
    state = torch.randn(batch + 1, HV, V, K, device=device, dtype=STATE_DTYPE)
    return dict(q=q, k=k, v=v, g=g, beta=beta, scale=K**-0.5,
                initial_state=state, inplace_final_state=True, cu_seqlens=None,
                ssm_state_indices=indices, num_accepted_tokens=None,
                use_qk_l2norm_in_kernel=True)


def _shape_matrix():
    return [(tp, batch, H, HV) for tp, H, HV in HEAD_CONFIGS for batch in BATCHES]


def _graph_samples(fwd, inputs, initial):
    """Amortized warm-cache device time, including Graph device overhead."""
    device = test_backend.device_fn
    if not test_backend.graph_available() or not hasattr(device, "Event"):
        pytest.fail("this benchmark requires T-Head Graph and Event support")
    state = inputs["initial_state"]
    for _ in range(INITIAL_WARMUP):
        fwd(**inputs)
    stream = device.Stream()
    stream.wait_stream(device.current_stream())
    graph = device.CUDAGraph()
    with device.graph(graph, stream=stream):
        for _ in range(CALLS_PER_GRAPH):
            captured_out, _ = fwd(**inputs)
    device.current_stream().wait_stream(stream)
    device.synchronize()

    # A replay must perform exactly the same updates as 32 eager calls.
    state.copy_(initial)
    graph.replay()
    graph_out, graph_state = captured_out.clone(), state.clone()
    state.copy_(initial)
    for _ in range(CALLS_PER_GRAPH):
        eager_out, _ = fwd(**inputs)
    device.synchronize()
    assert torch.equal(graph_out, eager_out), "Graph/eager output mismatch"
    assert torch.equal(graph_state, state), "Graph/eager 32-update state mismatch"
    assert torch.equal(state[0], initial[0]), "null state slot changed"
    del graph_out, graph_state, eager_out

    events = [(device.Event(enable_timing=True), device.Event(enable_timing=True))
              for _ in range(REPEAT_ITERS)]
    for start, end in events:
        start.record()
        end.record()
    device.synchronize()
    rounds = []
    for _ in range(ROUNDS):
        state.copy_(initial)
        for _ in range(ROUND_WARMUP):
            fwd(**inputs)
        for _ in range(DRY_RUN_ITERS):
            graph.replay()
        device.synchronize()
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        device.synchronize()
        samples = [start.elapsed_time(end) * 1000.0 / CALLS_PER_GRAPH
                   for start, end in events]
        assert all(math.isfinite(value) and value > 0 for value in samples)
        assert torch.isfinite(state).all(), "nonfinite state after timing"
        assert torch.isfinite(captured_out).all(), "nonfinite output after timing"
        assert torch.equal(state[0], initial[0]), "null state slot changed"
        rounds.append(samples)
    return rounds


@torch.inference_mode()
def run_benchmark(record_property, shape):
    shapes = [shape]
    fwd = _load_fwd()
    recorder = BenchmarkRecorder(
        record_property, op_name="fused_recurrent_gated_delta_rule_fwd",
        dtype=str(DTYPE), phase="forward", baseline=None,
    )
    for tp, batch, H, HV in shapes:
        inputs = _make_inputs(batch, H, HV)
        pristine = inputs["initial_state"].clone()
        errors = _check_case(inputs)
        samples_by_round = _graph_samples(fwd, inputs, pristine)
        del pristine
        samples = [value for row in samples_by_round for value in row]
        medians = [statistics.median(row) for row in samples_by_round]
        median = statistics.median(medians)
        p20, p80 = torch.quantile(torch.tensor(samples), torch.tensor([0.2, 0.8])).tolist()
        detail = dict(TP=tp, B=batch, T=1, H=H, HV=HV, K=HEAD_DIM, V=HEAD_DIM,
                      q_shape=list(inputs["q"].shape), total_tokens=batch,
                      state_shape=list(inputs["initial_state"].shape),
                      state_dtype=str(STATE_DTYPE), gate_dtype=str(inputs["g"].dtype),
                      beta_dtype=str(inputs["beta"].dtype), inplace=True,
                      qk_normalization=True, spec_decoding=False)
        recorder.add(
            shape_detail=detail, latency=median, latency_base=None,
            p20_us=p20, p80_us=p80, latency_unit="us", sample_count=len(samples),
            output_relative_rms=errors[0], state_relative_rms=errors[1],
            timing_protocol="warm_repeated_state_updates",
            timing_method="device_event", execution_kind="multi_call_graph_replay",
            cache_protocol="warm_repeated_no_flush", latency_scope="amortized_graph_update",
            calls_per_graph=CALLS_PER_GRAPH, rounds=ROUNDS, samples_per_round=REPEAT_ITERS,
            samples_us=samples_by_round, round_medians_us=medians,
            initial_warmup=INITIAL_WARMUP, round_warmup=ROUND_WARMUP,
            dry_run_iters=DRY_RUN_ITERS, correctness="cpu_fp32_and_graph_vs_eager_32_updates",
        )
        del inputs
    recorder.record()
    return (*shape, median)


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
    write(f"{'TP':>3} {'B':>4} {'T':>3} {'H':>4} {'HV':>4} {'K':>4} {'V':>4} "
          f"{'dtype':>8} {'flag_attn(us)':>14}")
    write("-" * 65)

    def emit(row):
        tp, batch, H, HV, median = row
        write(f"{tp:>3} {batch:>4} {1:>3} {H:>4} {HV:>4} "
              f"{HEAD_DIM:>4} {HEAD_DIM:>4} {str(DTYPE).removeprefix('torch.'):>8} "
              f"{median:>14.3f}")

    try:
        yield emit
    finally:
        manager.unregister(plugin)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "shape", _shape_matrix(),
    ids=[f"TP{tp}-B{batch}-T1-H{H}-HV{HV}-K128-V128-bf16" for tp, batch, H, HV in _shape_matrix()],
)
def test_fused_recurrent_gated_delta_rule_benchmark(record_property, shape, benchmark_table):
    benchmark_table(run_benchmark(record_property, shape))
