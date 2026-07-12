#include "profiler.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>

#ifdef LLAISYS_ENABLE_PROFILING
#include "../core/context/context.hpp"
#include "../core/runtime/runtime.hpp"
#endif

namespace llaisys {

// ============================================================
// Meyer's singleton
// ============================================================
OpProfiler& OpProfiler::instance() {
    static OpProfiler profiler;
    return profiler;
}

// ============================================================
// Model registration — always compiled, called behind #ifdef
// ============================================================
void OpProfiler::register_sequence(
    const std::string& model, size_t layers,
    const std::vector<std::string>& per_layer_ops,
    const std::vector<std::string>& pre_ops,
    const std::vector<std::string>& post_ops)
{
    _model = model;
    _layers = layers;
    _per_layer_ops = per_layer_ops;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
    _build_op_sequence();
}

void OpProfiler::set_sample_points(const std::vector<int>& points) {
    _sample_points = points;
    std::sort(_sample_points.begin(), _sample_points.end());
}

void OpProfiler::_build_op_sequence() {
    _op_names.clear();

    // Pre-ops (e.g. "embed")
    for (auto& name : _pre_ops) {
        _op_names.push_back(name);
    }

    // Per-layer ops repeated for each layer
    for (size_t l = 0; l < _layers; ++l) {
        for (auto& name : _per_layer_ops) {
            _op_names.push_back(name);
        }
    }

    // Post-ops (e.g. "embed:gather", "out_norm", "linear:lm_head", "top_k")
    for (auto& name : _post_ops) {
        _op_names.push_back(name);
    }
}

int OpProfiler::_get_layer_for_op_index(size_t idx) const {
    // Pre-ops: no layer
    if (idx < _pre_ops.size())
        return -1;

    idx -= _pre_ops.size();

    // Per-layer ops
    size_t per_layer_count = _per_layer_ops.size();
    if (per_layer_count > 0) {
        size_t total_per_layer = _layers * per_layer_count;
        if (idx < total_per_layer) {
            return static_cast<int>(idx / per_layer_count);
        }
    }

    // Post-ops: no layer
    return -1;
}

// ============================================================
// Profiling runtime — CUDA events, step counter, snapshots
// ============================================================
#ifdef LLAISYS_ENABLE_PROFILING

void OpProfiler::_ensure_events() {
    // We need one event per op boundary: N ops + 1 final event = N+1 events
    size_t needed = _op_names.size() + 1;
    if (_events.size() >= needed)
        return;

    // Destroy old events before resizing
    for (auto e : _events) {
        if (e) cudaEventDestroy(e);
    }
    _events.clear();
    _events.resize(needed, nullptr);

    for (size_t i = 0; i < needed; ++i) {
        cudaEventCreate(&_events[i]);
    }
}

cudaStream_t OpProfiler::_get_stream() {
    auto& runtime = llaisys::core::context().runtime();
    if (runtime.deviceType() != LLAISYS_DEVICE_NVIDIA)
        return nullptr;
    return static_cast<cudaStream_t>(runtime.stream());
}

void OpProfiler::begin_forward(bool is_prefill) {
    if (_op_names.empty())
        return;

    _step_counter = 0;
    _active = true;
    _is_prefill = is_prefill;

    // Ensure event pool is large enough for the sequence
    _ensure_events();

    // Record the first event (start of forward pass / first op)
    cudaStream_t stream = _get_stream();
    if (stream && !_events.empty()) {
        cudaEventRecord(_events[0], stream);
    }
}

void OpProfiler::step() {
    if (!_active || _op_names.empty())
        return;

    ++_step_counter;

    // Record an event at this step boundary (end of previous op, start of next)
    cudaStream_t stream = _get_stream();
    if (stream && _step_counter < _events.size()) {
        cudaEventRecord(_events[_step_counter], stream);
    }
}

void OpProfiler::end_forward(int decode_step) {
    if (!_active)
        return;
    _active = false;

    if (_op_names.empty())
        return;

    size_t n_ops = _op_names.size();

    // Guard: if fewer step() calls than ops, clamp to what we have
    if (_step_counter > n_ops)
        _step_counter = n_ops;

    // Record final event (end of last op)
    cudaStream_t stream = _get_stream();
    if (stream && _step_counter < _events.size()) {
        cudaEventRecord(_events[_step_counter], stream);
    }

    // Determine whether to sample this step
    bool should_sample = false;
    if (_is_prefill) {
        // Always capture prefill (step 0) for diagnostics
        should_sample = true;
    } else if (_sample_points.empty()) {
        // No filter configured → capture every decode step
        should_sample = true;
    } else {
        should_sample = std::binary_search(
            _sample_points.begin(), _sample_points.end(), decode_step);
    }

    if (!should_sample)
        return;

    // Synchronize on the last recorded event to ensure all GPU work is done
    if (stream && _step_counter < _events.size()) {
        cudaEventSynchronize(_events[_step_counter]);
    }

    _snapshot(decode_step);
}

void OpProfiler::_snapshot(int decode_step) {
    ForwardSnapshot snap;
    snap.step = decode_step;
    snap.phase = _is_prefill ? "prefill" : "decode";

    size_t n_ops = _op_names.size();

    for (size_t i = 0; i < n_ops && i + 1 < _events.size(); ++i) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, _events[i], _events[i + 1]);

        OpSnapshot op;
        op.name = _op_names[i];
        op.elapsed_ms = ms;
        op.layer = _get_layer_for_op_index(i);
        snap.ops.push_back(std::move(op));
    }

    _snapshots.push_back(std::move(snap));
}

void OpProfiler::dump_json(const char* path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "OpProfiler: cannot write to " << path << std::endl;
        return;
    }

    f << "{\n";
    f << "  \"model\": \"" << _model << "\",\n";
    f << "  \"snapshots\": [\n";

    for (size_t s = 0; s < _snapshots.size(); ++s) {
        const auto& snap = _snapshots[s];

        f << "    {\n";
        f << "      \"step\": " << snap.step << ",\n";
        f << "      \"phase\": \"" << snap.phase << "\",\n";
        f << "      \"ops\": [\n";

        for (size_t o = 0; o < snap.ops.size(); ++o) {
            const auto& op = snap.ops[o];

            f << "        {\"name\": \"" << op.name << "\", "
              << "\"elapsed_ms\": " << op.elapsed_ms;

            if (op.layer >= 0) {
                f << ", \"layer\": " << op.layer;
            }

            f << "}";
            if (o + 1 < snap.ops.size())
                f << ",";
            f << "\n";
        }

        f << "      ]\n";
        f << "    }";
        if (s + 1 < _snapshots.size())
            f << ",";
        f << "\n";
    }

    f << "  ]\n";
    f << "}\n";

    std::cout << "OpProfiler: "
              << _snapshots.size() << " snapshots written to "
              << path << std::endl;
}

#else // !LLAISYS_ENABLE_PROFILING

// No-op stubs — compiled when profiling is disabled.
// The PROFILE_* macros expand to ((void)0) so these are never called,
// but they must exist so the class links cleanly.
void OpProfiler::begin_forward(bool) {}
void OpProfiler::end_forward(int) {}
void OpProfiler::step() {}
void OpProfiler::dump_json(const char*) {}

#endif // LLAISYS_ENABLE_PROFILING

} // namespace llaisys
