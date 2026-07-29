#include "profiler.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <vector>

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

    for (auto& name : _pre_ops)
        _op_names.push_back(name);
    for (size_t l = 0; l < _layers; ++l)
        for (auto& name : _per_layer_ops)
            _op_names.push_back(name);
    for (auto& name : _post_ops)
        _op_names.push_back(name);

#ifdef LLAISYS_ENABLE_PROFILING
    // Pre-allocate event pool (one start+end pair per op in sequence)
    for (auto& e : _start_pool) cudaEventDestroy(e);
    for (auto& e : _end_pool)   cudaEventDestroy(e);
    _start_pool.resize(_op_names.size());
    _end_pool.resize(_op_names.size());
    for (size_t i = 0; i < _op_names.size(); ++i) {
        cudaEventCreate(&_start_pool[i]);
        cudaEventCreate(&_end_pool[i]);
    }
#endif
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

cudaStream_t OpProfiler::_get_stream() {
    return cudaStreamLegacy;
}

void OpProfiler::begin_forward(bool is_prefill) {
    if (_op_names.empty())
        return;

    _step_counter = 0;

    // Prefill always sampled. Decode: _active set by previous end_forward
    if (is_prefill)
        _active = true;

    _is_prefill = is_prefill;

    // Clear scoped timers and recycle events to pool
    _scoped_timers.clear();
    _pool_idx = 0;
}

// Per-operator ScopedTimer — uses pre-allocated event pool (zero cudaEventCreate)
OpProfiler::ScopedTimer::ScopedTimer(OpProfiler& p) : prof(&p) {
    if (!p._active || p._op_names.empty() || p._pool_idx >= p._start_pool.size()) {
        prof = nullptr;
        return;
    }
    pos = p._step_counter++;
    start_ev = p._start_pool[p._pool_idx];
    end_ev   = p._end_pool[p._pool_idx];
    p._pool_idx++;

    cudaStream_t stream = p._get_stream();
    if (stream) {
        cudaEventRecord(start_ev, stream);
    }
}

OpProfiler::ScopedTimer::~ScopedTimer() {
    if (!prof || !prof->_active) return;

    cudaStream_t stream = prof->_get_stream();
    if (stream) {
        cudaEventRecord(end_ev, stream);
    }

    prof->_scoped_timers.push_back({pos, start_ev, end_ev});
}

void OpProfiler::end_forward(int decode_step) {
    bool next_active = false;
    if (!_sample_points.empty()) {
        int next_step = _is_prefill ? 1 : decode_step + 1;
        next_active = std::binary_search(
            _sample_points.begin(), _sample_points.end(), next_step);
    }

    if (!_active) {
        _active = next_active;
        return;
    }
    _active = false;

    bool should_sample = _is_prefill || _sample_points.empty() ||
        std::binary_search(_sample_points.begin(), _sample_points.end(), decode_step);

    if (!should_sample) {
        _active = next_active;
        return;
    }

    // Sync all scoped timers
    cudaStream_t stream = _get_stream();
    if (stream) {
        for (auto& t : _scoped_timers) {
            cudaEventSynchronize(t.end_ev);
        }
    }

    _snapshot(decode_step);
    _active = next_active;
}

void OpProfiler::_snapshot(int decode_step) {
    ForwardSnapshot snap;
    snap.step = decode_step;
    snap.phase = _is_prefill ? "prefill" : "decode";

    for (auto& t : _scoped_timers) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t.start_ev, t.end_ev);

        OpSnapshot op;
        op.name = (t.pos < _op_names.size()) ? _op_names[t.pos] : "?";
        op.elapsed_ms = ms;
        op.layer = _get_layer_for_op_index(t.pos);
        snap.ops.push_back(std::move(op));
    }

    _snapshots.push_back(std::move(snap));
}

void OpProfiler::dump_json(const char* path) {
    // Ensure parent directory exists
    std::string p(path);
    auto pos = p.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = p.substr(0, pos);
        mkdir(dir.c_str(), 0755);
    }

    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "OpProfiler: cannot write to " << path << std::endl;
        return;
    }

    f << "{\n  \"model\": \"" << _model << "\",\n  \"snapshots\": [\n";

    for (size_t s = 0; s < _snapshots.size(); ++s) {
        const auto& snap = _snapshots[s];

        // ── aggregate by operator name ─────────────
        std::map<std::string, std::pair<double, int>> agg;
        for (const auto& op : snap.ops) {
            auto& entry = agg[op.name];
            entry.first  += op.elapsed_ms;
            entry.second += 1;
        }

        double total_ms = 0.0;
        for (const auto& [_, pair] : agg) total_ms += pair.first;

        // ── sort by total_ms descending ────────────
        std::vector<std::tuple<std::string, double, int>> sorted;
        for (const auto& [name, pair] : agg)
            sorted.emplace_back(name, pair.first, pair.second);
        std::sort(sorted.begin(), sorted.end(),
            [](auto& a, auto& b) { return std::get<1>(a) > std::get<1>(b); });

        // ── JSON (aggregated, sorted) ──────────────
        f << "    {\n";
        f << "      \"phase\": \"" << snap.phase << "\",\n";
        f << "      \"step\": " << snap.step << ",\n";
        f << "      \"total_ms\": " << total_ms << ",\n";
        f << "      \"ops\": [\n";
        for (size_t i = 0; i < sorted.size(); ++i) {
            const auto& [name, sum, cnt] = sorted[i];
            f << "        {\"name\": \"" << name
              << "\", \"avg_ms\": " << (sum / cnt)
              << ", \"count\": " << cnt
              << ", \"total_ms\": " << sum << "}";
            if (i + 1 < sorted.size()) f << ",";
            f << "\n";
        }
        f << "      ]\n    }";
        if (s + 1 < _snapshots.size()) f << ",";
        f << "\n";
    }

    f << "  ]\n}\n";
}

#else // !LLAISYS_ENABLE_PROFILING

// No-op stubs
void OpProfiler::begin_forward(bool) {}
void OpProfiler::end_forward(int) {}
void OpProfiler::dump_json(const char*) {}

#endif // LLAISYS_ENABLE_PROFILING

} // namespace llaisys
