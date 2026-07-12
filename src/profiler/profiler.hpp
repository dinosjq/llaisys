#pragma once
#include <string>
#include <vector>
#include <cstddef>
#ifdef LLAISYS_ENABLE_PROFILING
#include <cuda_runtime.h>
#endif

namespace llaisys {

class OpProfiler {
public:
    static OpProfiler& instance();

    // Model registration (called once in constructor)
    void register_sequence(const std::string& model, size_t layers,
        const std::vector<std::string>& per_layer_ops,
        const std::vector<std::string>& pre_ops,
        const std::vector<std::string>& post_ops);
    void set_sample_points(const std::vector<int>& points);

    // Event loop hooks
    void begin_forward(bool is_prefill);
    void end_forward(int decode_step);
    bool active() const { return _active; }

    // RAII per-operator timer (replaces per-boundary events)
    struct ScopedTimer {
        OpProfiler* prof;
        size_t pos;
        cudaEvent_t start_ev, end_ev;
        ScopedTimer(OpProfiler& p);
        ~ScopedTimer();
    };

    // Called from Qwen2::stop()
    void dump_json(const char* path);

private:
    OpProfiler() = default;

    void _build_op_sequence();
    int _get_layer_for_op_index(size_t idx) const;

    // Always-stored registration state (lightweight, no CUDA dependency)
    std::string _model;
    size_t _layers = 0;
    std::vector<std::string> _pre_ops;
    std::vector<std::string> _per_layer_ops;
    std::vector<std::string> _post_ops;
    std::vector<std::string> _op_names;
    std::vector<int> _sample_points;

#ifdef LLAISYS_ENABLE_PROFILING
    struct TimerEntry {
        size_t pos;
        cudaEvent_t start_ev, end_ev;
    };
    std::vector<TimerEntry> _scoped_timers;

    size_t _step_counter = 0;
    bool _active = false;
    bool _is_prefill = false;

    struct OpSnapshot {
        std::string name;
        int layer;
        float elapsed_ms;
    };
    struct ForwardSnapshot {
        int step;
        std::string phase;
        std::vector<OpSnapshot> ops;
    };
    std::vector<ForwardSnapshot> _snapshots;

    void _snapshot(int decode_step);
    cudaStream_t _get_stream();
#endif
};

} // namespace llaisys

#ifdef LLAISYS_ENABLE_PROFILING
  #define PROFILE_STEP()   llaisys::OpProfiler::ScopedTimer _pftimer(llaisys::OpProfiler::instance())
  #define PROFILE_BEGIN(p) llaisys::OpProfiler::instance().begin_forward(p)
  #define PROFILE_END(d)   llaisys::OpProfiler::instance().end_forward(d)
#else
  #define PROFILE_STEP()   ((void)0)
  #define PROFILE_BEGIN(p) ((void)0)
  #define PROFILE_END(d)   ((void)0)
#endif
