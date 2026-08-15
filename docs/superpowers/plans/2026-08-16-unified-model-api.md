# Unified Model C API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Qwen2-named C/Python ABI with a unified Model API, assemble layers inside `Qwen2`/`Llama::forward`, and add `Llama : public Qwen2`.

**Architecture:** Opaque `LlaisysModel` holds `shared_ptr<framework::Model>`. Create dispatches on `LlaisysModelArch`. Weights enter only via `SetWeight` → `Model::set_weight`. `Qwen2::forward` / `Llama::forward` explicitly call Layer objects; `run_*_layer_stack` is deleted.

**Tech Stack:** C++17, ctypes Python, xmake, existing LLAISYS ops/runtime.

**Spec:** `docs/superpowers/specs/2026-08-16-unified-model-api-design.md`

## Global Constraints

- Do not change Scheduler / Sequence / BlockManager.
- Do not implement Llama-3 `rope_scaling`.
- No `llaisysQwen2*` compatibility shims; delete old ABI in the same change set as consumers migrate.
- No `LLAISYS_LAYER_STACK` env switch.
- Layer assembly lives in Model `forward`, not in `layers/` helpers.
- Member style: `_` prefix, `this->`, Chinese hot-path comments (match `qwen2.cpp`).
- Same-round remove internal `_weights` / Scheme A sync (upload is SetWeight only).

## File Structure

| Path | Role |
|---|---|
| `include/llaisys/models/model.h` | New C ABI (Arch, Meta, WeightRole, Model/Request APIs) |
| `src/llaisys/models/model.cc` | C bridge; Create → Qwen2 or Llama |
| `src/models/qwen2/qwen2.{hpp,cpp}` | Drop Weights; inline Qwen2 layer assembly; protected helpers |
| `src/models/llama/llama.{hpp,cpp}` | `Llama : Qwen2`, override `forward` |
| `src/layers/qwen2/*`, `src/layers/llama/*` | Layer classes only; delete `run_*_layer_stack` |
| `python/llaisys/libllaisys/models/model.py` | ctypes for new ABI |
| `python/llaisys/models/{qwen2,llama}.py` | Create + SetWeight loaders |
| `python/llaisys/models/qwen2_weight_map.py` | Keep HF→role; drop legacy slot assign |
| Delete | `include/llaisys/models/qwen2.h`, `src/llaisys/models/qwen2.cc`, `libllaisys/models/qwen2.py` |

---

### Task 1: C header + bridge skeleton

**Files:**
- Create: `include/llaisys/models/model.h`
- Create: `src/llaisys/models/model.cc`
- Modify: `xmake.lua` (already `src/llaisys/models/*.cc` — ensure new file builds; remove dependency on old name later)
- Test: `test/test_model_api_symbols.py` (new)

**Interfaces:**
- Produces: `llaisysModelCreate/Destroy/SetWeight/Request*` symbols; `LlaisysModelMeta`, `LlaisysModelArch`, `LlaisysWeightRole`

- [ ] **Step 1: Add failing symbol test**

```python
# test/test_model_api_symbols.py
import unittest
from llaisys.libllaisys import LIB_LLAISYS

class TestModelApiSymbols(unittest.TestCase):
    def test_unified_symbols_exist(self):
        for name in (
            "llaisysModelCreate",
            "llaisysModelDestroy",
            "llaisysModelSetWeight",
            "llaisysModelRequestSubmit",
            "llaisysModelRequestAwait",
            "llaisysModelRequestAbort",
            "llaisysModelRequestRelease",
        ):
            self.assertTrue(hasattr(LIB_LLAISYS, name), name)

    def test_old_qwen2_symbols_gone(self):
        for name in (
            "llaisysQwen2ModelCreate",
            "llaisysQwen2ModelWeights",
            "llaisysQwen2RequestSubmit",
        ):
            self.assertFalse(hasattr(LIB_LLAISYS, name), name)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Write `include/llaisys/models/model.h`**

```c
#ifndef LLAISYS_MODELS_MODEL_H
#define LLAISYS_MODELS_MODEL_H

#include "../tensor.h"

__C {
    typedef enum LlaisysModelArch {
        LLAISYS_MODEL_QWEN2 = 0,
        LLAISYS_MODEL_LLAMA = 1,
    } LlaisysModelArch;

    /* Must match llaisys::framework::WeightRole order/values. */
    typedef enum LlaisysWeightRole {
        LLAISYS_WEIGHT_IN_EMBED = 0,
        LLAISYS_WEIGHT_OUT_EMBED = 1,
        LLAISYS_WEIGHT_OUT_NORM = 2,
        LLAISYS_WEIGHT_ATTN_NORM = 3,
        LLAISYS_WEIGHT_ATTN_Q_W = 4,
        LLAISYS_WEIGHT_ATTN_Q_B = 5,
        LLAISYS_WEIGHT_ATTN_K_W = 6,
        LLAISYS_WEIGHT_ATTN_K_B = 7,
        LLAISYS_WEIGHT_ATTN_V_W = 8,
        LLAISYS_WEIGHT_ATTN_V_B = 9,
        LLAISYS_WEIGHT_ATTN_O_W = 10,
        LLAISYS_WEIGHT_MLP_NORM = 11,
        LLAISYS_WEIGHT_MLP_GATE_W = 12,
        LLAISYS_WEIGHT_MLP_UP_W = 13,
        LLAISYS_WEIGHT_MLP_DOWN_W = 14,
    } LlaisysWeightRole;

    struct LlaisysModelMeta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysModel;
    struct LlaisysRequest;

    __export struct LlaisysModel *llaisysModelCreate(LlaisysModelArch arch, LlaisysModelMeta *meta,
                                                     llaisysDeviceType_t device, int *device_ids, int ndevice);
    __export void llaisysModelDestroy(struct LlaisysModel *model);
    __export void llaisysModelSetWeight(struct LlaisysModel *model, LlaisysWeightRole role, size_t layer,
                                        llaisysTensor_t tensor);

    __export struct LlaisysRequest *llaisysModelRequestSubmit(struct LlaisysModel *model, int64_t *token_ids,
                                                             size_t ntoken, int64_t max_new_tokens, int top_k,
                                                             float top_p, float temperature);
    __export int64_t llaisysModelRequestAwait(struct LlaisysRequest *request);
    __export void llaisysModelRequestAbort(struct LlaisysRequest *request);
    __export void llaisysModelRequestRelease(struct LlaisysRequest *request);
}
#endif
```

- [ ] **Step 3: Implement `src/llaisys/models/model.cc`**

Hold `std::shared_ptr<llaisys::framework::Model>`. For this task, Create may still only construct `Qwen2` for both arches if Llama class is not ready — **prefer**: Create `QWEN2` → Qwen2; `LLAMA` → nullptr until Task 4, **or** land Task 4 stub in same commit. Recommended: Task 1 Create only handles `QWEN2`; `LLAMA` returns nullptr; Task 4 fills Llama.

```cpp
struct LlaisysModel {
    std::shared_ptr<llaisys::framework::Model> model;
};
struct LlaisysRequest {
    std::shared_ptr<llaisys::framework::Model> model;
    llaisys::framework::model_request_t request;
};

// SetWeight:
auto *t = /* llaisysTensor_t → tensor_t */;
llaisys::framework::Model::set_weight(
    model->model->ctx(),
    static_cast<llaisys::framework::WeightRole>(role),
    layer,
    t->tensor);
```

Map Meta → `llaisys::Qwen2Meta` (byte-compatible field layout) for Qwen2 ctor.

Keep old `qwen2.cc` temporarily so the tree still builds until Task 5 deletes it — **or** delete in Task 5 only after Python migrates. Task 1 may add new symbols alongside old ones; Task 5 removes old.

- [ ] **Step 4: Minimal ctypes in `python/llaisys/libllaisys/models/model.py` + export from `__init__.py`**

Wire `load_model(lib)` like existing `load_qwen2`. Do not delete old bindings yet.

- [ ] **Step 5: Build**

```bash
xmake && xmake install
pip install -e ./python/
python test/test_model_api_symbols.py
```

Expected: unified symbols PASS; old symbols still present (test’s “gone” assertions should be skipped or commented until Task 5). **Adjust Step 1 test:** split into two tests; enable “gone” only after Task 5. For Task 1, only assert new symbols exist.

- [ ] **Step 6: Commit**

```bash
git add include/llaisys/models/model.h src/llaisys/models/model.cc \
  python/llaisys/libllaisys/models/model.py python/llaisys/libllaisys/__init__.py \
  python/llaisys/libllaisys/models/__init__.py test/test_model_api_symbols.py
git commit -m "$(cat <<'EOF'
feat: add unified LlaisysModel C API skeleton

EOF
)"
```

---

### Task 2: Drop Qwen2 `_weights`; SetWeight-only upload; bind KV in sync

**Files:**
- Modify: `src/models/qwen2/qwen2.hpp`, `src/models/qwen2/qwen2.cpp`
- Modify: `src/models/core/model.hpp` (optional instance `set_weight` helper)
- Modify: C++ tests that call `model.weights()` / fill legacy arrays (`test/models/qwen2_dual_path_nvidia_test.cpp`, etc.)

**Interfaces:**
- Consumes: `Model::set_weight(ctx, role, layer, tensor)`
- Produces: `Qwen2` without `Qwen2Weights`; `sync_weights()` only binds `_k_cache`/`_v_cache` into `_ctx`; public `weights()` removed

- [ ] **Step 1: Redefine `Qwen2Meta` in `qwen2.hpp` without including `qwen2.h`**

```cpp
struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};
// Remove: using Qwen2Weights = ::LlaisysQwen2Weights;
// Remove: Qwen2Weights _weights; weights();
```

Include `../core/model.hpp` / tensor headers only.

- [ ] **Step 2: Delete `init_weight_arrays` / `free_weight_arrays` / weight fields from ctor/dtor**

Constructor still sizes `_ctx.attns` / `_ctx.ffns` / caches as today.

- [ ] **Step 3: Replace `sync_weights` body**

```cpp
void Qwen2::sync_weights() {
    const size_t nlayer = this->_meta.nlayer;
    for (size_t layer = 0; layer < nlayer; ++layer) {
        this->_ctx.k_caches[layer] = this->_k_cache[layer];
        this->_ctx.v_caches[layer] = this->_v_cache[layer];
    }
}
```

Remove forward’s reliance on copying from `_weights` (already gone). C `SetWeight` fills ctx before `start()`.

- [ ] **Step 4: Update dual-path / nvidia tests** to upload via `Model::set_weight(model.ctx(), role, layer, tensor)` instead of filling `weights()`.

- [ ] **Step 5: Build shared lib + affected xmake tests**

```bash
xmake
```

Expected: compile success (Python still on old API until Task 5 — old `qwen2.cc` `ModelWeights` will break). **Order note:** Task 2 and Task 1’s bridge must stop calling `weights()`. If old `qwen2.cc` remains, either delete it in this task or make `ModelWeights` return nullptr and break Python until Task 5 — **do Task 5 immediately after Task 2–4 in one session**, or migrate Python in the same commit as removing `_weights`.

**Practical bundling:** Implement Tasks 2–5 as one continuous branch with commits at logical checkpoints; do not leave `main` unable to load weights mid-way.

- [ ] **Step 6: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor: remove Qwen2 legacy Weights; KV bind only in sync_weights

EOF
)"
```

---

### Task 3: Inline Qwen2 layer assembly; delete `run_*_layer_stack`

**Files:**
- Modify: `src/models/qwen2/qwen2.cpp` (`forward`)
- Modify: `src/layers/qwen2/qwen2.{hpp,cpp}` — remove `run_qwen2_layer_stack`
- Modify: `src/layers/llama/llama.{hpp,cpp}` — remove `run_llama_layer_stack`
- Modify: `test/models/qwen2_layer_*_test.cpp` — stop calling stack; call Layers or model forward
- Remove: env / `_use_llama_stack` from Qwen2

**Interfaces:**
- Produces: assembly only inside `Qwen2::forward` (and later `Llama::forward`)

- [ ] **Step 1: Move stack body into `Qwen2::forward`**

After `sync_weights()` + `bind_context_runtime(...)`:

```cpp
auto &ctx = this->_ctx;
ctx.runtime.is_prefill = is_prefill;
framework::Qwen2Embedding{}.forward(ctx);
for (size_t layer = 0; layer < ctx.meta.nlayer; ++layer) {
    framework::Qwen2Attention(&ctx.attns[layer], layer).forward(ctx);
    if (layer == 0 && ctx.enable_layer0_capture) {
        ctx.workspace.capture_post_attn0 = /* stable_capture local or shared util */;
    }
    framework::Qwen2FFN(&ctx.ffns[layer]).forward(ctx);
    if (layer == 0 && ctx.enable_layer0_capture) {
        ctx.workspace.capture_post_ffn0 = /* ... */;
    }
}
// last-token → rms_norm → linear logits (same as former stack tail)
return this->sample_from_logits(ctx.workspace.logits, pack);
```

Put `stable_capture` as a file-local helper in `qwen2.cpp` (or anonymous namespace).

- [ ] **Step 2: Delete `run_qwen2_layer_stack` / `run_llama_layer_stack` from headers and cpp**

- [ ] **Step 3: Fix layer tests** that called `run_qwen2_layer_stack` — duplicate the same assembly loop inline in the test, or call individual layers as they already do for FFN-only cases.

- [ ] **Step 4: Remove `_use_llama_stack`, `env_use_llama_stack`, llama include from `qwen2.cpp`**

- [ ] **Step 5: Build**

```bash
xmake
```

- [ ] **Step 6: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor: assemble Qwen2 layers in Model::forward; drop run_*_stack

EOF
)"
```

---

### Task 4: Add `src/models/llama` and Create(LLAMA)

**Files:**
- Create: `src/models/llama/llama.hpp`, `src/models/llama/llama.cpp`
- Modify: `src/llaisys/models/model.cc` — `LLAMA` → `make_shared<Llama>(...)`
- Modify: `xmake.lua` — `add_files("src/models/llama/*.cpp")`
- Make `bind_context_runtime` / `sample_from_logits` `protected` on Qwen2 so Llama can reuse them

**Interfaces:**
- Produces:

```cpp
namespace llaisys {
class Llama : public Qwen2 {
public:
    using Qwen2::Qwen2;
    ~Llama() override = default;
    std::vector<int64_t> forward(framework::BatchPack &pack,
                                 std::vector<int64_t> &block_ids,
                                 bool is_prefill) override;
};
}
```

- [ ] **Step 1: Expose helpers on Qwen2**

In `qwen2.hpp`, move to `protected:`:

```cpp
void bind_context_runtime(framework::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill);
std::vector<int64_t> sample_from_logits(tensor_t logits, framework::BatchPack &pack);
void sync_weights() override; // already virtual via Model
```

Keep `forward` as override of Model (already).

- [ ] **Step 2: Implement `Llama::forward`**

Same as Qwen2::forward but:

```cpp
framework::LlamaEmbedding{}.forward(ctx);
framework::LlamaAttention(&ctx.attns[layer], layer).forward(ctx);
framework::LlamaFFN(&ctx.ffns[layer]).forward(ctx);
// no layer0 capture required unless tests need it
```

Call `this->sync_weights()`, `this->bind_context_runtime(...)`, `this->sample_from_logits(...)`.

- [ ] **Step 3: Wire Create**

```cpp
if (arch == LLAISYS_MODEL_QWEN2) {
    model->model = std::make_shared<llaisys::Qwen2>(meta, device, device_ids, ndevice);
} else if (arch == LLAISYS_MODEL_LLAMA) {
    model->model = std::make_shared<llaisys::Llama>(meta, device, device_ids, ndevice);
} else {
    delete model;
    return nullptr;
}
```

`Qwen2Meta` and `LlaisysModelMeta` must be layout-compatible; use `reinterpret_cast` or field-wise copy.

- [ ] **Step 4: Build**

```bash
xmake && xmake install
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: add Llama model subclass and Create(LLAMA) path

EOF
)"
```

---

### Task 5: Python migration + delete old Qwen2 C ABI

**Files:**
- Create/finish: `python/llaisys/libllaisys/models/model.py`
- Modify: `python/llaisys/models/qwen2.py`, `llama.py`, `qwen2_weight_map.py`, `llama_weight_map.py`
- Modify: `python/llaisys/libllaisys/__init__.py`, `models/__init__.py`
- Delete: `include/llaisys/models/qwen2.h`, `src/llaisys/models/qwen2.cc`, `python/llaisys/libllaisys/models/qwen2.py`
- Modify: `test/test_qwen2_request_api.py`, `test_qwen2_request_lifetime.py`, `test_qwen2_invalid_submit.py`, `test_qwen2_weight_map.py`, `test_model_api_symbols.py` (enable old-gone asserts)
- Modify: `CLAUDE.md` model API section (brief)

**Interfaces:**
- Python load path:

```python
self._model = LIB_LLAISYS.llaisysModelCreate(
    LlaisysModelArch.QWEN2,  # or LLAMA
    byref(meta),
    int(device),
    device_ids,
    1,
)
# ...
LIB_LLAISYS.llaisysModelSetWeight(self._model, int(role), c_size_t(layer), t.lib_tensor())
```

- [ ] **Step 1: Rewrite weight map assign**

Remove `assign_to_legacy_weights` / `_ROLE_TO_LEGACY_SLOT`. Keep `resolve_hf_weight` + `WeightRole` IntEnum (values must match C enum).

Optional helper:

```python
def set_model_weight(lib, model, role: WeightRole, layer: int, lib_tensor) -> None:
    lib.llaisysModelSetWeight(model, int(role), c_size_t(layer), lib_tensor)
```

- [ ] **Step 2: Migrate `Qwen2` / `Llama` Python classes**

- Meta type → `LlaisysModelMeta`
- Create with arch
- Drop `_weights` struct; keep `self._tensors` list for lifetime
- Tie embedding: `SetWeight(OutEmbed, 0, in_embed_tensor)` if lm_head missing (track `in_embed` Tensor ref on self)
- Request APIs → `llaisysModelRequest*`
- Llama: remove `os.environ["LLAISYS_LAYER_STACK"]`

- [ ] **Step 3: Delete old C header/cc and ctypes qwen2 module; fix imports**

- [ ] **Step 4: Update Python unit tests** for new symbol names

- [ ] **Step 5: Build + run symbol test + weight map test**

```bash
xmake && xmake install
pip install -e ./python/
python test/test_model_api_symbols.py
python test/test_qwen2_weight_map.py
python test/test_qwen2_request_api.py
```

Expected: all PASS; old symbols absent.

- [ ] **Step 6: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: migrate Python to unified Model API; remove Qwen2 C ABI

EOF
)"
```

---

### Task 6: Smoke verification

**Files:** none (run only); fix any stragglers found

- [ ] **Step 1: Qwen2 correctness smoke** (use project model path from `docs/常用命令.md` if present)

```bash
python test/test_infer.py --model <qwen2-path> --test --device cpu
# if GPU available:
python test/test_infer.py --model <qwen2-path> --test --device nvidia
```

Expected: existing pass criteria.

- [ ] **Step 2: Llama smoke**

```bash
python test/test_llama_smoke.py --model /home/songjq/models/Llama-3.2-3B-Instruct --device cpu
```

Expected: completes without Create/SetWeight errors; generates tokens (quality not gated).

- [ ] **Step 3: Layer CPU test if still in xmake**

```bash
xmake run qwen2_layer_cpu_test   # exact target name from xmake.lua
```

- [ ] **Step 4: Commit any test/doc fixes**

```bash
git commit -m "$(cat <<'EOF'
test: align remaining callers with unified Model API

EOF
)"
```

- [ ] **Step 5: Mark spec status**

Update `docs/superpowers/specs/2026-08-16-unified-model-api-design.md` status line to `已实施` (or leave for user).

---

## Spec coverage checklist

| Spec item | Task |
|---|---|
| Unified Model C API | 1, 5 |
| SetWeight upload | 1, 2, 5 |
| Delete Qwen2 Weights ABI | 2, 5 |
| Llama : Qwen2 + override forward | 4 |
| Assembly in Model; delete run stack | 3 |
| Remove LLAISYS_LAYER_STACK | 3, 5 |
| Python Create(arch) + SetWeight | 5 |
| No Scheduler/Sequence/BlockManager changes | all |
| Smoke Qwen2 + Llama | 6 |

## Open decisions locked by this plan

- Handle type: `shared_ptr<framework::Model>`
- `_weights` removed same effort (Tasks 2+5), not left dual-path
- layer0 capture stays in `Qwen2::forward` only
