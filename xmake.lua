add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("flash-v6-experiment")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile the retained Flash Decoding v6 experiment")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")
    add_files("src/kv_cache/*.cpp")
    add_files("src/scheduler/*.cpp")
    add_files("src/models/*.cpp")
    -- framework: model.cpp (set_weight + prepare_* stubs); weight_set.cpp removed
    add_files("src/models/framework/*.cpp")
    add_files("src/models/layers/qwen2/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    if has_config("nv-gpu") then
        add_packages("cuda")
        add_rules("cuda")
        set_policy("build.cuda.devlink", true)
        -- link cuBLAS libraries for GPU-accelerated ops
        add_linkdirs("/usr/local/cuda/lib64")
        add_syslinks("cublas", "cublasLt")
        add_ldflags("-Wl,-rpath=/usr/local/cuda/lib64")
        -- 导出全部 extern "C" 符号（实验算子 C wrapper 无 C++ 引用时也保留）
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    add_files("src/llaisys/models/*.cc")
    add_files("src/sequence/*.cpp")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()

target("llaisys-core-test")
    set_kind("binary")
    set_default(false)
    add_deps("llaisys-core")
    add_deps("llaisys-ops")
    if has_config("nv-gpu") then
        add_packages("cuda")
        add_rules("cuda")
        set_policy("build.cuda.devlink", true)
        add_linkdirs("/usr/local/cuda/lib64")
        add_syslinks("cublas", "cublasLt")
        add_ldflags("-Wl,-rpath=/usr/local/cuda/lib64")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_includedirs("src")
    add_files("test/core/*.cpp")
    add_files("src/llaisys/runtime.cc")
    add_files("src/ops/rearrange/op.cpp")
    add_files("src/sequence/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-prepare-parity-test")
    set_kind("binary")
    set_default(false)
    add_deps("llaisys-core")
    add_deps("llaisys-ops")
    if has_config("nv-gpu") then
        add_packages("cuda")
        add_rules("cuda")
        set_policy("build.cuda.devlink", true)
        add_linkdirs("/usr/local/cuda/lib64")
        add_syslinks("cublas", "cublasLt")
        add_ldflags("-Wl,-rpath=/usr/local/cuda/lib64")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_includedirs("src")
    add_files("test/models/prepare_parity_test.cpp")
    add_files("src/llaisys/runtime.cc")
    add_files("src/ops/rearrange/op.cpp")
    add_files("src/sequence/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-model-layer-cpu-test")
    set_kind("binary")
    set_default(false)
    add_deps("llaisys-core")
    add_deps("llaisys-ops")
    if has_config("nv-gpu") then
        add_packages("cuda")
        add_rules("cuda")
        set_policy("build.cuda.devlink", true)
        add_linkdirs("/usr/local/cuda/lib64")
        add_syslinks("cublas", "cublasLt")
        add_ldflags("-Wl,-rpath=/usr/local/cuda/lib64")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_includedirs("src")
    add_files("test/models/qwen2_layer_cpu_test.cpp")
    add_files("src/llaisys/runtime.cc")
    add_files("src/ops/rearrange/op.cpp")

    on_install(function (target) end)
target_end()

if has_config("nv-gpu") then
    target("llaisys-model-layer-nvidia-test")
        set_kind("binary")
        set_default(false)
        add_deps("llaisys-core")
        add_deps("llaisys-ops")
        add_packages("cuda")
        add_rules("cuda")
        set_policy("build.cuda.devlink", true)
        add_linkdirs("/usr/local/cuda/lib64")
        add_syslinks("cublas", "cublasLt")
        add_ldflags("-Wl,-rpath=/usr/local/cuda/lib64")

        set_languages("cxx17")
        set_warnings("all", "error")
        add_includedirs("src")
        add_files("test/models/qwen2_layer_nvidia_test.cpp")
        add_files("src/llaisys/runtime.cc")
        add_files("src/ops/rearrange/op.cpp")

        on_install(function (target) end)
    target_end()
end