add_rules("mode.debug", "mode.release")
-- 设置语言标准
set_languages("cxx17")
-- 添加 CUDA 支持
add_requires("cuda")

target("llaisys-device-nvidia")
    set_kind("static")
	add_rules("cuda")
	add_packages("cuda")
    set_policy("build.cuda.devlink", true)
    add_includedirs("../src")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_rules("cuda")
    add_packages("cuda")
    set_policy("build.cuda.devlink", true)
    add_includedirs("../src")
    add_deps("llaisys-tensor")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
        add_culdflags("-Xcompiler=-fPIC")
    end
    -- 目标 GPU (RTX 4060 = Ada sm_89)；mma/cp.async 需 sm_80+

    -- flash_decoding_nvidia.cu (warp-per-q 重写版) is the active Decode
    -- implementation. v4/v6_plus are archival; v3/v6 kept for A/B benchmark
    -- (llaisysFlashDecodingV3/V6 C API), v3's legacy `flash_decoding` renamed
    -- to flash_decoding_v3_legacy to avoid symbol clash with the active impl.
    add_files("../src/ops/*/nvidia/*.cu|paged_attention/nvidia/flash_decoding_v4_nvidia.cu|paged_attention/nvidia/flash_decoding_v6_nvidia_plus.cu")

    on_install(function (target) end)
target_end()

