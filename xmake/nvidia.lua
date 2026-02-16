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

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()

