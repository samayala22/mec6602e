set_project("mec6602e-hw2")
set_version("0.1.0")

add_rules("mode.debug", "mode.release", "mode.releasedbg")

-- set_policy("build.sanitizer.address", true)
set_policy("build.warning", true)
set_warnings("all")
set_languages("c++20", "c99")
set_runtimes("MD")

target("hw2")
    set_kind("binary")
    add_files("hw2.cpp")