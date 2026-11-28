// The messages of libomptarget and of the plugins have format strings that
// match their arguments on Darwin, where int64_t and uint64_t are long long
// but size_t and uintptr_t are long. The sources are compiled with the flags
// and include directories of their build, and with what their targets define.

// REQUIRES: darwin-runtime-build

// RUN: %offload_build_cxx -std=c++17 -fsyntax-only -Werror=format \
// RUN:   -DOMPTARGET_DEBUG -DTARGET_NAME=omptarget \
// RUN:   '-DDEBUG_PREFIX="omptarget"' \
// RUN:   %S/../../libomptarget/OpenMP/Mapping.cpp
// RUN: %offload_build_cxx -std=c++17 -fsyntax-only -Werror=format \
// RUN:   -DOMPTARGET_DEBUG -DTARGET_NAME=PluginInterface \
// RUN:   '-DDEBUG_PREFIX="PluginInterface"' \
// RUN:   %S/../../plugins-nextgen/common/src/PluginInterface.cpp
