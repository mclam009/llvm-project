// REQUIRES: darwin-clang-runtime
// RUN: %libomp_build_cxx -Werror=unused-command-line-argument -std=c++17 -fsyntax-only %s

#if __has_warning("-Wunused-but-set-global")
#pragma clang diagnostic error "-Wunused-but-set-global"
#endif
#include "../../src/kmp_affinity.cpp"
