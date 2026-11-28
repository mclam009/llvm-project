//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// Selecting the OpenMP backend of the parallel algorithms without compiling with OpenMP is an error that says so,
// not a failure to find the pragmas' effects or the runtime later.

// ADDITIONAL_COMPILE_FLAGS: -fno-openmp -D_LIBCPP_PSTL_BACKEND_OPENMP

#include <algorithm>

// expected-error@*:* {{OpenMP parallel backend (_LIBCPP_PSTL_BACKEND_OPENMP) requires compiling with -fopenmp}}
