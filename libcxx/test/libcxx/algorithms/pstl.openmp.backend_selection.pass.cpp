//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-incomplete-pstl
// REQUIRES: has-fopenmp
// ADDITIONAL_COMPILE_FLAGS: -fopenmp -D_LIBCPP_PSTL_BACKEND_OPENMP

// <algorithm>

// Defining _LIBCPP_PSTL_BACKEND_OPENMP selects the OpenMP backend of the parallel algorithms, over the backend that
// __config_site defines, and the algorithms with a parallel execution policy then run on the threads of an OpenMP
// team. The algorithms with a sequential policy keep running on the calling thread.

#include <__pstl/backends/openmp.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <execution>
#include <numeric>
#include <omp.h>
#include <type_traits>
#include <vector>

#include "test_macros.h"

// The configuration that the algorithms are dispatched through is the OpenMP backend's, with the default backend
// for everything that it does not implement.
static_assert(
    std::is_same_v<
        std::__pstl::__current_configuration,
        std::__pstl::__backend_configuration<std::__pstl::__openmp_backend_tag, std::__pstl::__default_backend_tag> >,
    "_LIBCPP_PSTL_BACKEND_OPENMP must select the OpenMP backend");

// The tags of the other backends are not what the algorithms are dispatched through.
static_assert(
    !std::is_same_v<
        std::__pstl::__current_configuration,
        std::__pstl::__backend_configuration<std::__pstl::__serial_backend_tag, std::__pstl::__default_backend_tag> >,
    "the serial backend must not be selected");

// The size of the team that a parallel region gets here: the limit of the machine, or of the environment, may be lower
// than what the test asks for.
static int team_size() {
  omp_set_num_threads(4);
  int size = 1;
#pragma omp parallel
#pragma omp single
  size = omp_get_num_threads();
  return size;
}

struct Counts {
  std::atomic<int> in_parallel{0};
  std::atomic<int> max_team{0};
  std::atomic<int> total{0};

  void record() {
    total.fetch_add(1, std::memory_order_relaxed);
    if (omp_in_parallel())
      in_parallel.fetch_add(1, std::memory_order_relaxed);
    const int team = omp_get_num_threads();
    int seen       = max_team.load(std::memory_order_relaxed);
    while (team > seen && !max_team.compare_exchange_weak(seen, team, std::memory_order_relaxed)) {
    }
  }
};

int main(int, char**) {
  assert(_OPENMP != 0);
  const int team = team_size();
  std::vector<int> v(100000, 1);

  { // a parallel policy runs on an OpenMP team, with every thread that the machine gives it
    Counts counts;
    std::for_each(std::execution::par, v.begin(), v.end(), [&](int) { counts.record(); });
    assert(counts.total == static_cast<int>(v.size()));
    if (team >= 2) {
      assert(counts.in_parallel == counts.total);
      assert(counts.max_team == team);
    } else {
      assert(counts.max_team <= 1);
    }
  }

  { // and so does par_unseq
    Counts counts;
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int) { counts.record(); });
    assert(counts.total == static_cast<int>(v.size()));
    if (team >= 2)
      assert(counts.in_parallel == counts.total);
  }

  { // a sequential policy stays on the calling thread, outside any parallel region
    Counts counts;
    std::for_each(std::execution::seq, v.begin(), v.end(), [&](int) { counts.record(); });
    assert(counts.total == static_cast<int>(v.size()));
    assert(counts.in_parallel == 0);
  }

  { // the algorithms that the backend implements give the answer of the sequential ones
    const std::ptrdiff_t n =
        std::transform_reduce(std::execution::par, v.begin(), v.end(), std::ptrdiff_t(0), std::plus<>(), [](int x) {
          return std::ptrdiff_t(x);
        });
    assert(n == static_cast<std::ptrdiff_t>(v.size()));
    assert(std::reduce(std::execution::par, v.begin(), v.end(), 0) == static_cast<int>(v.size()));
  }

  return 0;
}
