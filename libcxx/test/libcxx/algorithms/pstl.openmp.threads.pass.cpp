//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-incomplete-pstl
// UNSUPPORTED: no-threads
// REQUIRES: has-fopenmp
// ADDITIONAL_COMPILE_FLAGS: -fopenmp -D_LIBCPP_PSTL_BACKEND_OPENMP

// <algorithm>, <numeric>

// Threads that OpenMP did not create (std::thread) call the parallel algorithms, several at the same time, each on a
// team of its own, and many threads that live only for one call come and go: every call gives the right result.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <execution>
#include <functional>
#include <numeric>
#include <omp.h>
#include <thread>
#include <vector>

#include "test_macros.h"

static void work(int seed, std::atomic<int>& failures, std::atomic<int>& in_region) {
  const int n = 30000;
  std::vector<int> v(n);
  std::iota(v.begin(), v.end(), seed);
  std::atomic<int> seen_in_region{0};
  std::for_each(std::execution::par, v.begin(), v.end(), [&](int& x) {
    if (omp_in_parallel())
      seen_in_region.store(1, std::memory_order_relaxed);
    x += 1;
  });
  in_region += seen_in_region.load();
  for (int i = 0; i < n; ++i)
    if (v[i] != seed + i + 1)
      ++failures;
  const long long sum = std::transform_reduce(std::execution::par, v.begin(), v.end(), 0LL, std::plus<>(), [](int x) {
    return static_cast<long long>(x);
  });
  if (sum != static_cast<long long>(n) * (2LL * seed + n + 1) / 2)
    ++failures;
  std::reverse(v.begin(), v.end());
  std::sort(std::execution::par, v.begin(), v.end());
  if (!std::is_sorted(v.begin(), v.end()))
    ++failures;
}

int main(int, char**) {
  omp_set_num_threads(4);
  std::atomic<int> failures{0}, in_region{0};

  { // at the same time, repeatedly
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
      threads.emplace_back([&, t] {
        for (int round = 0; round < 10; ++round)
          work(t * 1000 + round, failures, in_region);
      });
    for (std::thread& t : threads)
      t.join();
    assert(failures == 0);
  }

  { // one after another, each living for one call
    for (int t = 0; t < 100; ++t) {
      std::thread([&, t] { work(t, failures, in_region); }).join();
    }
    assert(failures == 0);
  }

  { // in batches, so that threads exit while others of the batch are still in the middle of a call
    for (int batch = 0; batch < 10; ++batch) {
      std::vector<std::thread> threads;
      for (int t = 0; t < 6; ++t)
        threads.emplace_back([&, t] { work(batch * 10 + t, failures, in_region); });
      for (std::thread& t : threads)
        t.join();
    }
    assert(failures == 0);
  }

  // The calls opened a team where the machine allows one.
  int team = 1;
#pragma omp parallel
#pragma omp single
  team = omp_get_num_threads();
  if (team >= 2)
    assert(in_region > 0);
  return 0;
}
