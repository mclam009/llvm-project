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

// <algorithm>

// Parallel algorithms called from inside an active OpenMP parallel region, from the user's code or from the element
// access functions of another parallel algorithm, run their pieces as tasks of the team that they are called from:
// they do not open nested regions (the callables see the level of the region, 1), give the results of the sequential
// algorithms, and wait for their own tasks only, so that they return while an unrelated task of the caller is still
// incomplete.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <omp.h>
#include <thread>
#include <vector>

#include "test_macros.h"

static int team_size() {
  int size = 1;
#pragma omp parallel
#pragma omp single
  size = omp_get_num_threads();
  return size;
}

// The levels that the element access functions of the nested calls see.
struct Levels {
  std::atomic<long> calls{0};
  std::atomic<long> level_one{0};
  std::atomic<long> active_level_one{0};

  void record() {
    calls.fetch_add(1, std::memory_order_relaxed);
    if (omp_get_level() == 1)
      level_one.fetch_add(1, std::memory_order_relaxed);
    if (omp_get_active_level() == 1)
      active_level_one.fetch_add(1, std::memory_order_relaxed);
  }
};

// Large enough that a call inside a region still splits (a piece has at least 2000 elements there).
static const int inner_size = 60000;

static void test_from_user_openmp_code(int team) {
  omp_set_num_threads(team);
  Levels levels;
  std::atomic<int> wrong{0};
  std::atomic<int> threads_calling{0};

  // Every thread of the team calls an algorithm on a range of its own, at the same time.
#pragma omp parallel
  {
    threads_calling.fetch_add(1);
    std::vector<int> v(inner_size, 1), out(inner_size);
    std::for_each(std::execution::par, v.begin(), v.end(), [&](int) { levels.record(); });
    std::transform(std::execution::par, v.begin(), v.end(), out.begin(), [](int x) { return x + 2; });
    if (!std::all_of(out.begin(), out.end(), [](int x) { return x == 3; }))
      ++wrong;
    const long long sum = std::transform_reduce(std::execution::par, v.begin(), v.end(), 0LL, std::plus<>(), [](int x) {
      return static_cast<long long>(x);
    });
    if (sum != inner_size)
      ++wrong;
    std::iota(out.rbegin(), out.rend(), 0);
    std::sort(std::execution::par, out.begin(), out.end());
    if (!std::is_sorted(out.begin(), out.end()))
      ++wrong;
  }
  assert(wrong == 0);
  assert(threads_calling == team);
  assert(levels.calls == static_cast<long>(team) * inner_size);
  assert(levels.level_one == levels.calls); // no nested region

  // From one thread only, while the others of the team idle: they take the pieces as tasks.
  Levels single;
  long long sum = 0;
#pragma omp parallel
#pragma omp single
  {
    std::vector<int> v(inner_size, 1);
    sum = std::transform_reduce(std::execution::par, v.begin(), v.end(), 0LL, std::plus<>(), [&](int x) {
      single.record();
      return static_cast<long long>(x);
    });
  }
  assert(sum == inner_size);
  assert(single.calls == inner_size);
  assert(single.level_one == single.calls);
  assert(single.active_level_one == single.calls);

  // From an OpenMP task.
  std::atomic<long long> from_tasks{0};
#pragma omp parallel
#pragma omp single
  {
    for (int t = 0; t < 8; ++t) {
#pragma omp task shared(from_tasks)
      {
        std::vector<int> v(inner_size, 1);
        from_tasks += std::reduce(std::execution::par, v.begin(), v.end(), 0LL);
      }
    }
  }
  assert(from_tasks == 8LL * inner_size);
}

static void test_from_a_parallel_algorithm(int team) {
  omp_set_num_threads(team);
  const int outer = 16;
  std::vector<int> indices(outer);
  std::iota(indices.begin(), indices.end(), 0);

  Levels levels;
  std::vector<long long> sums(outer, -1);
  std::vector<char> sorted(outer, 0);
  std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
    // An algorithm in the callable of another one, on a range of its own.
    std::vector<int> v(inner_size, i + 1);
    sums[i] = std::transform_reduce(std::execution::par, v.begin(), v.end(), 0LL, std::plus<>(), [&](int x) {
      levels.record();
      return static_cast<long long>(x);
    });
    std::iota(v.rbegin(), v.rend(), i);
    std::stable_sort(std::execution::par, v.begin(), v.end());
    sorted[i] = std::is_sorted(v.begin(), v.end()) && v.front() == i;
  });
  for (int i = 0; i < outer; ++i) {
    assert(sums[i] == static_cast<long long>(i + 1) * inner_size);
    assert(sorted[i]);
  }
  assert(levels.calls == static_cast<long>(outer) * inner_size);
  // The outer algorithm's region is the only one: the inner calls ran as tasks of its team.
  assert(levels.level_one == levels.calls);
  assert(!omp_in_parallel());
}

static void test_three_levels(int team) {
  omp_set_num_threads(team);
  const int a = 6, b = 5;
  std::vector<int> outer(a), middle(b);
  std::iota(outer.begin(), outer.end(), 0);
  std::iota(middle.begin(), middle.end(), 0);
  std::atomic<long long> total{0};
  Levels levels;
  std::for_each(std::execution::par, outer.begin(), outer.end(), [&](int) {
    std::for_each(std::execution::par, middle.begin(), middle.end(), [&](int) {
      std::vector<int> v(inner_size / 2, 1);
      std::for_each(std::execution::par, v.begin(), v.end(), [&](int) { levels.record(); });
      total += std::reduce(std::execution::par, v.begin(), v.end(), 0LL);
    });
  });
  assert(total == static_cast<long long>(a) * b * (inner_size / 2));
  assert(levels.calls == static_cast<long>(a) * b * (inner_size / 2));
  assert(levels.level_one == levels.calls);
}

// The call waits for its own tasks only: the detached task of the caller below is not complete until its event is
// fulfilled, which a taskwait inside the call would wait for. A watchdog fulfils it after a minute, so that a call
// that waits for it fails the test instead of hanging.
static void test_waits_for_its_own_tasks_only(int team) {
  omp_set_num_threads(team);
  std::atomic<bool> timed_out{false};
  std::atomic<bool> returned{false};
  long long sum = 0;
#pragma omp parallel
#pragma omp single
  {
    omp_event_handle_t event;
#pragma omp task detach(event)
    {
    }
    std::thread watchdog([&] {
      const auto start = std::chrono::steady_clock::now();
      while (!returned && std::chrono::steady_clock::now() - start < std::chrono::seconds(60))
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (!returned) {
        timed_out = true;
        omp_fulfill_event(event);
      }
    });
    std::vector<int> v(inner_size, 1);
    sum      = std::transform_reduce(std::execution::par, v.begin(), v.end(), 0LL, std::plus<>(), [](int x) {
      return static_cast<long long>(x);
    });
    returned = true;
    omp_fulfill_event(event); // now the caller's own task may complete
    watchdog.join();
  }
  assert(!timed_out);
  assert(sum == inner_size);
}

// A call from a region that has one thread (an inactive region) is not inside an active region: it opens its own team.
static void test_from_inactive_region(int team) {
  omp_set_num_threads(team);
  Levels levels;
  std::vector<int> v(inner_size, 1);
#pragma omp parallel num_threads(1)
  {
    assert(!omp_in_parallel());
    std::for_each(std::execution::par, v.begin(), v.end(), [&](int) { levels.record(); });
  }
  assert(levels.calls == inner_size);
  // The outer region is level 1 (inactive), the one that the backend opens level 2.
  assert(levels.level_one == 0);
}

int main(int, char**) {
  for (int requested : {2, 3, 4}) {
    omp_set_num_threads(requested);
    const int team = team_size();
    if (team < 2)
      continue; // only one thread is allowed (OMP_THREAD_LIMIT)
    test_from_user_openmp_code(team);
    test_from_a_parallel_algorithm(team);
    test_three_levels(team);
    test_waits_for_its_own_tasks_only(team);
    test_from_inactive_region(team);
  }
  return 0;
}
