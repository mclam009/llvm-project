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

// With the OpenMP backend the element access functions of an algorithm with a parallel execution policy run on the
// threads of an OpenMP team: with a team of at least two threads two of them run at the same time, they see an active
// parallel region of one level, and the team is the one of omp_set_num_threads. With one thread, or a sequential
// policy, they run on the calling thread, outside any region.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <execution>
#include <functional>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <set>
#include <thread>
#include <vector>

#include "test_macros.h"

// What the element access functions of a call see.
struct Observed {
  std::atomic<int> calls{0};
  std::atomic<int> in_parallel{0};
  std::atomic<int> outside_level_one{0};
  std::atomic<int> max_team{0};
  std::mutex mutex;
  std::set<int> threads;
  std::set<std::thread::id> std_threads;

  void record() {
    calls.fetch_add(1, std::memory_order_relaxed);
    if (omp_in_parallel())
      in_parallel.fetch_add(1, std::memory_order_relaxed);
    if (omp_get_level() != 1)
      outside_level_one.fetch_add(1, std::memory_order_relaxed);
    const int team = omp_get_num_threads();
    int seen       = max_team.load(std::memory_order_relaxed);
    while (team > seen && !max_team.compare_exchange_weak(seen, team, std::memory_order_relaxed)) {
    }
    std::lock_guard<std::mutex> lock(mutex);
    threads.insert(omp_get_thread_num());
    std_threads.insert(std::this_thread::get_id());
  }
};

static int team_size() {
  int size = 1;
#pragma omp parallel
#pragma omp single
  size = omp_get_num_threads();
  return size;
}

// Two elements that are handled at the same time: the first to arrive waits for the second, which can only be on
// another thread, as the first one is busy waiting.
static bool two_run_at_once() {
  std::vector<int> v(10000);
  std::atomic<int> arrived{0};
  std::atomic<bool> timed_out{false};
  std::for_each(std::execution::par, v.begin(), v.end(), [&](int&) {
    if (arrived.fetch_add(1) == 0) {
      const auto start = std::chrono::steady_clock::now();
      while (arrived.load() < 2) {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(60)) {
          timed_out = true;
          return;
        }
      }
    }
  });
  return !timed_out;
}

static void check_team(const Observed& seen, int team, int calls) {
  assert(seen.calls == calls);
  assert(seen.in_parallel == calls);
  assert(seen.outside_level_one == 0);
  assert(seen.max_team == team);
  // Which threads take the pieces is up to the scheduler (one may take all of them before the others wake up), and
  // two_run_at_once() shows that two do run at the same time: here they are only threads of the team.
  assert(!seen.threads.empty());
  for (int t : seen.threads)
    assert(t >= 0 && t < team);
}

int main(int, char**) {
  for (int requested : {2, 3, 4}) {
    omp_set_num_threads(requested);
    const int team = team_size();
    if (team < 2)
      continue; // the machine or the environment (OMP_THREAD_LIMIT) allows only one thread

    // Two elements run at the same time.
    assert(two_run_at_once());

    const int n = 100000;
    std::vector<int> v(n, 1), w(n, 2), out(n);

    { // for_each, for_each_n
      Observed seen;
      std::for_each(std::execution::par, v.begin(), v.end(), [&](int) { seen.record(); });
      check_team(seen, team, n);
    }
    {
      Observed seen;
      std::for_each_n(std::execution::par_unseq, v.begin(), n, [&](int) { seen.record(); });
      check_team(seen, team, n);
    }
    { // transform, unary and binary
      Observed seen;
      std::transform(std::execution::par, v.begin(), v.end(), out.begin(), [&](int x) {
        seen.record();
        return x + 1;
      });
      check_team(seen, team, n);
      assert(std::all_of(out.begin(), out.end(), [](int x) { return x == 2; }));
    }
    {
      Observed seen;
      std::transform(std::execution::par, v.begin(), v.end(), w.begin(), out.begin(), [&](int x, int y) {
        seen.record();
        return x + y;
      });
      check_team(seen, team, n);
      assert(std::all_of(out.begin(), out.end(), [](int x) { return x == 3; }));
    }
    { // transform_reduce, with a transform and with a combiner
      Observed seen;
      const long long sum =
          std::transform_reduce(std::execution::par, v.begin(), v.end(), 0LL, std::plus<>(), [&](int x) {
            seen.record();
            return static_cast<long long>(x);
          });
      assert(sum == n);
      check_team(seen, team, n);
    }
    {
      Observed seen;
      const long long dot = std::transform_reduce(
          std::execution::par, v.begin(), v.end(), w.begin(), 0LL, std::plus<>(), [&](int x, int y) {
            seen.record();
            return static_cast<long long>(x) * y;
          });
      assert(dot == 2LL * n);
      check_team(seen, team, n);
    }
    { // a search runs its predicate on the team, not necessarily on every element
      Observed seen;
      const auto it = std::find_if(std::execution::par, v.begin(), v.end(), [&](int x) {
        seen.record();
        return x == 7;
      });
      assert(it == v.end());
      check_team(seen, team, n);
      Observed any;
      assert(!std::any_of(std::execution::par, v.begin(), v.end(), [&](int x) {
        any.record();
        return x == 7;
      }));
      check_team(any, team, n);
    }
    { // generate, fill-like writes
      Observed seen;
      std::generate(std::execution::par, out.begin(), out.end(), [&] {
        seen.record();
        return 5;
      });
      check_team(seen, team, n);
      assert(std::all_of(out.begin(), out.end(), [](int x) { return x == 5; }));
    }
    { // a sort and a merge run their comparisons on the team, the ones that find where to split the runs on the
      // calling thread, so only some of them see the region
      std::vector<int> data(n);
      std::iota(data.rbegin(), data.rend(), 0);
      Observed seen;
      std::sort(std::execution::par, data.begin(), data.end(), [&](int a, int b) {
        seen.record();
        return a < b;
      });
      assert(std::is_sorted(data.begin(), data.end()));
      assert(seen.in_parallel > 0);
      assert(seen.max_team == team);
      assert(seen.outside_level_one == seen.calls - seen.in_parallel); // level 1 inside the region, 0 outside

      std::vector<int> a(n / 2), b(n / 2), merged(n);
      std::iota(a.begin(), a.end(), 0);
      std::iota(b.begin(), b.end(), 0);
      Observed merging;
      std::merge(std::execution::par, a.begin(), a.end(), b.begin(), b.end(), merged.begin(), [&](int x, int y) {
        merging.record();
        return x < y;
      });
      assert(std::is_sorted(merged.begin(), merged.end()));
      assert(merging.in_parallel > 0);
    }

    { // a sequential policy never leaves the calling thread, and is not in a region
      Observed seen;
      std::for_each(std::execution::seq, v.begin(), v.end(), [&](int) { seen.record(); });
      assert(seen.calls == n && seen.in_parallel == 0 && seen.max_team == 1);
      assert(seen.std_threads.size() == 1 && *seen.std_threads.begin() == std::this_thread::get_id());
    }
    assert(!omp_in_parallel());
  }

  { // A team of one thread: the algorithm runs on the calling thread, outside any region.
    omp_set_num_threads(1);
    std::vector<int> v(100000, 1);
    Observed seen;
    std::for_each(std::execution::par, v.begin(), v.end(), [&](int) { seen.record(); });
    assert(seen.calls == 100000);
    assert(seen.in_parallel == 0);
    assert(seen.max_team == 1);
    assert(seen.std_threads.size() == 1 && *seen.std_threads.begin() == std::this_thread::get_id());

    std::vector<int> data(100000);
    std::iota(data.rbegin(), data.rend(), 0);
    std::sort(std::execution::par, data.begin(), data.end());
    assert(std::is_sorted(data.begin(), data.end()));
  }

  return 0;
}
