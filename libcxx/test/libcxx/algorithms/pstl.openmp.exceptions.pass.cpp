//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: libcpp-has-no-incomplete-pstl
// `check_assertion.h` requires Unix headers and regex support.
// REQUIRES: has-unix-headers
// UNSUPPORTED: no-localization
// REQUIRES: has-fopenmp
// ADDITIONAL_COMPILE_FLAGS: -fopenmp -D_LIBCPP_PSTL_BACKEND_OPENMP

// <algorithm>, <numeric>

// An exception that leaves an element access function of a parallel algorithm calls std::terminate with the OpenMP
// backend, whichever thread of the team ran the function, in the calling task or in a task of a nested call, and
// does not propagate to the caller or hang the team. (pstl.exception_handling.pass.cpp does this for ranges of a few
// elements, which the backend runs on the calling thread.)

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <execution>
#include <functional>
#include <numeric>
#include <omp.h>
#include <vector>

#include "check_assertion.h"
#include "test_macros.h"

static const int n = 100000;

struct Boom {};

// Throws for exactly one element, somewhere in the middle of a piece that is not the first.
static bool is_the_one(int x) { return x == n / 2 + 17; }

// The algorithm returning means that the exception was swallowed: the child process then exits normally, which
// EXPECT_STD_TERMINATE reports as a failure. An exception that leaves the lambda is not terminate() either.
template <class F>
static void must_terminate(F f) {
  EXPECT_STD_TERMINATE([&] {
    f();
    // nothing terminated: return, which fails the expectation
  });
}

int main(int, char**) {
  omp_set_num_threads(4);
  std::vector<int> v(n);
  std::iota(v.begin(), v.end(), 0);
  std::vector<int> w = v, out(n);

  for (auto policy : {0, 1}) { // par, par_unseq
    const auto run = [&](auto&& body) {
      if (policy == 0)
        must_terminate([&] { body(std::execution::par); });
      else
        must_terminate([&] { body(std::execution::par_unseq); });
    };

    run([&](auto&& p) {
      std::for_each(p, v.begin(), v.end(), [](int x) {
        if (is_the_one(x))
          throw Boom();
      });
    });
    run([&](auto&& p) {
      std::for_each_n(p, v.begin(), n, [](int x) {
        if (is_the_one(x))
          throw Boom();
      });
    });
    run([&](auto&& p) {
      std::transform(p, v.begin(), v.end(), out.begin(), [](int x) -> int {
        if (is_the_one(x))
          throw Boom();
        return x;
      });
    });
    run([&](auto&& p) {
      std::transform(p, v.begin(), v.end(), w.begin(), out.begin(), [](int x, int y) -> int {
        if (is_the_one(y))
          throw Boom();
        return x + y;
      });
    });
    run([&](auto&& p) {
      (void)std::transform_reduce(p, v.begin(), v.end(), 0LL, std::plus<>(), [](int x) -> long long {
        if (is_the_one(x))
          throw Boom();
        return x;
      });
    });
    // the combiner sees partial results, not elements: it throws whenever it is called, which it is for 100000 elements
    run([&](auto&& p) {
      (void)std::transform_reduce(
          p,
          v.begin(),
          v.end(),
          0LL,
          [](long long, long long) -> long long { throw Boom(); },
          [](int x) -> long long { return x; });
    });
    run([&](auto&& p) { (void)std::reduce(p, v.begin(), v.end(), 0, [](int, int) -> int { throw Boom(); }); });
    run([&](auto&& p) {
      (void)std::find_if(p, v.begin(), v.end(), [](int x) -> bool {
        if (is_the_one(x))
          throw Boom();
        return false;
      });
    });
    run([&](auto&& p) {
      (void)std::any_of(p, v.begin(), v.end(), [](int x) -> bool {
        if (is_the_one(x))
          throw Boom();
        return false;
      });
    });
    run([&](auto&& p) { std::generate(p, out.begin(), out.end(), []() -> int { throw Boom(); }); });
    // comparisons of the leaves of a sort and of the pieces of a merge run on the threads of the team
    run([&](auto&& p) { std::sort(p, w.rbegin(), w.rend(), [](int, int) -> bool { throw Boom(); }); });
    run([&](auto&& p) { std::stable_sort(p, w.rbegin(), w.rend(), [](int, int) -> bool { throw Boom(); }); });
    run([&](auto&& p) {
      std::vector<int> a(n / 2), b(n / 2), merged(n);
      std::iota(a.begin(), a.end(), 0);
      std::iota(b.begin(), b.end(), 0);
      std::merge(p, a.begin(), a.end(), b.begin(), b.end(), merged.begin(), [](int, int) -> bool { throw Boom(); });
    });

    // from inside an element access function of another parallel algorithm, and so from a task
    run([&](auto&& p) {
      std::vector<int> outer(8);
      std::iota(outer.begin(), outer.end(), 0);
      std::for_each(p, outer.begin(), outer.end(), [&](int i) {
        std::for_each(p, v.begin(), v.end(), [&](int x) {
          if (i == 5 && is_the_one(x))
            throw Boom();
        });
      });
    });

    // from user OpenMP code that calls an algorithm in a region, from every thread and from a task
    run([&](auto&& p) {
#pragma omp parallel
      std::for_each(p, v.begin(), v.end(), [](int x) {
        if (is_the_one(x))
          throw Boom();
      });
    });
    run([&](auto&& p) {
#pragma omp parallel
#pragma omp single
      {
#pragma omp task
        std::for_each(p, v.begin(), v.end(), [](int x) {
          if (is_the_one(x))
            throw Boom();
        });
      }
    });
  }

  return 0;
}
