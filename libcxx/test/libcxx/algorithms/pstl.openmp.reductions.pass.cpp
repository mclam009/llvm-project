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

// <numeric>

// The reductions of the OpenMP backend split a range into pieces that depend on its length alone and combine their
// partial results in order, so that the result of an operation that is not associative, such as the addition of
// floating-point numbers, is the same for every number of threads and in every context, also inside a parallel
// region. The operations may be function objects that can only be called as non-const objects, are not default
// constructible or, if they are moved from, invalid: every piece reduces with an operation of its own.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <execution>
#include <functional>
#include <numeric>
#include <omp.h>
#include <vector>

#include "test_macros.h"

// Values whose sum depends on the order of the additions: large ones that cancel and small ones that vanish next to
// them.
static std::vector<double> make_values(int n) {
  std::vector<double> v(n);
  unsigned state = 12345;
  for (int i = 0; i < n; ++i) {
    state                 = state * 1664525u + 1013904223u;
    const double mantissa = static_cast<double>((state >> 8) % 1000003) / 1000003.0;
    const double scale    = (i % 3 == 0) ? 1e16 : (i % 3 == 1) ? 1.0 : 1e-9;
    v[i]                  = ((state >> 4) & 1 ? -1.0 : 1.0) * mantissa * scale;
  }
  return v;
}

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

static double reduce_plain(const std::vector<double>& v) {
  return std::reduce(std::execution::par, v.begin(), v.end());
}

static double reduce_op(const std::vector<double>& v) {
  return std::reduce(std::execution::par, v.begin(), v.end(), 0.5, [](double a, double b) { return a + b; });
}

static double transform_reduce_unary(const std::vector<double>& v) {
  return std::transform_reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>(), [](double x) {
    return x * 1.0000001;
  });
}

static double transform_reduce_binary(const std::vector<double>& v) {
  return std::transform_reduce(std::execution::par, v.begin(), v.end(), v.rbegin(), 0.0);
}

typedef double (*Reduction)(const std::vector<double>&);

static void test_same_for_every_team_and_context() {
  const Reduction reductions[] = {reduce_plain, reduce_op, transform_reduce_unary, transform_reduce_binary};
  for (int n : {2, 3, 10, 1000, 4097, 100003, 1000001}) {
    const std::vector<double> v = make_values(n);
    for (Reduction reduction : reductions) {
      omp_set_num_threads(1);
      const double reference = reduction(v);

      for (int threads : {2, 3, 4, 7, 8}) {
        omp_set_num_threads(threads);
        // outside a region
        assert(same_bits(reduction(v), reference));
        // inside an active region: from every thread of the team, and from one thread
        std::vector<double> results(threads, 0.0);
        int team = 1;
#pragma omp parallel
        {
#pragma omp single
          team              = omp_get_num_threads();
          const double mine = reduction(v);
#pragma omp critical
          results[omp_get_thread_num() % threads] = mine;
        }
        for (int t = 0; t < std::min(team, threads); ++t)
          assert(same_bits(results[t], reference));
        // from the element access function of another parallel algorithm
        std::vector<int> outer(8);
        std::vector<double> nested(8, 0.0);
        std::iota(outer.begin(), outer.end(), 0);
        std::for_each(std::execution::par, outer.begin(), outer.end(), [&](int i) { nested[i] = reduction(v); });
        for (double r : nested)
          assert(same_bits(r, reference));
      }
    }
  }
}

// An operation that is invalid once it has been moved from.
template <class T>
struct MoveSensitivePlus {
  bool alive                                  = true;
  MoveSensitivePlus()                         = default;
  MoveSensitivePlus(const MoveSensitivePlus&) = default;
  MoveSensitivePlus(MoveSensitivePlus&& other) noexcept : alive(other.alive) { other.alive = false; }
  MoveSensitivePlus& operator=(const MoveSensitivePlus&) = default;
  MoveSensitivePlus& operator=(MoveSensitivePlus&& other) noexcept {
    alive       = other.alive;
    other.alive = false;
    return *this;
  }
  T operator()(const T& a, const T& b) const {
    assert(alive);
    return a + b;
  }
};

// A transformation that is invalid once it has been moved from.
struct MoveSensitiveSquare {
  bool alive                                      = true;
  MoveSensitiveSquare()                           = default;
  MoveSensitiveSquare(const MoveSensitiveSquare&) = default;
  MoveSensitiveSquare(MoveSensitiveSquare&& other) noexcept : alive(other.alive) { other.alive = false; }
  long long operator()(int x) const {
    assert(alive);
    return static_cast<long long>(x) * x;
  }
};

// A value that cannot be default constructed.
struct NoDefault {
  long long value;
  explicit NoDefault(long long v) : value(v) {}
};

static void test_operations_are_copied_not_shared() {
  omp_set_num_threads(4);
  std::vector<int> v(100003);
  std::iota(v.begin(), v.end(), 1);
  const long long expected = std::transform_reduce(v.begin(), v.end(), 0LL, std::plus<>(), MoveSensitiveSquare());

  assert(std::reduce(std::execution::par, v.begin(), v.end(), 0LL, MoveSensitivePlus<long long>()) ==
         static_cast<long long>(v.size()) * (v.size() + 1) / 2);
  assert(std::transform_reduce(
             std::execution::par, v.begin(), v.end(), 0LL, MoveSensitivePlus<long long>(), MoveSensitiveSquare()) ==
         expected);
  assert(
      std::transform_reduce(
          std::execution::par_unseq, v.begin(), v.end(), 0LL, MoveSensitivePlus<long long>(), MoveSensitiveSquare()) ==
      expected);
  assert(std::transform_reduce(
             std::execution::par, v.begin(), v.end(), v.begin(), 0LL, MoveSensitivePlus<long long>(), [](int a, int b) {
               return static_cast<long long>(a) * b;
             }) == expected);

  // Operations that are callable only as non-const objects, with state of their own: each piece has its own copy,
  // so that none of them races on it.
  long long total = std::transform_reduce(
      std::execution::par,
      v.begin(),
      v.end(),
      0LL,
      [](long long a, long long b) mutable { return a + b; },
      [scratch = 0LL](int x) mutable {
        scratch = x; // the state of this copy of the operation, which no other piece may touch
        return scratch;
      });
  assert(total == static_cast<long long>(v.size()) * (v.size() + 1) / 2);

  // A value type that cannot be default constructed: the partial results are made in uninitialised storage.
  const std::vector<NoDefault> values = [] {
    std::vector<NoDefault> r;
    for (int i = 1; i <= 50000; ++i)
      r.emplace_back(i);
    return r;
  }();
  const NoDefault sum = std::reduce(
      std::execution::par, values.begin(), values.end(), NoDefault(0), [](const NoDefault& a, const NoDefault& b) {
        return NoDefault(a.value + b.value);
      });
  assert(sum.value == 50000LL * 50001 / 2);
}

int main(int, char**) {
  test_same_for_every_team_and_context();
  test_operations_are_copied_not_shared();
  return 0;
}
