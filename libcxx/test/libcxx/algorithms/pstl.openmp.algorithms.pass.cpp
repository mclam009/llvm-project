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

// <algorithm>, <numeric>

// The parallel algorithms with the OpenMP backend give the result of the sequential algorithms, for every execution
// policy, for ranges of no, one and a few elements and of lengths around the sizes at which the backend starts to
// split a range into pieces, and for teams of one to four threads.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"

// Lengths: empty, tiny, around the cache-line alignment of the pieces (64, 128), the smallest sort and merge pieces
// (500, 2000) and the 64 pieces that the longest ranges are split into.
static const int lengths[] = {0,   1,    2,    3,    63,   64,   65,   127,  128,  129,   499,   500,   501,
                              999, 1000, 1001, 1999, 2000, 2001, 3999, 4000, 4001, 12345, 32001, 100003};

// Values from a simple generator, small enough that there are many equal ones.
static std::vector<int> make_values(int n, unsigned seed, int modulus = 1000) {
  std::vector<int> v(n);
  unsigned state = seed;
  for (int& x : v) {
    state = state * 1664525u + 1013904223u;
    x     = static_cast<int>((state >> 8) % static_cast<unsigned>(modulus));
  }
  return v;
}

static std::vector<std::pair<int, int> > make_pairs(int n, unsigned seed, int keys) {
  const std::vector<int> k = make_values(n, seed, keys);
  std::vector<std::pair<int, int> > v(n);
  for (int i = 0; i < n; ++i)
    v[i] = {k[i], i};
  return v;
}

template <class Policy>
static void test_each(Policy&& policy, int n) {
  const std::vector<int> in = make_values(n, 1);
  {
    std::vector<int> a = in, b = in;
    std::for_each(policy, a.begin(), a.end(), [](int& x) { x = x * 3 + 1; });
    std::for_each(b.begin(), b.end(), [](int& x) { x = x * 3 + 1; });
    assert(a == b);
  }
  {
    std::vector<int> a = in, b = in;
    std::for_each_n(policy, a.begin(), n, [](int& x) { x += 7; });
    std::for_each_n(b.begin(), n, [](int& x) { x += 7; });
    assert(a == b);
    if (n > 3) {
      std::vector<int> c = in, d = in;
      std::for_each_n(policy, c.begin(), n - 3, [](int& x) { x += 7; });
      std::for_each_n(d.begin(), n - 3, [](int& x) { x += 7; });
      assert(c == d);
    }
  }
}

template <class Policy>
static void test_transform(Policy&& policy, int n) {
  const std::vector<int> in = make_values(n, 2), in2 = make_values(n, 3);
  {
    std::vector<int> a(n), b(n);
    std::transform(policy, in.begin(), in.end(), a.begin(), [](int x) { return x * x + 1; });
    std::transform(in.begin(), in.end(), b.begin(), [](int x) { return x * x + 1; });
    assert(a == b);
  }
  {
    std::vector<int> a(n), b(n);
    std::transform(policy, in.begin(), in.end(), in2.begin(), a.begin(), [](int x, int y) { return x - 2 * y; });
    std::transform(in.begin(), in.end(), in2.begin(), b.begin(), [](int x, int y) { return x - 2 * y; });
    assert(a == b);
  }
  { // in place
    std::vector<int> a = in, b = in;
    std::transform(policy, a.begin(), a.end(), a.begin(), [](int x) { return -x; });
    std::transform(b.begin(), b.end(), b.begin(), [](int x) { return -x; });
    assert(a == b);
  }
  { // to a different type
    std::vector<long long> a(n), b(n);
    std::transform(policy, in.begin(), in.end(), a.begin(), [](int x) { return 1LL << (x % 40); });
    std::transform(in.begin(), in.end(), b.begin(), [](int x) { return 1LL << (x % 40); });
    assert(a == b);
  }
  { // reversed output
    std::vector<int> a(n), b(n);
    std::transform(policy, in.begin(), in.end(), a.rbegin(), [](int x) { return x + 5; });
    std::transform(in.begin(), in.end(), b.rbegin(), [](int x) { return x + 5; });
    assert(a == b);
  }
}

template <class Policy>
static void test_reductions(Policy&& policy, int n) {
  const std::vector<int> in = make_values(n, 4), in2 = make_values(n, 5);
  assert(std::reduce(policy, in.begin(), in.end()) == std::reduce(in.begin(), in.end()));
  assert(std::reduce(policy, in.begin(), in.end(), 17) == std::accumulate(in.begin(), in.end(), 17));
  assert(std::reduce(policy, in.begin(), in.end(), 0LL, std::plus<long long>()) ==
         std::accumulate(in.begin(), in.end(), 0LL));
  assert(std::reduce(policy, in.begin(), in.end(), -1, [](int a, int b) { return std::max(a, b); }) ==
         std::accumulate(in.begin(), in.end(), -1, [](int a, int b) { return std::max(a, b); }));
  assert(std::transform_reduce(policy, in.begin(), in.end(), 0LL, std::plus<>(), [](int x) {
           return static_cast<long long>(x) * x;
         }) == std::transform_reduce(in.begin(), in.end(), 0LL, std::plus<>(), [](int x) {
           return static_cast<long long>(x) * x;
         }));
  assert(std::transform_reduce(policy, in.begin(), in.end(), in2.begin(), 0LL) ==
         std::transform_reduce(in.begin(), in.end(), in2.begin(), 0LL));
  assert(std::transform_reduce(policy, in.begin(), in.end(), in2.begin(), 5LL, std::plus<>(), [](int x, int y) {
           return static_cast<long long>(x ^ y);
         }) == std::transform_reduce(in.begin(), in.end(), in2.begin(), 5LL, std::plus<>(), [](int x, int y) {
           return static_cast<long long>(x ^ y);
         }));
  assert(std::count(policy, in.begin(), in.end(), 7) == std::count(in.begin(), in.end(), 7));
  assert(std::count_if(policy, in.begin(), in.end(), [](int x) { return x % 3 == 0; }) ==
         std::count_if(in.begin(), in.end(), [](int x) { return x % 3 == 0; }));
}

template <class Policy>
static void test_searches(Policy&& policy, int n) {
  std::vector<int> in  = make_values(n, 6, 100000); // few repeats, so that 99999 is absent
  const auto is_target = [](int x) { return x == 99999 || x == -1; };
  // The match is first, in the middle, last, several times, and absent: find returns the first one.
  std::vector<int> positions;
  if (n > 0) {
    positions = {0, n / 2, n - 1};
    if (n > 5)
      positions.push_back(n / 3);
  }
  for (std::size_t p = 0; p <= positions.size(); ++p) {
    std::vector<int> v = in;
    for (std::size_t q = 0; q < p; ++q)
      v[positions[q]] = -1;
    assert(std::find(policy, v.begin(), v.end(), -1) - v.begin() == std::find(v.begin(), v.end(), -1) - v.begin());
    assert(std::find_if(policy, v.begin(), v.end(), is_target) - v.begin() ==
           std::find_if(v.begin(), v.end(), is_target) - v.begin());
    assert(std::find_if_not(policy, v.begin(), v.end(), [](int x) { return x != -1; }) - v.begin() ==
           std::find_if_not(v.begin(), v.end(), [](int x) { return x != -1; }) - v.begin());
    assert(std::any_of(policy, v.begin(), v.end(), is_target) == std::any_of(v.begin(), v.end(), is_target));
    assert(std::all_of(policy, v.begin(), v.end(), is_target) == std::all_of(v.begin(), v.end(), is_target));
    assert(std::none_of(policy, v.begin(), v.end(), is_target) == std::none_of(v.begin(), v.end(), is_target));
    assert(std::all_of(policy, v.begin(), v.end(), [](int) { return true; }));
    assert(!std::any_of(policy, v.begin(), v.end(), [](int) { return false; }));
  }
}

template <class Policy>
static void test_comparisons(Policy&& policy, int n) {
  const std::vector<int> in = make_values(n, 7);
  std::vector<int> other    = in;
  assert(std::equal(policy, in.begin(), in.end(), other.begin()));
  assert(std::equal(policy, in.begin(), in.end(), other.begin(), other.end()));
  if (n > 0) {
    for (int where : {0, n / 2, n - 1}) {
      other = in;
      other[where] += 1;
      assert(!std::equal(policy, in.begin(), in.end(), other.begin()));
      assert(!std::equal(policy, in.begin(), in.end(), other.begin(), other.end()));
    }
    assert(!std::equal(policy, in.begin(), in.end(), other.begin(), other.end() - 1));
  }
  const auto is_even      = [](int x) { return x % 2 == 0; };
  std::vector<int> parted = in;
  std::partition(parted.begin(), parted.end(), is_even);
  assert(std::is_partitioned(policy, parted.begin(), parted.end(), is_even) ==
         std::is_partitioned(parted.begin(), parted.end(), is_even));
  assert(std::is_partitioned(policy, in.begin(), in.end(), is_even) ==
         std::is_partitioned(in.begin(), in.end(), is_even));
}

template <class Policy>
static void test_copy_fill(Policy&& policy, int n) {
  const std::vector<int> in = make_values(n, 8);
  { // copy, copy_n, move
    std::vector<int> a(n, -5), b(n, -5);
    std::copy(policy, in.begin(), in.end(), a.begin());
    std::copy(in.begin(), in.end(), b.begin());
    assert(a == b && a == in);
    std::vector<int> c(n, -5), d(n, -5);
    std::copy_n(policy, in.begin(), n, c.begin());
    std::copy_n(in.begin(), n, d.begin());
    assert(c == d && c == in);
    std::vector<int> e(n, -5);
    std::move(policy, in.begin(), in.end(), e.begin());
    assert(e == in);
  }
  { // a type with a destructor and an allocation
    std::vector<std::string> strings(n);
    for (int i = 0; i < n; ++i)
      strings[i] = std::to_string(i) + " a string that does not fit in the small buffer";
    std::vector<std::string> a(n), b(n);
    std::copy(policy, strings.begin(), strings.end(), a.begin());
    assert(a == strings);
    std::move(policy, a.begin(), a.end(), b.begin());
    assert(b == strings);
    std::fill(policy, b.begin(), b.end(), std::string("filled with a string of some length"));
    assert(std::all_of(b.begin(), b.end(), [](const std::string& s) {
      return s == "filled with a string of some length";
    }));
  }
  { // fill, fill_n, generate, generate_n, replace
    std::vector<int> a(n, 1), b(n, 1);
    std::fill(policy, a.begin(), a.end(), 9);
    std::fill(b.begin(), b.end(), 9);
    assert(a == b);
    std::fill_n(policy, a.begin(), n, 4);
    std::fill_n(b.begin(), n, 4);
    assert(a == b);
    std::generate(policy, a.begin(), a.end(), [] { return 11; });
    std::generate(b.begin(), b.end(), [] { return 11; });
    assert(a == b);
    std::generate_n(policy, a.begin(), n, [] { return 12; });
    std::generate_n(b.begin(), n, [] { return 12; });
    assert(a == b);

    std::vector<int> c = in, d = in;
    std::replace(policy, c.begin(), c.end(), 7, -7);
    std::replace(d.begin(), d.end(), 7, -7);
    assert(c == d);
    std::replace_if(policy, c.begin(), c.end(), [](int x) { return x % 5 == 0; }, 0);
    std::replace_if(d.begin(), d.end(), [](int x) { return x % 5 == 0; }, 0);
    assert(c == d);
    std::vector<int> e(n), f(n);
    std::replace_copy(policy, in.begin(), in.end(), e.begin(), 3, 33);
    std::replace_copy(in.begin(), in.end(), f.begin(), 3, 33);
    assert(e == f);
    std::replace_copy_if(policy, in.begin(), in.end(), e.begin(), [](int x) { return x < 100; }, -1);
    std::replace_copy_if(in.begin(), in.end(), f.begin(), [](int x) { return x < 100; }, -1);
    assert(e == f);
  }
  if (n > 2) { // rotate_copy
    std::vector<int> a(n), b(n);
    std::rotate_copy(policy, in.begin(), in.begin() + n / 3, in.end(), a.begin());
    std::rotate_copy(in.begin(), in.begin() + n / 3, in.end(), b.begin());
    assert(a == b);
  }
}

template <class Policy>
static void test_sorting(Policy&& policy, int n) {
  for (int modulus : {1, 2, 10, 1000, 1000000}) { // from all equal to nearly all different
    const std::vector<int> in = make_values(n, 9 + modulus, modulus);
    std::vector<int> a = in, b = in;
    std::sort(policy, a.begin(), a.end());
    std::sort(b.begin(), b.end());
    assert(a == b);
    a = in;
    b = in;
    std::sort(policy, a.begin(), a.end(), std::greater<int>());
    std::sort(b.begin(), b.end(), std::greater<int>());
    assert(a == b);
    a = in;
    b = in;
    std::stable_sort(policy, a.begin(), a.end());
    std::stable_sort(b.begin(), b.end());
    assert(a == b);
  }
  { // sorted, reversed and organ-pipe input
    std::vector<int> up(n), down(n), pipe(n);
    std::iota(up.begin(), up.end(), 0);
    std::iota(down.rbegin(), down.rend(), 0);
    for (int i = 0; i < n; ++i)
      pipe[i] = i < n / 2 ? i : n - i;
    for (std::vector<int>* v : {&up, &down, &pipe}) {
      std::vector<int> a = *v, b = *v;
      std::sort(policy, a.begin(), a.end());
      std::sort(b.begin(), b.end());
      assert(a == b);
    }
  }
  { // stability: the elements with equal keys stay in the order of the input
    for (int keys : {1, 3, 50, 100000}) {
      std::vector<std::pair<int, int> > a = make_pairs(n, 21, keys), b = a;
      const auto by_key = [](const std::pair<int, int>& x, const std::pair<int, int>& y) { return x.first < y.first; };
      std::stable_sort(policy, a.begin(), a.end(), by_key);
      std::stable_sort(b.begin(), b.end(), by_key);
      assert(a == b);
    }
  }
  { // elements that own memory, and descending order
    std::vector<std::string> a(n);
    for (int i = 0; i < n; ++i)
      a[i] = std::to_string((i * 7919) % 10007) + " a string that does not fit in the small buffer";
    std::vector<std::string> b = a;
    std::stable_sort(policy, a.begin(), a.end());
    std::stable_sort(b.begin(), b.end());
    assert(a == b);
    std::sort(policy, a.begin(), a.end(), std::greater<std::string>());
    std::sort(b.begin(), b.end(), std::greater<std::string>());
    assert(a == b);
  }
}

template <class Policy>
static void test_merge(Policy&& policy, int n) {
  for (int keys : {1, 5, 1000000}) {
    for (int split : {0, 1, n / 3, n / 2, n - 1, n}) {
      if (split < 0 || split > n)
        continue;
      std::vector<std::pair<int, int> > all = make_pairs(n, 31, keys);
      const auto by_key = [](const std::pair<int, int>& x, const std::pair<int, int>& y) { return x.first < y.first; };
      std::sort(all.begin(), all.begin() + split, by_key);
      std::sort(all.begin() + split, all.end(), by_key);
      std::vector<std::pair<int, int> > a(n), b(n);
      std::merge(policy, all.begin(), all.begin() + split, all.begin() + split, all.end(), a.begin(), by_key);
      std::merge(all.begin(), all.begin() + split, all.begin() + split, all.end(), b.begin(), by_key);
      assert(a == b); // equal keys: the first range's elements first
    }
  }
  { // ints, with ranges of very different lengths
    std::vector<int> x = make_values(n, 41), y = make_values(n / 7, 42);
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    std::vector<int> a(x.size() + y.size()), b(x.size() + y.size());
    std::merge(policy, x.begin(), x.end(), y.begin(), y.end(), a.begin());
    std::merge(x.begin(), x.end(), y.begin(), y.end(), b.begin());
    assert(a == b);
    std::merge(policy, y.begin(), y.end(), x.begin(), x.end(), a.begin());
    std::merge(y.begin(), y.end(), x.begin(), x.end(), b.begin());
    assert(a == b);
  }
}

int main(int, char**) {
  for (int threads : {1, 2, 3, 4}) {
    omp_set_num_threads(threads);
    for (int n : lengths) {
      test_execution_policies([&](auto&& policy) {
        test_each(policy, n);
        test_transform(policy, n);
        test_reductions(policy, n);
        test_searches(policy, n);
        test_comparisons(policy, n);
        test_copy_fill(policy, n);
        test_sorting(policy, n);
        test_merge(policy, n);
      });
    }
  }
  return 0;
}
