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

// The bits of a std::vector<bool> share words, so threads must not write the bits of one word at the same time, or
// updates are lost: the OpenMP backend splits a range of bits that an algorithm writes at word boundaries. The
// ranges here split into many more pieces than they have words, begin inside a word, and are reversed, and every
// operation is repeated, as such a race loses an update only now and then.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"

static std::vector<bool> make_bits(int n, unsigned seed) {
  std::vector<bool> v(n);
  unsigned state = seed;
  for (int i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u;
    v[i]  = (state >> 13) & 1;
  }
  return v;
}

template <class Policy>
static void test_forward(Policy&& policy, int total, int start, int n) {
  const std::vector<bool> in = make_bits(total, 99), in2 = make_bits(total, 7);
  const auto first = [&](std::vector<bool>& v) { return v.begin() + start; };
  const auto last  = [&](std::vector<bool>& v) { return v.begin() + start + n; };

  { // fill
    std::vector<bool> a = in, b = in;
    std::fill(policy, first(a), last(a), true);
    std::fill(first(b), last(b), true);
    assert(a == b);
    std::fill_n(policy, first(a), n, false);
    std::fill_n(first(b), n, false);
    assert(a == b);
  }
  { // generate, generate_n
    std::vector<bool> a = in, b = in;
    std::generate(policy, first(a), last(a), [] { return true; });
    std::generate(first(b), last(b), [] { return true; });
    assert(a == b);
    std::generate_n(policy, first(a), n, [] { return false; });
    std::generate_n(first(b), n, [] { return false; });
    assert(a == b);
  }
  { // transform, with the bits as output, and in place
    std::vector<bool> a = in, b = in;
    std::transform(policy, in.begin() + start, in.begin() + start + n, first(a), [](bool x) { return !x; });
    std::transform(in.begin() + start, in.begin() + start + n, first(b), [](bool x) { return !x; });
    assert(a == b);
    std::transform(policy, first(a), last(a), first(a), [](bool x) { return !x; });
    std::transform(first(b), last(b), first(b), [](bool x) { return !x; });
    assert(a == b);
    std::transform(
        policy, in.begin() + start, in.begin() + start + n, in2.begin() + start, first(a), [](bool x, bool y) {
          return x != y;
        });
    std::transform(in.begin() + start, in.begin() + start + n, in2.begin() + start, first(b), [](bool x, bool y) {
      return x != y;
    });
    assert(a == b);
  }
  { // for_each over the proxies
    std::vector<bool> a = in, b = in;
    std::for_each(policy, first(a), last(a), [](std::vector<bool>::reference r) { r = !r; });
    std::for_each(first(b), last(b), [](std::vector<bool>::reference r) { r = !r; });
    assert(a == b);
    std::for_each_n(policy, first(a), n, [](std::vector<bool>::reference r) { r = !r; });
    std::for_each_n(first(b), n, [](std::vector<bool>::reference r) { r = !r; });
    assert(a == b);
  }
  { // copy, copy_n, replace, replace_copy
    std::vector<bool> a = in, b = in;
    std::copy(policy, in2.begin() + start, in2.begin() + start + n, first(a));
    std::copy(in2.begin() + start, in2.begin() + start + n, first(b));
    assert(a == b);
    std::copy_n(policy, in.begin() + start, n, first(a));
    std::copy_n(in.begin() + start, n, first(b));
    assert(a == b);
    std::replace(policy, first(a), last(a), true, false);
    std::replace(first(b), last(b), true, false);
    assert(a == b);
    std::replace_copy(policy, in2.begin() + start, in2.begin() + start + n, first(a), false, true);
    std::replace_copy(in2.begin() + start, in2.begin() + start + n, first(b), false, true);
    assert(a == b);
  }
  { // reading is split freely
    assert(std::count(policy, in.begin() + start, in.begin() + start + n, true) ==
           std::count(in.begin() + start, in.begin() + start + n, true));
    assert(std::reduce(policy, in.begin() + start, in.begin() + start + n, 0) ==
           std::accumulate(in.begin() + start, in.begin() + start + n, 0));
  }
}

template <class Policy>
static void test_reverse(Policy&& policy, int total, int start, int n) {
  const std::vector<bool> in = make_bits(total, 5);
  const auto rfirst          = [&](std::vector<bool>& v) { return std::make_reverse_iterator(v.begin() + start + n); };
  const auto rlast           = [&](std::vector<bool>& v) { return std::make_reverse_iterator(v.begin() + start); };
  {
    std::vector<bool> a = in, b = in;
    std::fill(policy, rfirst(a), rlast(a), true);
    std::fill(rfirst(b), rlast(b), true);
    assert(a == b);
    std::generate(policy, rfirst(a), rlast(a), [] { return false; });
    std::generate(rfirst(b), rlast(b), [] { return false; });
    assert(a == b);
    std::transform(policy, in.begin() + start, in.begin() + start + n, rfirst(a), [](bool x) { return !x; });
    std::transform(in.begin() + start, in.begin() + start + n, rfirst(b), [](bool x) { return !x; });
    assert(a == b);
    std::copy(policy, in.begin() + start, in.begin() + start + n, rfirst(a));
    std::copy(in.begin() + start, in.begin() + start + n, rfirst(b));
    assert(a == b);
  }
}

template <class Policy>
static void test_sort(Policy&& policy, int n) {
  std::vector<bool> a = make_bits(n, 3), b = a;
  std::sort(policy, a.begin(), a.end());
  std::sort(b.begin(), b.end());
  assert(a == b);
  a = make_bits(n, 4);
  b = a;
  std::stable_sort(policy, a.begin(), a.end(), std::greater<bool>());
  std::stable_sort(b.begin(), b.end(), std::greater<bool>());
  assert(a == b);
}

int main(int, char**) {
  for (int threads : {1, 2, 4}) {
    omp_set_num_threads(threads);
    // The range starts at every offset in a word of 64 bits that a thread could see, and has lengths from a few bits to
    // many words; with at most 64 pieces, the short ones have many pieces in a word.
    test_execution_policies([&](auto&& policy) {
      for (int start : {0, 1, 7, 31, 63, 64, 65, 100}) {
        for (int n : {1, 5, 63, 64, 65, 200, 1000, 4097, 20000}) {
          const int total = start + n + 70;
          test_forward(policy, total, start, n);
          test_reverse(policy, total, start, n);
        }
      }
      for (int n : {0, 1, 63, 64, 65, 999, 1000, 1001, 5000, 50000})
        test_sort(policy, n);
    });

    // The same operation again and again on a few words, where two threads touching one word would lose an update in
    // some of the rounds: 64 pieces of a few bits each.
    for (int round = 0; round < 200; ++round) {
      std::vector<bool> bits(300, false), expected(300, true);
      std::fill(std::execution::par, bits.begin() + 3, bits.end() - 5, true);
      std::fill(expected.begin(), expected.begin() + 3, false);
      std::fill(expected.end() - 5, expected.end(), false);
      assert(bits == expected);
      std::for_each(std::execution::par, bits.begin(), bits.end(), [](std::vector<bool>::reference r) { r = !r; });
      std::transform(expected.begin(), expected.end(), expected.begin(), [](bool x) { return !x; });
      assert(bits == expected);
    }
  }
  return 0;
}
