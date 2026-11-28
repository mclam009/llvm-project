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

// How the OpenMP backend splits a range into pieces: __pieces_of, __piece_count and __pieces, which decide how much
// work every thread gets, and the pieces of a std::vector<bool>, which must never share a word, as threads must not
// write the same word at the same time.

#include <__pstl/backends/openmp.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <omp.h>
#include <set>
#include <vector>

#include "test_macros.h"

namespace backend = std::__pstl::__omp_backend;

template <class Size>
void test_pieces_of() {
  const Size big = std::numeric_limits<Size>::max();

  // Nothing to split: one piece, also for no elements.
  assert(backend::__pieces_of<Size>(0, 1) == 1);
  assert(backend::__pieces_of<Size>(1, 1) == 1);
  assert(backend::__pieces_of<Size>(1, 64) == 1);
  assert(backend::__pieces_of<Size>(10, 11) == 1);
  assert(backend::__pieces_of<Size>(10, 10) == 1);

  // One piece for every grain of elements.
  assert(backend::__pieces_of<Size>(20, 10) == 2);
  assert(backend::__pieces_of<Size>(29, 10) == 2);
  assert(backend::__pieces_of<Size>(30, 10) == 3);
  assert(backend::__pieces_of<Size>(big, 2) == std::min<long long>(big / 2, backend::__max_pieces));

  // No more pieces than __max_pieces, however long the range, also where Size is as narrow as signed char
  // ([iterator.traits] allows it), and a grain that does not fit in Size is no problem either.
  const auto expected = [](long long n, long long grain) {
    return std::max<long long>(1, std::min<long long>(n / grain, backend::__max_pieces));
  };
  assert(backend::__pieces_of<Size>(big, 1) == expected(big, 1));
  assert(backend::__pieces_of<Size>(big, 1 << 30) == expected(big, 1 << 30));
  assert(backend::__pieces_of<Size>(big, std::numeric_limits<std::ptrdiff_t>::max()) == 1);
}

void test_pieces_of_all_sizes() {
  test_pieces_of<signed char>();
  test_pieces_of<short>();
  test_pieces_of<int>();
  test_pieces_of<long>();
  test_pieces_of<long long>();
  test_pieces_of<std::ptrdiff_t>();

  // Exactly at the limit and just below it.
  const long long grain = 7;
  assert(backend::__pieces_of<long long>(grain * backend::__max_pieces, grain) == backend::__max_pieces);
  assert(backend::__pieces_of<long long>(grain * backend::__max_pieces - 1, grain) == backend::__max_pieces - 1);
  assert(backend::__pieces_of<long long>(grain * (backend::__max_pieces + 1), grain) == backend::__max_pieces);
}

void test_piece_count() {
  const long long grain = 100;
  const long long n     = grain * 10;

  // Outside a parallel region the grain is the caller's.
  assert(!omp_in_parallel());
  assert(backend::__piece_count<long long>(n, grain) == 10);
  assert(backend::__piece_count<long long>(n, grain) == backend::__pieces_of<long long>(n, grain));

  // Inside an active region, where the rest of the team has work of its own, a piece has at least __nested_grain
  // elements (or the caller's grain, if that is larger).
  omp_set_num_threads(4);
  long long inside_small = 0, inside_large = 0, inside_nested = 0;
  bool active = false;
#pragma omp parallel
#pragma omp single
  {
    active        = omp_in_parallel();
    inside_small  = backend::__piece_count<long long>(n, grain);
    inside_large  = backend::__piece_count<long long>(backend::__nested_grain * 5, grain);
    inside_nested = backend::__piece_count<long long>(backend::__nested_grain * 5, 3 * backend::__nested_grain);
  }
  if (active) {
    assert(inside_small == backend::__pieces_of<long long>(n, backend::__nested_grain));
    assert(inside_small == 1);
    assert(inside_large == 5);
    assert(inside_nested == 1); // 5 * nested / (3 * nested)
  } else {
    // The machine gave the region one thread, which makes it inactive: the caller's grain applies.
    assert(inside_small == 10);
  }
}

template <class Size>
void check_partition(Size n, Size count) {
  backend::__pieces<Size> p{n, count};
  assert(p.__begin(0) == 0);
  assert(p.__end(count - 1) == n);
  assert(p.__begin(count) == n);

  const Size even = n / count;
  for (Size i = 0; i < count; ++i) {
    // A partition: consecutive, in order, and none of its pieces is empty.
    assert(p.__end(i) == p.__begin(i + 1));
    assert(p.__begin(i) < p.__end(i));
    const Size size = p.__end(i) - p.__begin(i);
    if (even < 2 * backend::__align) {
      // Nearly equal pieces.
      assert(size == even || size == even + 1);
    } else {
      // Pieces start at multiples of __align, which keeps neighbouring pieces off each other's cache lines, and
      // differ from the even share by less than that.
      if (i > 0 && i < count)
        assert(p.__begin(i) % backend::__align == 0);
      assert(size + backend::__align > even && size < even + backend::__align + 1);
    }
  }
}

void test_pieces() {
  const long long lengths[] = {1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 4096, 10007, 100000, 1000003};
  const long long counts[]  = {1, 2, 3, 7, 8, 63, 64};
  for (long long n : lengths) {
    for (long long count : counts) {
      if (count <= n)
        check_partition<long long>(n, count);
    }
    // The same in the narrowest type that holds the length.
    if (n < std::numeric_limits<short>::max())
      check_partition<short>(static_cast<short>(n), static_cast<short>(std::min<long long>(n, 7)));
  }
  check_partition<signed char>(100, 64);
  check_partition<signed char>(127, 3);
}

// Whether the pieces of the bits [first, first + n) of a std::vector<bool> that the backend makes for __count pieces
// are disjoint in the words that they touch. `position(k)` is the index of the k-th element of the range from the start
// of the vector's storage.
template <class Iterator, class Position>
void check_word_disjoint(const Iterator& first, long long n, long long count, Position position) {
  using Words = std::__specialized_algorithm<backend::__word_layout, std::__single_iterator<Iterator> >;
  static_assert(Words::__has_algorithm, "the backend knows where the elements of a std::vector<bool> lie");
  const backend::__pieces<long long> p = backend::__pieces_from<long long>(first, n, count);
  assert(p.__count_ >= 1 && p.__count_ <= count);
  assert(p.__word_ == Words::__word);
  assert(p.__offset_ == static_cast<long long>(Words::__offset(first)));
  assert(p.__begin(0) == 0);
  assert(p.__end(p.__count_ - 1) == n);

  long long previous_last_word = -1;
  for (long long i = 0; i < p.__count_; ++i) {
    const long long b = p.__begin(i), e = p.__end(i);
    assert(p.__end(i) == p.__begin(i + 1));
    assert(b <= e && e <= n);
    if (b == e)
      continue; // pieces past the last word are empty, which is allowed
    const long long low = position(b) / Words::__word, high = position(e - 1) / Words::__word;
    const long long lo = std::min(low, high), hi = std::max(low, high);
    // Forward ranges run up through the words, reverse ranges down, so compare the extremes of each piece with the
    // words of the pieces before it.
    if (previous_last_word >= 0 && position(1) >= position(0))
      assert(lo > previous_last_word);
    if (previous_last_word >= 0 && position(1) < position(0))
      assert(hi < previous_last_word);
    previous_last_word = (position(1) >= position(0)) ? hi : lo;
  }
}

void test_vector_bool_pieces() {
  using Iterator        = std::vector<bool>::iterator;
  using ConstIterator   = std::vector<bool>::const_iterator;
  using ReverseIterator = std::reverse_iterator<Iterator>;

  static_assert(backend::__writes_bits<Iterator>, "std::vector<bool>::iterator writes bits");
  static_assert(backend::__writes_bits<ReverseIterator>, "so does its reverse iterator");
  static_assert(!backend::__writes_bits<ConstIterator>, "a const_iterator does not");
  static_assert(!backend::__writes_bits<bool*>, "a bool* does not share words with other elements");
  static_assert(!backend::__writes_bits<int*>, "an int* does not");
  static_assert(backend::__knows_words<Iterator>, "the backend knows where the elements of a vector<bool> lie");
  static_assert(backend::__knows_words<ReverseIterator>, "and those of its reverse iterator");
  static_assert(!backend::__knows_words<bool*>, "but not those of a bool*");

  const long long total = 20000;
  std::vector<bool> vb(total);
  const long long starts[]  = {0, 1, 31, 63, 64, 65, 100, 127, 4097};
  const long long lengths[] = {1, 2, 63, 64, 65, 128, 129, 1000, 12345};
  const long long counts[]  = {1, 2, 3, 8, 64};
  for (long long start : starts) {
    for (long long n : lengths) {
      for (long long count : counts) {
        // Forward: element k is bit start + k.
        check_word_disjoint(vb.begin() + start, n, count, [&](long long k) { return start + k; });
        // Reverse: element k is the bit before the base, k elements down.
        const long long base = start + n;
        check_word_disjoint(ReverseIterator(vb.begin() + base), n, count, [&](long long k) { return base - 1 - k; });
      }
    }
  }
}

// An iterator of bits that the backend cannot see into: its reference is a proxy, so that elements may share words
// without the backend knowing where, and the backend must not split a range that it writes.
struct OpaqueBitIterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = bool;
  using difference_type   = std::ptrdiff_t;
  using pointer           = void;
  using reference         = std::vector<bool>::reference;

  std::vector<bool>::iterator it_;
  reference operator*() const { return *it_; }
  OpaqueBitIterator& operator++() {
    ++it_;
    return *this;
  }
  difference_type operator-(const OpaqueBitIterator& other) const { return it_ - other.it_; }
};

void test_opaque_bits() {
  static_assert(backend::__writes_bits<OpaqueBitIterator>, "a bool proxy is a write of bits");
  static_assert(!backend::__knows_words<OpaqueBitIterator>, "which the backend cannot place in words");
  std::vector<bool> vb(1000);
  const OpaqueBitIterator first{vb.begin()};
  for (long long count : {1LL, 2LL, 8LL, 64LL}) {
    const backend::__pieces<long long> p = backend::__pieces_from<long long>(first, 1000, count);
    assert(p.__count_ == 1);
    assert(p.__n_ == 1000);
    assert(p.__begin(0) == 0 && p.__end(0) == 1000);
  }

  // Ordinary elements are split as asked, and ones that are only read too.
  std::vector<int> v(1000);
  const backend::__pieces<long long> a = backend::__pieces_from<long long>(v.begin(), 1000, 8);
  assert(a.__count_ == 8 && a.__word_ == 0);
  const std::vector<bool>& cvb         = vb;
  const backend::__pieces<long long> b = backend::__pieces_from<long long>(cvb.begin(), 1000, 8);
  assert(b.__count_ == 8 && b.__word_ == 0);
}

void test_num_threads() {
  omp_set_num_threads(3);
  assert(backend::__num_threads() == omp_get_max_threads());
  int inside = 0, team = 0;
  bool active = false;
#pragma omp parallel
#pragma omp single
  {
    active = omp_in_parallel();
    inside = backend::__num_threads();
    team   = omp_get_num_threads();
  }
  // Inside an active region the team that is running is the one that counts.
  if (active)
    assert(inside == team && team >= 2);
  assert(!omp_in_parallel());
}

int main(int, char**) {
  test_pieces_of_all_sizes();
  test_piece_count();
  test_pieces();
  test_vector_bool_pieces();
  test_opaque_bits();
  test_num_threads();
  return 0;
}
