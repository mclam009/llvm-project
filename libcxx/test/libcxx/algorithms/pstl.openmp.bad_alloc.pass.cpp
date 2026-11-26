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
// REQUIRES: has-fopenmp
// ADDITIONAL_COMPILE_FLAGS: -fopenmp -D_LIBCPP_PSTL_BACKEND_OPENMP

// <algorithm>

// The OpenMP backend sorts a range of at least two pieces through a buffer. When it cannot get the memory it returns
// an empty optional, which libc++'s front end turns into std::bad_alloc, and it has not touched a single element by
// then. The range here has 2^60 elements, so that no machine has the memory, and its iterator aborts if it is
// dereferenced.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <execution>
#include <iterator>
#include <new>
#include <omp.h>

#include "test_macros.h"

struct NeverDereferenced {
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = char;
  using difference_type   = std::ptrdiff_t;
  using pointer           = char*;
  using reference         = char&;

  std::ptrdiff_t position_;

  reference operator*() const { std::abort(); }
  reference operator[](difference_type) const { std::abort(); }
  NeverDereferenced& operator++() {
    ++position_;
    return *this;
  }
  NeverDereferenced operator++(int) {
    NeverDereferenced t = *this;
    ++position_;
    return t;
  }
  NeverDereferenced& operator--() {
    --position_;
    return *this;
  }
  NeverDereferenced operator--(int) {
    NeverDereferenced t = *this;
    --position_;
    return t;
  }
  NeverDereferenced& operator+=(difference_type n) {
    position_ += n;
    return *this;
  }
  NeverDereferenced& operator-=(difference_type n) {
    position_ -= n;
    return *this;
  }
  friend NeverDereferenced operator+(NeverDereferenced it, difference_type n) {
    it += n;
    return it;
  }
  friend NeverDereferenced operator+(difference_type n, NeverDereferenced it) {
    it += n;
    return it;
  }
  friend NeverDereferenced operator-(NeverDereferenced it, difference_type n) {
    it -= n;
    return it;
  }
  friend difference_type operator-(NeverDereferenced a, NeverDereferenced b) { return a.position_ - b.position_; }
  friend bool operator==(NeverDereferenced a, NeverDereferenced b) { return a.position_ == b.position_; }
  friend bool operator!=(NeverDereferenced a, NeverDereferenced b) { return a.position_ != b.position_; }
  friend bool operator<(NeverDereferenced a, NeverDereferenced b) { return a.position_ < b.position_; }
  friend bool operator>(NeverDereferenced a, NeverDereferenced b) { return a.position_ > b.position_; }
  friend bool operator<=(NeverDereferenced a, NeverDereferenced b) { return a.position_ <= b.position_; }
  friend bool operator>=(NeverDereferenced a, NeverDereferenced b) { return a.position_ >= b.position_; }
};

int main(int, char**) {
  // At least two threads, as a sort on one is a sequential sort, which would dereference the elements.
  omp_set_num_threads(4);
  assert(omp_get_max_threads() >= 2);

  const NeverDereferenced first{0}, last{std::ptrdiff_t(1) << 60};

  bool stable_sort_threw = false;
  try {
    std::stable_sort(std::execution::par, first, last);
  } catch (const std::bad_alloc&) {
    stable_sort_threw = true;
  }
  assert(stable_sort_threw);

  bool sort_threw = false;
  try {
    std::sort(std::execution::par_unseq, first, last, [](char a, char b) { return a < b; });
  } catch (const std::bad_alloc&) {
    sort_threw = true;
  }
  assert(sort_threw);

  // The same inside a parallel region.
  bool threw_in_region = false;
#pragma omp parallel num_threads(4)
#pragma omp single
  {
    try {
      std::stable_sort(std::execution::par, first, last);
    } catch (const std::bad_alloc&) {
      threw_in_region = true;
    }
  }
  assert(threw_in_region);
  return 0;
}
