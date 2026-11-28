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

// The OpenMP backend with iterators and elements of every kind that the parallel algorithms take: forward and
// random-access iterators that are not contiguous, an iterator whose difference_type is as narrow as short, elements
// that are over-aligned, that can only be moved, and whose construction and destruction are counted, so that the
// buffers of the sorts and the merges neither leak nor destroy an object twice.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <execution>
#include <forward_list>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <numeric>
#include <omp.h>
#include <vector>

#include "test_macros.h"

template <class Container>
static void test_container(int n) {
  const std::vector<int> source = [&] {
    std::vector<int> v(n);
    unsigned state = 1;
    for (int& x : v) {
      state = state * 1664525u + 1013904223u;
      x     = static_cast<int>((state >> 8) % 1000);
    }
    return v;
  }();
  Container c(source.begin(), source.end());
  const Container& cc = c;

  Container a(source.begin(), source.end()), b(source.begin(), source.end());
  std::for_each(std::execution::par, a.begin(), a.end(), [](int& x) { x += 1; });
  std::for_each(b.begin(), b.end(), [](int& x) { x += 1; });
  assert(std::equal(a.begin(), a.end(), b.begin(), b.end()));

  Container out(n), expected(n);
  std::transform(std::execution::par, cc.begin(), cc.end(), out.begin(), [](int x) { return x * 2; });
  std::transform(cc.begin(), cc.end(), expected.begin(), [](int x) { return x * 2; });
  assert(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));

  assert(std::reduce(std::execution::par, cc.begin(), cc.end(), 0) == std::accumulate(cc.begin(), cc.end(), 0));
  assert(std::transform_reduce(std::execution::par, cc.begin(), cc.end(), 0LL, std::plus<>(), [](int x) {
           return static_cast<long long>(x) * x;
         }) == std::transform_reduce(cc.begin(), cc.end(), 0LL, std::plus<>(), [](int x) {
           return static_cast<long long>(x) * x;
         }));
  assert(std::count(std::execution::par, cc.begin(), cc.end(), 5) == std::count(cc.begin(), cc.end(), 5));
  assert(std::find_if(std::execution::par, cc.begin(), cc.end(), [](int x) { return x > 990; }) ==
         std::find_if(cc.begin(), cc.end(), [](int x) { return x > 990; }));
  assert(std::any_of(std::execution::par, cc.begin(), cc.end(), [](int x) { return x == 3; }) ==
         std::any_of(cc.begin(), cc.end(), [](int x) { return x == 3; }));

  Container filled(n, 0);
  std::fill(std::execution::par, filled.begin(), filled.end(), 6);
  assert(std::all_of(filled.begin(), filled.end(), [](int x) { return x == 6; }));
  std::replace(std::execution::par, filled.begin(), filled.end(), 6, 8);
  assert(std::all_of(filled.begin(), filled.end(), [](int x) { return x == 8; }));
  Container copied(n);
  std::copy(std::execution::par, cc.begin(), cc.end(), copied.begin());
  assert(std::equal(copied.begin(), copied.end(), cc.begin(), cc.end()));
}

// A random-access iterator over ints whose difference_type is short, which [iterator.traits] allows.
struct ShortIterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = short;
  using pointer           = int*;
  using reference         = int&;

  int* p_;
  ShortIterator() = default;
  explicit ShortIterator(int* p) : p_(p) {}
  reference operator*() const { return *p_; }
  reference operator[](difference_type n) const { return p_[n]; }
  ShortIterator& operator++() {
    ++p_;
    return *this;
  }
  ShortIterator operator++(int) {
    ShortIterator t = *this;
    ++p_;
    return t;
  }
  ShortIterator& operator--() {
    --p_;
    return *this;
  }
  ShortIterator operator--(int) {
    ShortIterator t = *this;
    --p_;
    return t;
  }
  ShortIterator& operator+=(difference_type n) {
    p_ += n;
    return *this;
  }
  ShortIterator& operator-=(difference_type n) {
    p_ -= n;
    return *this;
  }
  friend ShortIterator operator+(ShortIterator it, difference_type n) {
    it += n;
    return it;
  }
  friend ShortIterator operator+(difference_type n, ShortIterator it) {
    it += n;
    return it;
  }
  friend ShortIterator operator-(ShortIterator it, difference_type n) {
    it -= n;
    return it;
  }
  friend difference_type operator-(ShortIterator a, ShortIterator b) { return static_cast<short>(a.p_ - b.p_); }
  friend bool operator==(ShortIterator a, ShortIterator b) { return a.p_ == b.p_; }
  friend bool operator!=(ShortIterator a, ShortIterator b) { return a.p_ != b.p_; }
  friend bool operator<(ShortIterator a, ShortIterator b) { return a.p_ < b.p_; }
  friend bool operator>(ShortIterator a, ShortIterator b) { return a.p_ > b.p_; }
  friend bool operator<=(ShortIterator a, ShortIterator b) { return a.p_ <= b.p_; }
  friend bool operator>=(ShortIterator a, ShortIterator b) { return a.p_ >= b.p_; }
};

static void test_narrow_difference_type() {
  for (int n : {1, 2, 100, 127, 128, 1000, 20000, 32000}) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    const ShortIterator first(v.data()), last(v.data() + n);
    assert(last - first == n);

    std::atomic<long long> sum{0};
    std::for_each(std::execution::par, first, last, [&](int x) { sum += x; });
    assert(sum == static_cast<long long>(n) * (n - 1) / 2);

    assert(std::transform_reduce(std::execution::par, first, last, 0LL, std::plus<>(), [](int x) {
             return static_cast<long long>(x);
           }) == static_cast<long long>(n) * (n - 1) / 2);
    assert(std::find_if(std::execution::par, first, last, [&](int x) { return x == n - 1; }) - first == n - 1);
    assert(std::any_of(std::execution::par, first, last, [&](int x) { return x == n - 1; }));

    std::vector<int> out(n);
    std::transform(std::execution::par, first, last, ShortIterator(out.data()), [](int x) { return x + 1; });
    for (int i = 0; i < n; ++i)
      assert(out[i] == i + 1);

    std::reverse(v.begin(), v.end());
    std::sort(std::execution::par, first, last);
    assert(std::is_sorted(v.begin(), v.end()));
    std::reverse(v.begin(), v.end());
    std::stable_sort(std::execution::par, first, last, std::greater<int>());
    assert(std::is_sorted(v.begin(), v.end(), std::greater<int>()));
  }
}

// An element that counts its constructions and destructions, can be moved but not copied, and is not trivial.
struct Tracked {
  static std::atomic<long> live;
  int key;
  int index;
  Tracked(int k, int i) : key(k), index(i) { ++live; }
  Tracked(Tracked&& other) noexcept : key(other.key), index(other.index) { ++live; }
  Tracked& operator=(Tracked&& other) noexcept {
    key   = other.key;
    index = other.index;
    return *this;
  }
  ~Tracked() { --live; }
};
std::atomic<long> Tracked::live{0};

static void test_move_only_and_counted() {
  for (int threads : {1, 2, 4}) {
    omp_set_num_threads(threads);
    for (int n : {0, 1, 999, 1000, 4097, 40000}) {
      {
        std::vector<Tracked> v;
        v.reserve(n);
        unsigned state = 11;
        for (int i = 0; i < n; ++i) {
          state = state * 1664525u + 1013904223u;
          v.emplace_back(static_cast<int>((state >> 8) % 30), i);
        }
        const long live_before = Tracked::live;
        const auto by_key      = [](const Tracked& a, const Tracked& b) { return a.key < b.key; };
        std::stable_sort(std::execution::par, v.begin(), v.end(), by_key);
        assert(Tracked::live == live_before); // the buffer's objects are all gone, and none was destroyed twice
        for (int i = 1; i < n; ++i)
          assert(v[i - 1].key < v[i].key || (v[i - 1].key == v[i].key && v[i - 1].index < v[i].index));
        std::sort(std::execution::par, v.begin(), v.end(), [](const Tracked& a, const Tracked& b) {
          return a.index < b.index;
        });
        assert(Tracked::live == live_before);
        for (int i = 0; i < n; ++i)
          assert(v[i].index == i);
      }
      assert(Tracked::live == 0);
    }
  }

  // Move-only elements that own memory.
  omp_set_num_threads(4);
  std::vector<std::unique_ptr<int> > owned;
  for (int i = 0; i < 20000; ++i)
    owned.push_back(std::make_unique<int>((i * 7919) % 10007));
  std::stable_sort(std::execution::par,
                   owned.begin(),
                   owned.end(),
                   [](const std::unique_ptr<int>& a, const std::unique_ptr<int>& b) { return *a < *b; });
  assert(std::all_of(owned.begin(), owned.end(), [](const std::unique_ptr<int>& p) { return p != nullptr; }));
  assert(std::is_sorted(owned.begin(), owned.end(), [](const std::unique_ptr<int>& a, const std::unique_ptr<int>& b) {
    return *a < *b;
  }));
}

int main(int, char**) {
  for (int threads : {1, 2, 4}) {
    omp_set_num_threads(threads);
    for (int n : {0, 1, 5, 1000, 5000}) {
      test_container<std::vector<int> >(n);
      test_container<std::deque<int> >(n);
      test_container<std::list<int> >(n);
      test_container<std::forward_list<int> >(n);
    }
  }
  test_narrow_difference_type();
  test_move_only_and_counted();
  return 0;
}
