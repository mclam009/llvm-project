//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-incomplete-pstl
// UNSUPPORTED: sanitizer-new-delete
// REQUIRES: has-fopenmp
// ADDITIONAL_COMPILE_FLAGS: -fopenmp -D_LIBCPP_PSTL_BACKEND_OPENMP

// <algorithm>

// The OpenMP backend sorts through a buffer of its own, which must be aligned for the elements: for a type that is
// aligned to more than the default operator new gives, the backend uses the aligned operator new. Here every default
// allocation is 16 bytes past a 64-byte boundary, so that only an allocation made with the alignment of the type
// comes out aligned (the allocators of some systems return blocks of that size aligned by chance).

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <new>
#include <omp.h>
#include <vector>

#include "test_macros.h"

// Allocations of the default alignment, 16 bytes past a 64-byte boundary. The address of the block that malloc gave is
// kept in the word before the one that is returned.
static void* misaligned_allocate(std::size_t size) {
  void* raw = std::malloc(size + 96);
  if (raw == nullptr)
    return nullptr;
  const std::uintptr_t base       = (reinterpret_cast<std::uintptr_t>(raw) + 8 + 63) & ~std::uintptr_t(63);
  void* result                    = reinterpret_cast<void*>(base + 16);
  static_cast<void**>(result)[-1] = raw;
  return result;
}

static void misaligned_free(void* p) noexcept {
  if (p != nullptr)
    std::free(static_cast<void**>(p)[-1]);
}

void* operator new(std::size_t size) {
  if (void* p = misaligned_allocate(size))
    return p;
  TEST_THROW(std::bad_alloc());
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return misaligned_allocate(size); }
void operator delete(void* p) noexcept { misaligned_free(p); }
void operator delete(void* p, std::size_t) noexcept { misaligned_free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { misaligned_free(p); }

struct alignas(64) Aligned {
  int key;
  int index;
};

static_assert(alignof(Aligned) == 64, "");

int main(int, char**) {
  // The replacement is in effect for the default alignment.
  {
    void* p = ::operator new(10);
    assert(reinterpret_cast<std::uintptr_t>(p) % 64 == 16);
    ::operator delete(p);
  }

  for (int threads : {2, 4}) {
    omp_set_num_threads(threads);
    for (int n : {1000, 4097, 20000}) {
      std::vector<Aligned> v(n), expected;
      unsigned state = 3;
      for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        v[i]  = Aligned{static_cast<int>((state >> 8) % 50), i};
      }
      expected = v;
      std::atomic<int> misaligned{0};
      const auto by_key = [&](const Aligned& a, const Aligned& b) {
        if (reinterpret_cast<std::uintptr_t>(&a) % alignof(Aligned) != 0 ||
            reinterpret_cast<std::uintptr_t>(&b) % alignof(Aligned) != 0)
          ++misaligned;
        return a.key < b.key;
      };
      std::stable_sort(std::execution::par, v.begin(), v.end(), by_key);
      std::stable_sort(expected.begin(), expected.end(), [](const Aligned& a, const Aligned& b) {
        return a.key < b.key;
      });
      assert(misaligned == 0);
      for (int i = 0; i < n; ++i)
        assert(v[i].key == expected[i].key && v[i].index == expected[i].index);
    }
  }
  return 0;
}
