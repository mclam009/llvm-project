// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_OPENMP_H
#define _LIBCPP___PSTL_BACKENDS_OPENMP_H

#include <__algorithm/copy_n.h>
#include <__algorithm/fill.h>
#include <__algorithm/fill_n.h>
#include <__algorithm/for_each.h>
#include <__algorithm/generate_n.h>
#include <__algorithm/max.h>
#include <__algorithm/min.h>
#include <__algorithm/specialized_algorithms.h>
#include <__algorithm/transform.h>
#include <__assert>
#include <__atomic/atomic.h>
#include <__atomic/memory_order.h>
#include <__config>
#include <__cstddef/ptrdiff_t.h>
#include <__cstddef/size_t.h>
#include <__fwd/bit_reference.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/reverse_iterator.h>
#include <__memory/construct_at.h>
#include <__memory/destroy.h>
#include <__new/align_val_t.h>
#include <__new/allocate.h>
#include <__new/global_new_delete.h>
#include <__new/nothrow_t.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/any_of.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__pstl/cpu_algos/fill.h>
#include <__pstl/cpu_algos/find_if.h>
#include <__pstl/cpu_algos/for_each.h>
#include <__pstl/cpu_algos/merge.h>
#include <__pstl/cpu_algos/stable_sort.h>
#include <__pstl/cpu_algos/transform.h>
#include <__pstl/cpu_algos/transform_reduce.h>
#include <__pstl/dispatch.h>
#include <__type_traits/common_type.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_same.h>
#include <__utility/convert_to_integral.h>
#include <__utility/element_count.h>
#include <__utility/empty.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <optional>

#if !defined(_OPENMP)
#  error "libc++'s OpenMP parallel backend (_LIBCPP_PSTL_BACKEND_OPENMP) requires compiling with -fopenmp"
#endif
#include <omp.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

// An OpenMP backend for libc++'s parallel algorithms, from pstl-omp, which also has one for GCC's libstdc++.
//
// Every operation splits its range into pieces and runs each piece on some thread:
// - A call made outside an active parallel region opens one, whose threads take the pieces in order from a dynamically
//   scheduled loop. The region has all the threads even for a few pieces, so that parallel algorithms called from the
//   element access functions can use them all.
// - A call made inside an active region (from user OpenMP code, or from an element access function of another parallel
//   algorithm) runs every piece as a task of the current team, spawned as a binary tree, rather than nesting another
//   region. No piece runs in the calling task, and the call waits for its own tasks only, so that it never waits for,
//   or runs inside, other work of the caller.
// LLVM's libomp creates and wakes tasks far more slowly than it shares a loop, so tasks are only used where a loop
// cannot be shared: pstl-omp's bench/ompcost.cpp measures the constructs.
//
// A reduction's pieces depend only on its length, so the partial results, combined in order, are the same whatever
// the number of threads, also inside parallel regions. A sort sorts its pieces as leaves and then merges pairs of runs
// level by level, through a buffer, every merge split into pieces at matching positions of the two runs. The pieces
// of a std::vector<bool> that is written hold whole words, as threads must not write the same word at the same time.
//
// The front end calls the backend from noexcept functions, so an exception escaping an element access function
// terminates the program, as the standard requires for the parallel policies, and the backend reports a failed
// allocation by returning an empty optional, which the front end turns into std::bad_alloc.

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {
namespace __omp_backend {
//! Selects the specialisations of __specialized_algorithm below.
struct __word_layout {};
} // namespace __omp_backend
} // namespace __pstl

// Where the element of a std::vector<bool> iterator lies in its word, which __bit_iterator keeps private but shows to
// every specialisation of __specialized_algorithm.
template <class _Cp>
struct __specialized_algorithm<__pstl::__omp_backend::__word_layout, __single_iterator<__bit_iterator<_Cp, false> > > {
  static const bool __has_algorithm = true;
  static constexpr unsigned __word  = __bit_iterator<_Cp, false>::__bits_per_word;

  _LIBCPP_HIDE_FROM_ABI static unsigned __offset(const __bit_iterator<_Cp, false>& __it) noexcept {
    return __it.__ctz_;
  }
};

// The same for a reverse iterator, whose elements run down from the one before base(): they reach a word boundary
// after as many elements as base() lies into its word.
template <class _Cp>
struct __specialized_algorithm<__pstl::__omp_backend::__word_layout,
                               __single_iterator<reverse_iterator<__bit_iterator<_Cp, false> > > > {
  static const bool __has_algorithm = true;
  static constexpr unsigned __word  = __bit_iterator<_Cp, false>::__bits_per_word;

  _LIBCPP_HIDE_FROM_ABI static unsigned __offset(const reverse_iterator<__bit_iterator<_Cp, false> >& __it) noexcept {
    return (__word - __it.base().__ctz_) % __word;
  }
};

namespace __pstl {
namespace __omp_backend {

//! Whether writes through _Iterator write bits of words that other elements share: its elements are bools that it
//! refers to through an assignable proxy, as std::vector<bool>'s iterators and their adaptors do.
template <class _Iterator>
inline constexpr bool __writes_bits = is_same<__iterator_value_type<_Iterator>, bool>::value &&
                                      !is_reference<__iterator_reference<_Iterator> >::value &&
                                      is_assignable<__iterator_reference<_Iterator>, bool>::value;

//! Whether the backend knows where the elements of _Iterator lie in their words: those of std::vector<bool>'s iterators
//! and their reverse iterators.
template <class _Iterator>
inline constexpr bool __knows_words =
    __specialized_algorithm<__word_layout, __single_iterator<_Iterator> >::__has_algorithm;

// A call splits its range into at most this many pieces, several per thread, so that threads that progress unevenly,
// such as those on efficiency cores, still finish at about the same time.
inline constexpr ptrdiff_t __max_pieces = 64;

// Smallest piece of a call made inside an active parallel region, where the rest of the team usually has work of its
// own, so that nested calls only split large ranges.
inline constexpr ptrdiff_t __nested_grain = 2000;

// Smallest pieces worth sorting or merging in parallel.
inline constexpr ptrdiff_t __sort_grain = 500;
inline constexpr ptrdiff_t __merge_grain = 2000;

//! The number of pieces of __n elements, each of at least __grain elements, whatever the context; 1 means no
//! splitting. The grains are not converted to _Size, which [iterator.traits] allows to be as narrow as signed char.
template <class _Size>
_LIBCPP_HIDE_FROM_ABI _Size __pieces_of(_Size __n, ptrdiff_t __grain) noexcept {
  using _Wide = common_type_t<_Size, ptrdiff_t>;
  return _Size(std::max(_Wide(1), std::min(_Wide(__n) / __grain, _Wide(__max_pieces))));
}

//! The same for the calling context, with pieces of at least __nested_grain elements inside an active parallel region.
template <class _Size>
_LIBCPP_HIDE_FROM_ABI _Size __piece_count(_Size __n, ptrdiff_t __grain) noexcept {
  return __omp_backend::__pieces_of(__n, omp_in_parallel() ? std::max(__grain, __nested_grain) : __grain);
}

//! The number of threads that can run the pieces of a call: the current team inside an active parallel region, and
//! otherwise the team that a parallel region would get.
_LIBCPP_HIDE_FROM_ABI inline int __num_threads() noexcept {
  return omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
}

// Pieces of at least twice this many elements start at multiples of it, so that neighbouring pieces of small
// elements share fewer cache lines.
inline constexpr ptrdiff_t __align = 64;

//! Partition of [0, __n) into __count contiguous pieces of nearly equal size. In a std::vector<bool>, whose first
//! element lies __offset_ elements into a word of __word_ elements, the pieces hold nearly equal numbers of whole words
//! instead, so that threads never write the same word; pieces past the last word, if there are more pieces than words,
//! are empty at __n_.
template <class _Size>
struct __pieces {
  _Size __n_;
  _Size __count_;
  _Size __offset_ = 0;
  _Size __word_   = 0; // 0 for elements that do not share words

  //! Where part __i starts when __total is split into __count_ nearly equal parts.
  _LIBCPP_HIDE_FROM_ABI _Size __share(_Size __total, _Size __i) const noexcept {
    return _Size(__i * (__total / __count_) + std::min(__i, _Size(__total % __count_)));
  }
  //! The number of words that the elements lie in, without overflowing for a range as long as _Size can count.
  _LIBCPP_HIDE_FROM_ABI _Size __words() const noexcept {
    return _Size(__n_ / __word_ + (__offset_ + __n_ % __word_ + __word_ - 1) / __word_);
  }

  _LIBCPP_HIDE_FROM_ABI _Size __begin(_Size __i) const noexcept {
    if (__i == __count_)
      return __n_;
    if (__word_ != 0)
      return std::min(__n_, std::max(_Size(0), _Size(__share(__words(), __i) * __word_ - __offset_)));
    const _Size __b = __share(__n_, __i);
    return __n_ / __count_ < 2 * __align ? __b : _Size(__b - __b % __align);
  }
  _LIBCPP_HIDE_FROM_ABI _Size __end(_Size __i) const noexcept { return __begin(_Size(__i + 1)); }

  //! __count pieces of the __len elements that start __lo elements into this partition's range.
  _LIBCPP_HIDE_FROM_ABI __pieces __sub_pieces(_Size __lo, _Size __len, _Size __count) const noexcept {
    if (__word_ == 0)
      return {__len, __count};
    return {__len, __count, _Size((__offset_ + __lo % __word_) % __word_), __word_};
  }
};

//! __count pieces of the __n elements from __first, which the caller writes: in a std::vector<bool> no more than the
//! words they lie in, and for other iterators that write bits, whose words the backend cannot see, one.
template <class _Size, class _Iterator>
_LIBCPP_HIDE_FROM_ABI __pieces<_Size> __pieces_from(const _Iterator& __first, _Size __n, _Size __count) noexcept {
  if constexpr (__knows_words<_Iterator>) {
    using _Words = __specialized_algorithm<__word_layout, __single_iterator<_Iterator> >;
    const __pieces<_Size> __all{__n, __count, _Size(_Words::__offset(__first)), _Size(_Words::__word)};
    return {__n, std::min(__count, __all.__words()), __all.__offset_, __all.__word_};
  } else if constexpr (__writes_bits<_Iterator>) {
    return {__n, _Size(1)};
  } else {
    return {__n, __count};
  }
}

//! Spawns a task of the current team for each __i in [__lo, __hi), which calls __f(__i), as a binary tree of tasks.
//! The caller waits for them, in a taskgroup.
template <class _Size, class _Fp>
_LIBCPP_HIDE_FROM_ABI void __task_tree(_Size __lo, _Size __hi, const _Fp& __f) {
  while (__hi - __lo > 1) {
    const _Size __mid = _Size(__lo + (__hi - __lo) / 2);
#  pragma omp task shared(__f)
    __omp_backend::__task_tree(__mid, __hi, __f);
    __hi = __mid;
  }
#  pragma omp task shared(__f)
  __f(__lo);
}

//! Calls __f(__i) for each __i in [0, __count), with __count at most __max_pieces, on the threads of a new team, or as
//! tasks of the current team inside an active parallel region, and waits for all the calls. __f may be called
//! concurrently, and the calls start in order of __i, so that searches skip the pieces after a match.
template <class _Size, class _Fp>
_LIBCPP_HIDE_FROM_ABI void __parallel_for(_Size __count, const _Fp& __f) {
  _LIBCPP_ASSERT_INTERNAL(__count <= __max_pieces, "too many pieces");
  if (__count < 2 || __omp_backend::__num_threads() < 2) {
    for (_Size __i = 0; __i < __count; ++__i)
      __f(__i);
    return;
  }
  if (omp_in_parallel()) {
    // Whichever task runs first takes the next piece. A cancelled enclosing parallel region or taskgroup discards the
    // tasks that have not started, so their pieces run here.
    atomic<_Size> __next(0);
    bool __done[__max_pieces] = {};
    const auto __run_next = [&](_Size) {
      const _Size __i = __next.fetch_add(1, memory_order_relaxed);
      __f(__i);
      __done[__i] = true;
    };
#  pragma omp taskgroup
    __omp_backend::__task_tree(_Size(0), __count, __run_next);
    for (_Size __i = 0; __i < __count; ++__i)
      if (!__done[__i])
        __f(__i);
    return;
  }
#  pragma omp parallel for schedule(monotonic : dynamic, 1)
  for (_Size __i = 0; __i < __count; ++__i)
    __f(__i);
}

//! Uninitialised storage for __n objects of type _Tp, or none if there is not enough memory.
template <class _Tp>
class __buffer {
  _Tp* __ptr_ = nullptr;
  size_t __n_;

public:
  _LIBCPP_HIDE_FROM_ABI explicit __buffer(size_t __n) noexcept : __n_(__n) {
    if (__n > static_cast<size_t>(-1) / sizeof(_Tp))
      return;
#  if _LIBCPP_HAS_ALIGNED_ALLOCATION
    if (std::__is_overaligned_for_new(alignof(_Tp))) {
      __ptr_ = static_cast<_Tp*>(::operator new(__n * sizeof(_Tp), static_cast<align_val_t>(alignof(_Tp)), nothrow));
      return;
    }
#  endif
    __ptr_ = static_cast<_Tp*>(::operator new(__n * sizeof(_Tp), nothrow));
  }
  __buffer(const __buffer&)            = delete;
  __buffer& operator=(const __buffer&) = delete;
  _LIBCPP_HIDE_FROM_ABI ~__buffer() {
    if (__ptr_)
      std::__libcpp_deallocate<_Tp>(__ptr_, __element_count(__n_));
  }

  _LIBCPP_HIDE_FROM_ABI explicit operator bool() const noexcept { return __ptr_ != nullptr; }
  _LIBCPP_HIDE_FROM_ABI _Tp* __get() const noexcept { return __ptr_; }
};

//! How many of the first __k elements of the stable merge of [__a, __a + __na) and [__b, __b + __nb) come from the
//! first range. Elements of the first range precede equivalent elements of the second. _Size may be wider than the
//! iterators' difference types, but every offset into a range fits that range's.
template <class _Iterator1, class _Iterator2, class _Size, class _Compare>
_LIBCPP_HIDE_FROM_ABI _Size
__split(_Iterator1 __a, _Size __na, _Iterator2 __b, _Size __nb, _Size __k, _Compare& __comp) {
  _Size __lo = std::max(_Size(0), _Size(__k - __nb));
  _Size __hi = std::min(__k, __na);
  while (__lo < __hi) {
    const _Size __mid = _Size(__lo + (__hi - __lo) / 2);
    if (__comp(*(__b + __iterator_difference_type<_Iterator2>(__k - __mid - 1)),
               *(__a + __iterator_difference_type<_Iterator1>(__mid))))
      __hi = __mid;
    else
      __lo = _Size(__mid + 1);
  }
  return __lo;
}

//! Move-constructs *__from into the uninitialised *__to.
struct __move_construct {
  template <class _Iterator, class _Tp>
  _LIBCPP_HIDE_FROM_ABI void operator()(_Iterator __from, _Tp* __to) const {
    std::__construct_at(__to, std::move(*__from));
  }
};

//! Move-assigns *__from to *__to, then destroys *__from.
struct __move_assign_destroy {
  template <class _Tp, class _Iterator>
  _LIBCPP_HIDE_FROM_ABI void operator()(_Tp* __from, _Iterator __to) const {
    *__to = std::move(*__from);
    std::__destroy_at(__from);
  }
};

//! Stable merge of [__a, __a_end) and [__b, __b_end) into __dest, moving every element with _Move.
template <class _Move, class _Iterator, class _Out, class _Compare>
_LIBCPP_HIDE_FROM_ABI void
__move_merge(_Iterator __a, _Iterator __a_end, _Iterator __b, _Iterator __b_end, _Out __dest, _Compare& __comp) {
  for (; __a != __a_end; ++__dest) {
    if (__b == __b_end) {
      for (; __a != __a_end; ++__a, (void)++__dest)
        _Move()(__a, __dest);
      return;
    }
    if (__comp(*__b, *__a)) {
      _Move()(__b, __dest);
      ++__b;
    } else {
      _Move()(__a, __dest);
      ++__a;
    }
  }
  for (; __b != __b_end; ++__b, (void)++__dest)
    _Move()(__b, __dest);
}

//! Elements __k0 to __k1 of the stable merge of the adjacent runs that start at __src and __src + __nx, moved into
//! __dst + __k0 with _Move, where the first __i0 and the first __i1 elements of the merge hold that many elements of
//! the first run.
template <class _Move, class _Iterator, class _Out, class _Size, class _Compare>
_LIBCPP_HIDE_FROM_ABI void __move_merge_piece(
    _Iterator __src, _Size __nx, _Out __dst, _Size __k0, _Size __k1, _Size __i0, _Size __i1, _Compare& __comp) {
  const _Iterator __ys = __src + __nx;
  __omp_backend::__move_merge<_Move>(
      __src + __i0, __src + __i1, __ys + (__k0 - __i0), __ys + (__k1 - __i1), __dst + __k0, __comp);
}

//! Calls __func(__piece_first, __piece_last) for the pieces of [__first, __last). The pieces of a range that the calls
//! write (_Writes) hold whole words of a std::vector<bool>; libc++'s cpu_algos also call the backend's __for_each for
//! ranges that they only read, which keep a piece for each element.
template <bool _Writes, class _RandomAccessIterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI void
__for_each_piece(_RandomAccessIterator __first, _RandomAccessIterator __last, _Functor&& __func) {
  using _Size     = __iterator_difference_type<_RandomAccessIterator>;
  const _Size __n = __last - __first;
  // A negative length is a precondition violation, which does nothing here.
  if (__n <= 0)
    return;
  const _Size __count = __omp_backend::__piece_count(__n, 1);
  __pieces<_Size> __parts{__n, __count};
  if constexpr (_Writes)
    __parts = __omp_backend::__pieces_from(__first, __n, __count);
  if (__parts.__count_ < 2 || __omp_backend::__num_threads() < 2) {
    __func(__first, __last);
    return;
  }
  __omp_backend::__parallel_for(__parts.__count_, [&](_Size __i) {
    __func(__first + __parts.__begin(__i), __first + __parts.__end(__i));
  });
}

} // namespace __omp_backend

template <>
struct __cpu_traits<__openmp_backend_tag> {
  template <class _RandomAccessIterator, class _Functor>
  _LIBCPP_HIDE_FROM_ABI static optional<__empty>
  __for_each(_RandomAccessIterator __first, _RandomAccessIterator __last, _Functor __func) {
    __omp_backend::__for_each_piece<false>(std::move(__first), std::move(__last), __func);
    return __empty{};
  }

  template <class _RandomAccessIterator, class _Transform, class _Value, class _Combiner, class _Reduction>
  _LIBCPP_HIDE_FROM_ABI static optional<_Value> __transform_reduce(
      _RandomAccessIterator __first,
      _RandomAccessIterator __last,
      _Transform __transform,
      _Value __init,
      _Combiner __combiner,
      _Reduction __reduction) {
    using _Size    = __iterator_difference_type<_RandomAccessIterator>;
    const _Size __n = __last - __first;
    // With no identity element to start from, every piece but the first starts from the combination of its first two
    // elements, so pieces need at least two. The pieces depend on the length alone, so that the result is the same in
    // every context; inside an active parallel region each task reduces several consecutive pieces.
    const _Size __count = __n > 0 ? __omp_backend::__pieces_of(__n, 2) : _Size(1);
    if (__count < 2)
      return __reduction(std::move(__first), std::move(__last), std::move(__init));
    __omp_backend::__buffer<_Value> __buffer(static_cast<size_t>(__count));
    if (!__buffer)
      return nullopt;
    _Value* const __sums = __buffer.__get();
    const __omp_backend::__pieces<_Size> __pieces{__n, __count};
    const __omp_backend::__pieces<_Size> __tasks{__count, __omp_backend::__piece_count(__n, 2)};
    // The operations may be callable only as non-const objects, so each task calls copies of its own rather than
    // share them with the others.
    __omp_backend::__parallel_for(__tasks.__count_, [&](_Size __t) {
      _Transform __tf = __transform;
      _Combiner __comb = __combiner;
      for (_Size __i = __tasks.__begin(__t); __i != __tasks.__end(__t); ++__i) {
        const _RandomAccessIterator __b = __first + __pieces.__begin(__i);
        const _RandomAccessIterator __e = __first + __pieces.__end(__i);
        if (__i == 0)
          std::__construct_at(__sums, __reduction(__b, __e, std::move(__init)));
        else
          std::__construct_at(__sums + __i, __reduction(__b + 2, __e, __comb(__tf(__b), __tf(__b + 1))));
      }
    });
    optional<_Value> __result(std::move(__sums[0]));
    for (_Size __i = 1; __i < __count; ++__i) {
      // The combination may refer to its operands, as std::max does, so it is complete before *__result changes.
      _Value __next(__combiner(std::move(*__result), std::move(__sums[__i])));
      __result.emplace(std::move(__next));
    }
    std::__destroy(__sums, __sums + __count);
    return __result;
  }

  template <class _RandomAccessIterator1,
            class _RandomAccessIterator2,
            class _RandomAccessIterator3,
            class _Compare,
            class _LeafMerge>
  _LIBCPP_HIDE_FROM_ABI static optional<__empty>
  __merge(_RandomAccessIterator1 __first1,
          _RandomAccessIterator1 __last1,
          _RandomAccessIterator2 __first2,
          _RandomAccessIterator2 __last2,
          _RandomAccessIterator3 __result,
          _Compare __comp,
          _LeafMerge __leaf_merge) {
    using _Diff1 = __iterator_difference_type<_RandomAccessIterator1>;
    using _Diff2 = __iterator_difference_type<_RandomAccessIterator2>;
    using _Diff3 = __iterator_difference_type<_RandomAccessIterator3>;
    // The output's difference type counts the whole output, which those of the inputs may not.
    using _Size                                  = common_type_t<_Diff1, _Diff2, _Diff3>;
    const _Size __n1                             = _Size(__last1 - __first1);
    const _Size __n2                             = _Size(__last2 - __first2);
    const _Size __n                              = _Size(__n1 + __n2);
    const __omp_backend::__pieces<_Size> __pieces = __omp_backend::__pieces_from(
        __result, __n, __omp_backend::__piece_count(__n, __omp_backend::__merge_grain));
    if (__pieces.__count_ < 2 || __omp_backend::__num_threads() < 2) {
      __leaf_merge(__first1, __last1, __first2, __last2, __result, __comp);
      return __empty{};
    }
    // All the splits are found before any piece merges: a leaf merge through move iterators changes the elements that
    // it moves from, which the searches for the other pieces' splits read, even with a single thread.
    _Size __splits[__omp_backend::__max_pieces + 1];
    for (_Size __i = 0; __i <= __pieces.__count_; ++__i)
      __splits[__i] = __omp_backend::__split(__first1, __n1, __first2, __n2, __pieces.__begin(__i), __comp);
    __omp_backend::__parallel_for(__pieces.__count_, [&](_Size __i) {
      _Compare __c     = __comp;
      const _Size __k0 = __pieces.__begin(__i);
      const _Size __k1 = __pieces.__end(__i);
      const _Size __i0 = __splits[__i];
      const _Size __i1 = __splits[__i + 1];
      __leaf_merge(__first1 + _Diff1(__i0),
                   __first1 + _Diff1(__i1),
                   __first2 + _Diff2(__k0 - __i0),
                   __first2 + _Diff2(__k1 - __i1),
                   __result + _Diff3(__k0),
                   __c);
    });
    return __empty{};
  }

  template <class _RandomAccessIterator, class _Comp, class _LeafSort>
  _LIBCPP_HIDE_FROM_ABI static optional<__empty>
  __stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Comp __comp, _LeafSort __leaf_sort) {
    using _Size                                = __iterator_difference_type<_RandomAccessIterator>;
    using _Value                               = __iterator_value_type<_RandomAccessIterator>;
    const _Size __n                            = __last - __first;
    const __omp_backend::__pieces<_Size> __leaf = __omp_backend::__pieces_from(
        __first, __n, __omp_backend::__piece_count(__n, __omp_backend::__sort_grain));
    const _Size __leaves = __leaf.__count_;
    if (__leaves < 2 || __omp_backend::__num_threads() < 2) {
      __leaf_sort(__first, __last, __comp);
      return __empty{};
    }
    __omp_backend::__buffer<_Value> __buffer(static_cast<size_t>(__n));
    if (!__buffer)
      return nullopt;
    _Value* const __buf = __buffer.__get();
    __omp_backend::__parallel_for(__leaves, [&](_Size __i) {
      __leaf_sort(__first + __leaf.__begin(__i), __first + __leaf.__end(__i), __comp);
    });

    // Runs of __width leaves are merged pairwise, from the range into the buffer and back, each pair in pieces so that
    // every level keeps all threads busy. Piece __p starts __splits[__p] elements into its pair's first run. All the
    // splits of a level are found before any piece moves elements out of the runs, which the searches read beyond
    // their own piece. The buffer holds objects only while it holds the elements. In a std::vector<bool> the leaves
    // and the pieces of the merges into the range hold whole words.
    bool __in_buffer = false;
    _Size __splits[__omp_backend::__max_pieces];
    for (_Size __width = 1; __width < __leaves; __width = _Size(2 * __width)) {
      const _Size __pairs      = _Size((__leaves - 1) / (2 * __width) + 1);
      const _Size __per_pair   = std::max(_Size(1), _Size(__leaves / __pairs));
      const auto __run         = [&](_Size __r) { return __leaf.__begin(std::min(_Size(__r * __width), __leaves)); };
      const auto __pair_output = [&](_Size __pair) {
        const _Size __lo = __run(_Size(2 * __pair));
        return __leaf.__sub_pieces(__lo, _Size(__run(_Size(2 * __pair + 2)) - __lo), __per_pair);
      };
      for (_Size __p = 0; __p < __pairs * __per_pair; ++__p) {
        const _Size __pair = _Size(__p / __per_pair);
        const _Size __lo   = __run(_Size(2 * __pair));
        const _Size __nx   = _Size(__run(_Size(2 * __pair + 1)) - __lo);
        const _Size __ny   = _Size(__run(_Size(2 * __pair + 2)) - __lo - __nx);
        const _Size __k    = __pair_output(__pair).__begin(_Size(__p % __per_pair));
        __splits[__p] =
            __in_buffer
                ? __omp_backend::__split(__buf + __lo, __nx, __buf + __lo + __nx, __ny, __k, __comp)
                : __omp_backend::__split(__first + __lo, __nx, __first + __lo + __nx, __ny, __k, __comp);
      }
      __omp_backend::__parallel_for(_Size(__pairs * __per_pair), [&](_Size __p) {
        _Comp __c                                        = __comp;
        const _Size __lo                                 = __run(_Size(2 * (__p / __per_pair)));
        const _Size __nx                                 = _Size(__run(_Size(2 * (__p / __per_pair) + 1)) - __lo);
        const __omp_backend::__pieces<_Size> __piece_out = __pair_output(_Size(__p / __per_pair));
        const _Size __k0                                 = __piece_out.__begin(_Size(__p % __per_pair));
        const _Size __k1                                 = __piece_out.__end(_Size(__p % __per_pair));
        const _Size __i1                                 = __p % __per_pair + 1 < __per_pair ? __splits[__p + 1] : __nx;
        if (__in_buffer)
          __omp_backend::__move_merge_piece<__omp_backend::__move_assign_destroy>(
              __buf + __lo, __nx, __first + __lo, __k0, __k1, __splits[__p], __i1, __c);
        else
          __omp_backend::__move_merge_piece<__omp_backend::__move_construct>(
              __first + __lo, __nx, __buf + __lo, __k0, __k1, __splits[__p], __i1, __c);
      });
      __in_buffer = !__in_buffer;
    }
    if (__in_buffer)
      __omp_backend::__parallel_for(__leaves, [&](_Size __i) {
        for (_Size __k = __leaf.__begin(__i); __k != __leaf.__end(__i); ++__k)
          __omp_backend::__move_assign_destroy()(__buf + __k, __first + __k);
      });
    return __empty{};
  }

  // A cancel directive must be lexically inside the cancelled construct and needs OMP_CANCELLATION=true, so this is a
  // no-op: the callers already skip the pieces that start after the result is known.
  _LIBCPP_HIDE_FROM_ABI static void __cancel_execution() {}

  static constexpr size_t __lane_size = 64;
};

// Mandatory implementations of the computational basis
template <class _ExecutionPolicy>
struct __find_if<__openmp_backend_tag, _ExecutionPolicy>
    : __cpu_parallel_find_if<__openmp_backend_tag, _ExecutionPolicy> {};

// The bits of a std::vector<bool> share words, so the backend splits a range of them that an algorithm writes into
// pieces of whole words. With -fopenmp, libc++'s cpu_algos run the unsequenced part of an algorithm, a whole range or
// each piece, as an omp simd loop, whose writes to one word overwrite one another, so the backend writes bits without
// one. A range of negative length, which only a precondition violation makes now that __fill_n and __generate_n below
// check their counts, does nothing, where with seq libc++'s cpu_algos run past the range.
template <class _ExecutionPolicy>
struct __for_each<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Function>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Function __func) const noexcept {
    constexpr bool __random_access = __has_random_access_iterator_category_or_concept<_ForwardIterator>::value;
    if constexpr (__random_access) {
      if (__last - __first <= 0)
        return __empty{};
    }
    if constexpr (!__random_access || !__omp_backend::__writes_bits<_ForwardIterator>) {
      return __cpu_parallel_for_each<__openmp_backend_tag, _ExecutionPolicy>()(
          std::forward<_Policy>(__policy), std::move(__first), std::move(__last), std::move(__func));
    } else if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy>) {
      __omp_backend::__for_each_piece<true>(
          __first, __last, [&](_ForwardIterator __piece_first, _ForwardIterator __piece_last) {
            std::for_each(__piece_first, __piece_last, __func);
          });
      return __empty{};
    } else {
      std::for_each(std::move(__first), std::move(__last), std::move(__func));
      return __empty{};
    }
  }
};

template <class _ExecutionPolicy>
struct __merge<__openmp_backend_tag, _ExecutionPolicy>
    : __cpu_parallel_merge<__openmp_backend_tag, _ExecutionPolicy> {};

template <class _ExecutionPolicy>
struct __stable_sort<__openmp_backend_tag, _ExecutionPolicy>
    : __cpu_parallel_stable_sort<__openmp_backend_tag, _ExecutionPolicy> {};

// libc++'s cpu_algos split a transform's input, which the copies and moves use too, and write the output at the same
// positions. The backend splits an output of bits instead, into pieces of whole words, and writes it without simd
// loops, as in __for_each. A negative length writes nothing and returns __result plus that length, as libc++'s
// cpu_algos do, which with seq run past the range; copy_n, which libc++'s default backend made pass one for a negative
// count, no longer gets here with one (__copy_n below).
template <class _ExecutionPolicy>
struct __transform<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _ForwardOutIterator __result,
             _UnaryOperation __op) const noexcept {
    using _Out                     = _ForwardOutIterator;
    constexpr bool __random_access = __has_random_access_iterator_category_or_concept<_ForwardIterator>::value &&
                                     __has_random_access_iterator_category_or_concept<_Out>::value;
    if constexpr (__random_access) {
      if (__last - __first <= 0)
        return __result + __iterator_difference_type<_Out>(__last - __first);
    }
    if constexpr (!__random_access || !__omp_backend::__writes_bits<_Out>) {
      return __cpu_parallel_transform<__openmp_backend_tag, _ExecutionPolicy>()(
          std::forward<_Policy>(__policy), std::move(__first), std::move(__last), std::move(__result), std::move(__op));
    } else {
      using _Size              = __iterator_difference_type<_ForwardIterator>;
      const _Out __result_last = __result + __iterator_difference_type<_Out>(__last - __first);
      if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy>) {
        __omp_backend::__for_each_piece<true>(__result, __result_last, [&](_Out __out_first, _Out __out_last) {
          std::transform(
              __first + _Size(__out_first - __result), __first + _Size(__out_last - __result), __out_first, __op);
        });
      } else {
        std::transform(std::move(__first), std::move(__last), __result, std::move(__op));
      }
      return __result_last;
    }
  }
};

template <class _ExecutionPolicy>
struct __transform_binary<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _ForwardOutIterator,
            class _BinaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardOutIterator __result,
             _BinaryOperation __op) const noexcept {
    using _Out                     = _ForwardOutIterator;
    constexpr bool __random_access = __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                                     __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value &&
                                     __has_random_access_iterator_category_or_concept<_Out>::value;
    if constexpr (__random_access) {
      if (__last1 - __first1 <= 0)
        return __result + __iterator_difference_type<_Out>(__last1 - __first1);
    }
    if constexpr (!__random_access || !__omp_backend::__writes_bits<_Out>) {
      return __cpu_parallel_transform_binary<__openmp_backend_tag, _ExecutionPolicy>()(
          std::forward<_Policy>(__policy),
          std::move(__first1),
          std::move(__last1),
          std::move(__first2),
          std::move(__result),
          std::move(__op));
    } else {
      using _Size1             = __iterator_difference_type<_ForwardIterator1>;
      using _Size2             = __iterator_difference_type<_ForwardIterator2>;
      const _Out __result_last = __result + __iterator_difference_type<_Out>(__last1 - __first1);
      if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy>) {
        __omp_backend::__for_each_piece<true>(__result, __result_last, [&](_Out __out_first, _Out __out_last) {
          std::transform(__first1 + _Size1(__out_first - __result),
                         __first1 + _Size1(__out_last - __result),
                         __first2 + _Size2(__out_first - __result),
                         __out_first,
                         __op);
        });
      } else {
        std::transform(std::move(__first1), std::move(__last1), std::move(__first2), __result, std::move(__op));
      }
      return __result_last;
    }
  }
};

// libc++'s cpu_algos wrap the transform in a lambda that is not mutable, so that they reject a transform that can only
// be called as a non-const object, which the parallel overloads accept. The backend's lambda is mutable, and leaves
// the serial and unsequenced cases, which call the transform directly, to the cpu_algos.
template <class _ExecutionPolicy>
struct __transform_reduce<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _Tp __init,
             _BinaryOperation __reduce,
             _UnaryOperation __transform) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      return __cpu_traits<__openmp_backend_tag>::__transform_reduce(
          __first,
          std::move(__last),
          [__transform](_ForwardIterator __iter) mutable { return __transform(*__iter); },
          std::move(__init),
          __reduce,
          [&__policy, __reduce, __transform](
              _ForwardIterator __brick_first, _ForwardIterator __brick_last, _Tp __brick_init) {
            using _TransformReduceUnseq =
                __pstl::__transform_reduce<__openmp_backend_tag, __remove_parallel_policy_t<_ExecutionPolicy>>;
            return *_TransformReduceUnseq()(
                std::__remove_parallel_policy(__policy),
                std::move(__brick_first),
                std::move(__brick_last),
                std::move(__brick_init),
                __reduce,
                __transform);
          });
    } else {
      return __cpu_parallel_transform_reduce<__openmp_backend_tag, _ExecutionPolicy>()(
          std::forward<_Policy>(__policy),
          std::move(__first),
          std::move(__last),
          std::move(__init),
          std::move(__reduce),
          std::move(__transform));
    }
  }
};

// libc++'s cpu_algos pass std::move(__reduce) as one argument and, as the next, a lambda that copies __reduce, which
// Clang constructs after the move, so that every piece reduces with a moved-from operation (libcxx_bugs.cpp,
// moved-from-reduction). The backend gives both a copy instead, wraps the transform in a mutable lambda, as in
// __transform_reduce, and leaves the serial and unsequenced cases, which have no such lambdas, to the cpu_algos.
template <class _ExecutionPolicy>
struct __transform_reduce_binary<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _Tp,
            class _BinaryOperation1,
            class _BinaryOperation2>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp> operator()(
      _Policy&& __policy,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _Tp __init,
      _BinaryOperation1 __reduce,
      _BinaryOperation2 __transform) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      return __cpu_traits<__openmp_backend_tag>::__transform_reduce(
          __first1,
          std::move(__last1),
          [__first1, __first2, __transform](_ForwardIterator1 __iter) mutable {
            return __transform(*__iter, *(__first2 + (__iter - __first1)));
          },
          std::move(__init),
          __reduce,
          [&__policy, __first1, __first2, __reduce, __transform](
              _ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last, _Tp __brick_init) {
            using _TransformReduceBinaryUnseq =
                __pstl::__transform_reduce_binary<__openmp_backend_tag, __remove_parallel_policy_t<_ExecutionPolicy>>;
            return *_TransformReduceBinaryUnseq()(
                std::__remove_parallel_policy(__policy),
                __brick_first,
                std::move(__brick_last),
                __first2 + (__brick_first - __first1),
                std::move(__brick_init),
                __reduce,
                __transform);
          });
    } else {
      return __cpu_parallel_transform_reduce_binary<__openmp_backend_tag, _ExecutionPolicy>()(
          std::forward<_Policy>(__policy),
          std::move(__first1),
          std::move(__last1),
          std::move(__first2),
          std::move(__init),
          std::move(__reduce),
          std::move(__transform));
    }
  }
};

// copy_n must return __result for a count of 0 or less ([alg.copy]: result + max(0, n)), but libc++'s default backend
// forms __first + __n and returns __result + __n (libcxx_bugs.cpp, copy-n-negative), and libc++ 23's serial
// std::copy_n, which it calls for other iterators, runs past the range (#193613, fixed on main). The backend returns
// __result for those counts and does what the default backend does for the others.
template <class _ExecutionPolicy>
struct __copy_n<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Size, class _ForwardOutIterator>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy, _ForwardIterator __first, _Size __n, _ForwardOutIterator __result) const noexcept {
    const auto __count = std::__convert_to_integral(__n);
    if (__count <= 0)
      return __result;
    if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      using _Copy                   = __dispatch<__copy, __current_configuration, _ExecutionPolicy>;
      const _ForwardIterator __last = __first + __iterator_difference_type<_ForwardIterator>(__count);
      return _Copy()(__policy, std::move(__first), __last, std::move(__result));
    } else {
      return std::copy_n(std::move(__first), __count, std::move(__result));
    }
  }
};

// Not mandatory, but better optimised
template <class _ExecutionPolicy>
struct __any_of<__openmp_backend_tag, _ExecutionPolicy>
    : __cpu_parallel_any_of<__openmp_backend_tag, _ExecutionPolicy> {};

// Bits and negative lengths, as in __for_each.
template <class _ExecutionPolicy>
struct __fill<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Tp>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) const noexcept {
    constexpr bool __random_access = __has_random_access_iterator_category_or_concept<_ForwardIterator>::value;
    if constexpr (__random_access) {
      if (__last - __first <= 0)
        return __empty{};
    }
    if constexpr (!__random_access || !__omp_backend::__writes_bits<_ForwardIterator>) {
      return __cpu_parallel_fill<__openmp_backend_tag, _ExecutionPolicy>()(
          std::forward<_Policy>(__policy), std::move(__first), std::move(__last), __value);
    } else if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy>) {
      __omp_backend::__for_each_piece<true>(
          __first, __last, [&](_ForwardIterator __piece_first, _ForwardIterator __piece_last) {
            std::fill(__piece_first, __piece_last, __value);
          });
      return __empty{};
    } else {
      std::fill(std::move(__first), std::move(__last), __value);
      return __empty{};
    }
  }
};

// fill_n and generate_n must do nothing for a count of 0 or less, but libc++'s default backend forms __first + __n
// before it calls __fill or __for_each, which is undefined behaviour for a negative count (UBSan reports the pointer
// overflow for a null pointer). The backend checks the count first, and otherwise does what the default backend does.
// for_each_n requires a count of at least 0, so it is left to the default backend.
template <class _ExecutionPolicy>
struct __fill_n<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Size, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _Size __n, const _Tp& __value) const noexcept {
    const auto __count = std::__convert_to_integral(__n);
    if (__count <= 0)
      return __empty{};
    if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      using _Fill                   = __dispatch<__fill, __current_configuration, _ExecutionPolicy>;
      const _ForwardIterator __last = __first + __iterator_difference_type<_ForwardIterator>(__count);
      return _Fill()(__policy, std::move(__first), __last, __value);
    } else {
      std::fill_n(std::move(__first), __count, __value);
      return __empty{};
    }
  }
};

template <class _ExecutionPolicy>
struct __generate_n<__openmp_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Size, class _Generator>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _Size __n, _Generator&& __gen) const noexcept {
    const auto __count = std::__convert_to_integral(__n);
    if (__count <= 0)
      return __empty{};
    if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      using _Generate               = __dispatch<__generate, __current_configuration, _ExecutionPolicy>;
      const _ForwardIterator __last = __first + __iterator_difference_type<_ForwardIterator>(__count);
      return _Generate()(__policy, std::move(__first), __last, std::forward<_Generator>(__gen));
    } else {
      std::generate_n(std::move(__first), __count, std::forward<_Generator>(__gen));
      return __empty{};
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_OPENMP_H
