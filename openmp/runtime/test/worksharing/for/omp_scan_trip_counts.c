// clang-format off
// RUN: %libomp-compile && %libomp-run
// clang-format on

// A combined parallel worksharing loop with an inscan reduction over no, one
// and a few iterations: the combine step used to run past the buffer for at
// most one iteration (a crash), and the copy of the last element to the
// original variable read before the buffer for none (the variable got
// garbage). Every result is compared with the sequential loop.
//
// The original value of the variable is the identity of the reduction (0)
// where the loop runs, and 42 where it has no iterations, which must leave it
// as it was.

#include <stdio.h>
#include <stdlib.h>

#define MAXN 20
#define GUARD 7777

static long in[MAXN], out[MAXN];

static void reset(void) {
  for (int i = 0; i < MAXN; ++i) {
    in[i] = 3 * i + 1;
    out[i] = GUARD;
  }
}

// Compares out[] and the final value of the variable with those of the
// sequential loop over n iterations that starts from init: of the inclusive
// scan x += in[i]; out[i] = x, or the exclusive one
static int expected(const char *what, int n, long init, int inclusive,
                    long final_x) {
  int errors = 0;
  long x = init;
  for (int i = 0; i < n; ++i) {
    long before = x;
    x += in[i];
    long want = inclusive ? x : before;
    if (out[i] != want) {
      printf("%s n=%d: out[%d] = %ld, expected %ld\n", what, n, i, out[i],
             want);
      ++errors;
    }
  }
  for (int i = n; i < MAXN; ++i)
    if (out[i] != GUARD) {
      printf("%s n=%d: out[%d] was written\n", what, n, i);
      ++errors;
    }
  if (final_x != x) {
    printf("%s n=%d: the variable is %ld after the loop, expected %ld\n", what,
           n, final_x, x);
    ++errors;
  }
  return errors;
}

static long parallel_for_inclusive(int n, long init) {
  long x = init;
#pragma omp parallel for reduction(inscan, + : x)
  for (int i = 0; i < n; ++i) {
    x += in[i];
#pragma omp scan inclusive(x)
    out[i] = x;
  }
  return x;
}

static long parallel_for_exclusive(int n, long init) {
  long x = init;
#pragma omp parallel for reduction(inscan, + : x)
  for (int i = 0; i < n; ++i) {
    out[i] = x;
#pragma omp scan exclusive(x)
    x += in[i];
  }
  return x;
}

static long parallel_for_simd_inclusive(int n, long init) {
  long x = init;
#pragma omp parallel for simd reduction(inscan, + : x)
  for (int i = 0; i < n; ++i) {
    x += in[i];
#pragma omp scan inclusive(x)
    out[i] = x;
  }
  return x;
}

static long parallel_for_simd_exclusive(int n, long init) {
  long x = init;
#pragma omp parallel for simd reduction(inscan, + : x)
  for (int i = 0; i < n; ++i) {
    out[i] = x;
#pragma omp scan exclusive(x)
    x += in[i];
  }
  return x;
}

// A loop over a long (the trip count is then a 64-bit value)
static long parallel_for_long_inclusive(long n, long init) {
  long x = init;
#pragma omp parallel for reduction(inscan, + : x)
  for (long i = 0; i < n; ++i) {
    x += in[i];
#pragma omp scan inclusive(x)
    out[i] = x;
  }
  return x;
}

int main(void) {
  int errors = 0;
  static const int counts[] = {0, 1, 2, 3, 4, 5, 8, 16, 17};
  for (unsigned c = 0; c < sizeof counts / sizeof counts[0]; ++c) {
    for (unsigned k = 0; k < 2; ++k) {
      int n = counts[c];
      long init = k == 0 ? 0 : 42;
      if (k == 1 && n != 0)
        continue;
      reset();
      errors += expected("parallel for inclusive", n, init, 1,
                         parallel_for_inclusive(n, init));
      reset();
      errors += expected("parallel for exclusive", n, init, 0,
                         parallel_for_exclusive(n, init));
      reset();
      errors += expected("parallel for simd inclusive", n, init, 1,
                         parallel_for_simd_inclusive(n, init));
      reset();
      errors += expected("parallel for simd exclusive", n, init, 0,
                         parallel_for_simd_exclusive(n, init));
      reset();
      errors += expected("parallel for (long) inclusive", n, init, 1,
                         parallel_for_long_inclusive(n, init));
    }
  }

  if (errors) {
    printf("FAIL: %d errors\n", errors);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
