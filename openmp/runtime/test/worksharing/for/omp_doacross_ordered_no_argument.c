// clang-format off
// RUN: %libomp-compile -fopenmp-version=52 && %libomp-run
// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run
// clang-format on

// From OpenMP 5.2 an ordered clause without an argument is a doacross loop's
// when stand-alone ordered constructs with doacross clauses bind to it, as if
// its argument were the value of the collapse clause: the dependences that
// they name must hold between the iterations that different threads run.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 200
#define M 40

static int a[N];
static long b[N][M];

// a[i] = a[i - 1] + 1: a chain through all the iterations
static int one_loop(int threads, int chunk) {
  a[0] = 0;
#pragma omp parallel for ordered schedule(dynamic, chunk) num_threads(threads)
  for (int i = 1; i < N; ++i) {
#pragma omp ordered doacross(sink : i - 1)
    a[i] = a[i - 1] + 1;
#pragma omp ordered doacross(source:)
  }
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != i)
      ++errors;
  return errors;
}

// Pascal's triangle by the wavefront b[i][j] = b[i - 1][j] + b[i][j - 1]
static int two_loops(int threads, int chunk) {
  for (int j = 0; j < M; ++j)
    b[0][j] = 1;
  for (int i = 1; i < N; ++i)
    b[i][0] = 1;
#pragma omp parallel for ordered collapse(2) schedule(dynamic, chunk)          \
    num_threads(threads)
  for (int i = 1; i < N; ++i) {
    for (int j = 1; j < M; ++j) {
#pragma omp ordered doacross(sink : i - 1, j) doacross(sink : i, j - 1)
      b[i][j] = b[i - 1][j] + b[i][j - 1];
#pragma omp ordered doacross(source:)
    }
  }
  int errors = 0;
  // b[i][j] is the binomial coefficient C(i + j, i) where it fits
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      long expected = 1;
      if (i > 0 && j > 0)
        expected = b[i - 1][j] + b[i][j - 1];
      if (b[i][j] != expected)
        ++errors;
    }
  return errors;
}

// The same dependence with the iteration vector spelled omp_cur_iteration
static int cur_iteration(int threads) {
  a[0] = 0;
#pragma omp parallel for ordered schedule(dynamic, 1) num_threads(threads)
  for (int i = 1; i < N; ++i) {
#pragma omp ordered doacross(sink : omp_cur_iteration - 1)
    a[i] = a[i - 1] + 1;
#pragma omp ordered doacross(source : omp_cur_iteration)
  }
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != i)
      ++errors;
  return errors;
}

int main(void) {
  int errors = 0;
  for (int threads = 1; threads <= 8; threads *= 2) {
    for (int chunk = 1; chunk <= 4; chunk *= 2) {
      for (int rep = 0; rep < 10; ++rep) {
        errors += one_loop(threads, chunk);
        errors += two_loops(threads, chunk);
      }
    }
    errors += cur_iteration(threads);
  }
  if (errors) {
    printf("FAIL: %d errors\n", errors);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
