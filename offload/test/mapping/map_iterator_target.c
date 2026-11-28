// The iterator modifier of the map clause on a target construct maps every list item of its expansion (OpenMP 5.1,
// 5.8.6; 5.2 and 6.0 the same). Clang used to map the item once, with an index that no code set: wrong results on a
// device (8 rows set to 2 gave a sum of 9, not 16), a crash with assertions. The runs on the host (the offload
// disabled) give the same results, which are those of the serial loops.
//
// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic --check-prefixes=CHECK,DEVICE
// RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic | %fcheck-generic --check-prefixes=CHECK,HOST
// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic --check-prefixes=CHECK,DEVICE
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic --check-prefixes=CHECK,DEVICE

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8

static int sum_of(int *rows[N], int column) {
  int sum = 0;
  for (int i = 0; i < N; ++i)
    sum += rows[i][column];
  return sum;
}

int main(void) {
  int *rows[N];
  for (int i = 0; i < N; ++i) {
    rows[i] = malloc(4 * sizeof(int));
    for (int j = 0; j < 4; ++j)
      rows[i][j] = 10 * i + j;
  }
  int initial = -1;

  // The array of pointers is mapped explicitly: each row's first two elements are attached to it.
#pragma omp target map(iterator(it = 0 : N), tofrom : rows[it][:2]) map(to : rows) map(from : initial)
  {
    initial = omp_is_initial_device();
    for (int k = 0; k < N; ++k) {
      rows[k][0] += 1;
      rows[k][1] += 2;
    }
  }
  // DEVICE: initial 0
  // HOST: initial 1
  printf("initial %d\n", initial);
  // The columns 2 and 3 are outside the sections: they keep their values.
  // CHECK: explicit 288 304 296 304
  printf("explicit %d %d %d %d\n", sum_of(rows, 0), sum_of(rows, 1), sum_of(rows, 2), sum_of(rows, 3));

  // The array of pointers is mapped implicitly (tofrom) because the region uses it.
#pragma omp target map(iterator(it = 0 : N), tofrom : rows[it][:2])
  for (int k = 0; k < N; ++k)
    rows[k][1] += k;
  // CHECK: implicit 332
  printf("implicit %d\n", sum_of(rows, 1));

  // A step and a bound known only at run time: the even rows (the odd rows' first elements stay 11, 31, 51, 71).
  int n = N;
#pragma omp target map(iterator(it = 0 : n : 2), tofrom : rows[it][:1])
  for (int k = 0; k < N; k += 2)
    rows[k][0] = 100;
  // CHECK: step 2 564
  printf("step 2 %d\n", sum_of(rows, 0));

  // A negative step visits the same rows from the other end.
#pragma omp target map(iterator(it = N - 1 : -1 : -1), tofrom : rows[it][:1])
  for (int k = 0; k < N; ++k)
    rows[k][0] = k;
  // CHECK: negative step 28
  printf("negative step %d\n", sum_of(rows, 0));

  // Two iterators: the expansion is every pair.
  int *grid[2][4];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j) {
      grid[i][j] = malloc(sizeof(int));
      *grid[i][j] = 0;
    }
#pragma omp target map(iterator(i = 0 : 2, j = 0 : 4), tofrom : grid[i][j][:1])
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 4; ++b)
      grid[a][b][0] = 10 * a + b;
  int gsum = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      gsum += *grid[i][j];
  // CHECK: two iterators 52
  printf("two iterators %d\n", gsum);

  // An empty iteration space maps nothing; the region still runs.
  int none = 0;
  int ran = 0;
#pragma omp target map(iterator(it = 0 : none), tofrom : rows[it][:1]) map(tofrom : ran)
  ran = 1;
  // CHECK: empty 1
  printf("empty %d\n", ran);

  // An iterator map next to a plain map of the same base pointer.
  for (int i = 0; i < N; ++i)
    rows[i][0] = 1;
#pragma omp target map(tofrom : rows[0][:1]) map(iterator(it = 1 : N), tofrom : rows[it][:1])
  for (int k = 0; k < N; ++k)
    rows[k][0] += k;
  // CHECK: mixed 36
  printf("mixed %d\n", sum_of(rows, 0));

  // A section that does not start at the first element: the kernel indexes with the host's indices. The columns 0 and 3
  // are outside the sections.
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < 4; ++j)
      rows[i][j] = 10 * i + j;
#pragma omp target map(iterator(it = 0 : N), tofrom : rows[it][1 : 2]) map(to : rows)
  for (int k = 0; k < N; ++k) {
    rows[k][1] += 100;
    rows[k][2] += 100;
  }
  // CHECK: offset section 280 1088 1096 304
  printf("offset section %d %d %d %d\n", sum_of(rows, 0), sum_of(rows, 1), sum_of(rows, 2), sum_of(rows, 3));

  // With a false if clause the region runs on the host with the host's data.
  for (int i = 0; i < N; ++i)
    rows[i][0] = 0;
  int c = 0;
#pragma omp target if (c) map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(from : initial)
  {
    initial = omp_is_initial_device();
    for (int k = 0; k < N; ++k)
      rows[k][0] = 3;
  }
  // CHECK: if false 1 24
  printf("if false %d %d\n", initial, sum_of(rows, 0));
  c = 1;
#pragma omp target if (c) map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(from : initial)
  {
    initial = omp_is_initial_device();
    for (int k = 0; k < N; ++k)
      rows[k][0] = 4;
  }
  // DEVICE: if true 0 32
  // HOST: if true 1 32
  printf("if true %d %d\n", initial, sum_of(rows, 0));
  return 0;
}
