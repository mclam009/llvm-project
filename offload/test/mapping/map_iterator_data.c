// The iterator modifier on target data, target enter data, target exit data and target update (to and from clauses
// take the modifier, too), and on the combined constructs that offload a loop. Every item of the expansion must be
// mapped, updated or released. The host runs (offload disabled) give the same results except where a line says that
// the device holds its own copy.
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

static int *rows[N];

static int sum(void) {
  int s = 0;
  for (int i = 0; i < N; ++i)
    s += rows[i][0];
  return s;
}
static void init(int base) {
  for (int i = 0; i < N; ++i)
    rows[i][0] = base + i;
}

int main(void) {
  for (int i = 0; i < N; ++i)
    rows[i] = malloc(2 * sizeof(int));

  // target data: the regions inside use the mapped items; the host sees the values at the end of the data region.
  init(0);
#pragma omp target data map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows)
  {
#pragma omp target
    for (int k = 0; k < N; ++k)
      rows[k][0] *= 2;
    // DEVICE: inside 3
    // HOST: inside 6
    printf("inside %d\n", rows[3][0]);
#pragma omp target
    for (int k = 0; k < N; ++k)
      rows[k][0] += 1;
  }
  // CHECK: target data 64
  printf("target data %d\n", sum());

  // With a false if clause nothing is mapped; with a true one every item is, for the extent of the data region. The
  // host is the device it runs on when the offload is disabled, and everything is present there.
  init(0);
  int c = 0;
  int dev = omp_get_default_device();
#pragma omp target data if (c) map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows)
  {
    int n = 0;
    for (int i = 0; i < N; ++i)
      n += omp_target_is_present(rows[i], dev);
    // DEVICE: if false present 0
    printf("if false present %d\n", n);
  }
  c = 1;
#pragma omp target data if (c) map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows)
  {
    int n = 0;
    for (int i = 0; i < N; ++i)
      n += omp_target_is_present(rows[i], dev);
    // DEVICE: if true present 8
    printf("if true present %d\n", n);
  }
  int n = 0;
  for (int i = 0; i < N; ++i)
    n += omp_target_is_present(rows[i], dev);
  // DEVICE: after present 0
  printf("after present %d\n", n);

  // enter data, update to and from, and exit data.
  init(0);
#pragma omp target enter data map(to : rows) map(iterator(it = 0 : N), to : rows[it][:2])
  for (int i = 0; i < N; ++i)
    rows[i][0] = 5;
#pragma omp target update to(iterator(it = 0 : N) : rows[it][:1])
#pragma omp target
  for (int k = 0; k < N; ++k)
    rows[k][0] += 1;
#pragma omp target update from(iterator(it = 0 : N) : rows[it][:1])
  // CHECK: update 48
  printf("update %d\n", sum());
#pragma omp target
  for (int k = 0; k < N; ++k)
    rows[k][0] = 7;
#pragma omp target exit data map(iterator(it = 0 : N), from : rows[it][:1]) map(release : rows)
  // CHECK: exit data 56
  printf("exit data %d\n", sum());

  // update with a step: only the even rows are refreshed on the device, the odd ones keep the values mapped before.
  init(0);
#pragma omp target enter data map(to : rows) map(iterator(it = 0 : N), to : rows[it][:1])
  init(100);
#pragma omp target update to(iterator(it = 0 : N : 2) : rows[it][:1])
  for (int i = 0; i < N; ++i)
    rows[i][0] = -1;
#pragma omp target update from(iterator(it = 0 : N) : rows[it][:1])
  // DEVICE: step update 100 1 102 3 104 5 106 7
  // HOST: step update -1 -1 -1 -1 -1 -1 -1 -1
  printf("step update");
  for (int i = 0; i < N; ++i)
    printf(" %d", rows[i][0]);
  printf("\n");
#pragma omp target exit data map(iterator(it = 0 : N), delete : rows[it][:1]) map(delete : rows)

  // The combined constructs that offload a loop.
  init(0);
#pragma omp target parallel for map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows)
  for (int k = 0; k < N; ++k)
    rows[k][0] += 10 * k;
  // CHECK: parallel for 308
  printf("parallel for %d\n", sum());

  init(0);
#pragma omp target teams distribute parallel for map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows)
  for (int k = 0; k < N; ++k)
    rows[k][0] += 10 * k;
  // CHECK: teams distribute parallel for 308
  printf("teams distribute parallel for %d\n", sum());

  init(0);
#pragma omp target teams map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows) num_teams(1)
  for (int k = 0; k < N; ++k)
    rows[k][0] += 10 * k;
  // CHECK: teams 308
  printf("teams %d\n", sum());

  init(0);
#pragma omp target teams loop map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows)
  for (int k = 0; k < N; ++k)
    rows[k][0] += 10 * k;
  // CHECK: teams loop 308
  printf("teams loop %d\n", sum());

  init(0);
#pragma omp target simd map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows)
  for (int k = 0; k < N; ++k)
    rows[k][0] += 10 * k;
  // CHECK: simd 308
  printf("simd %d\n", sum());
  return 0;
}
