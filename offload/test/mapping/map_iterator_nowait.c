// The iterator modifier on constructs that are deferred target tasks (nowait). The list items of the expansion are
// found when the task runs, so a task that is created by a function that has returned by then must not use that
// function's frame (the records of the iterator maps are made by the task itself and freed after its region, also on
// the host fallback). The mapped variables are file-scope: a variable of the creating function would be shared with
// the task, and the function is gone (OpenMP 5.2, 13.8).
//
// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8
#define R 4

static int *rows[N];
static int *batch[R][N];

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

// Returns right after it creates the target task.
static void launch(int r) {
  int big[256]; // the task has its own copy of this; the frame is reused below
  for (int i = 0; i < 256; ++i)
    big[i] = -i;
#pragma omp target map(iterator(it = 0 : N), tofrom : batch[r][it][:1]) firstprivate(big) nowait
  for (int k = 0; k < N; ++k)
    batch[r][k][0] += 1 + big[0];
}
// Writes over the stack that launch used.
static int smash(int depth) {
  volatile int junk[512];
  for (int i = 0; i < 512; ++i)
    junk[i] = 0x55555555;
  return depth ? smash(depth - 1) + junk[3] : junk[7];
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  for (int i = 0; i < N; ++i)
    rows[i] = malloc(2 * sizeof(int));
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < N; ++i) {
      batch[r][i] = malloc(sizeof(int));
      *batch[r][i] = r;
    }

  // The creating function returns before the task runs, and its frame is overwritten.
#pragma omp parallel num_threads(2)
#pragma omp single
  {
    for (int r = 0; r < R; ++r) {
      launch(r);
      smash(8);
#pragma omp taskwait
    }
  }
  int s = 0;
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < N; ++i)
      s += *batch[r][i];
  // CHECK: returning function 80
  printf("returning function %d\n", s);

  // All the tasks are created first and wait for each other through depend.
  init(0);
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp target map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows) depend(inout : rows) nowait
    for (int k = 0; k < N; ++k)
      rows[k][0] += 1;
#pragma omp target map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows) depend(inout : rows) nowait
    for (int k = 0; k < N; ++k)
      rows[k][0] *= 3;
#pragma omp target map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows) depend(inout : rows) nowait
    for (int k = 0; k < N; ++k)
      rows[k][0] -= 2;
#pragma omp taskwait
  }
  // CHECK: chain 92
  printf("chain %d\n", sum());

  // enter data, update to, the region, update from and exit data as tasks: every item goes through all of them.
  init(0);
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp target enter data map(to : rows) map(iterator(it = 0 : N), alloc : rows[it][:1]) depend(out : rows) nowait
#pragma omp target update to(iterator(it = 0 : N) : rows[it][:1]) depend(inout : rows) nowait
#pragma omp target depend(inout : rows) nowait
    for (int k = 0; k < N; ++k)
      rows[k][0] += 100;
#pragma omp target update from(iterator(it = 0 : N) : rows[it][:1]) depend(inout : rows) nowait
#pragma omp target exit data map(iterator(it = 0 : N), delete : rows[it][:1]) map(delete : rows) \
    depend(inout : rows) nowait
#pragma omp taskwait
  }
  // CHECK: data tasks 828
  printf("data tasks %d\n", sum());

  // Every thread creates a target task with its own two rows.
  init(0);
  int threads = 0;
#pragma omp parallel num_threads(4)
  {
    int t = omp_get_thread_num();
#pragma omp single
    threads = omp_get_num_threads();
#pragma omp target map(iterator(it = 0 : 2), tofrom : rows[2 * t + it][:1]) map(to : rows) nowait
    for (int j = 0; j < 2; ++j)
      rows[2 * t + j][0] += 1;
#pragma omp taskwait
  }
  // CHECK: threads ok 1
  printf("threads ok %d\n", sum() == 28 + 2 * threads);

  // A false if clause: the region runs on the host, in the task, and the records are freed on that path, too.
  init(0);
#pragma omp parallel num_threads(4)
#pragma omp single
  {
    for (int i = 0; i < 16; ++i) {
#pragma omp target if (0) map(iterator(it = 0 : N), tofrom : rows[it][:1]) map(to : rows) depend(inout : rows) nowait
      for (int k = 0; k < N; ++k)
        rows[k][0] += 1;
    }
#pragma omp taskwait
  }
  // CHECK: if false 156
  printf("if false %d\n", sum());
  return 0;
}
