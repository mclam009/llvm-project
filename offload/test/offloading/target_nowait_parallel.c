// A target nowait construct that a thread of a parallel region encounters is a deferred target task: it must run its
// region once, and complete after the taskwait, taskgroup or depend that orders it. On macOS libomp was built without
// its hooks for libomptarget (OPENMP_ENABLE_LIBOMPTARGET was off on Apple), so it ran the region again and again while
// the device work was pending (461 times in one run) and then failed in the CUDA plugin, which did not make its context
// current on the thread that polls the stream (CUDA_ERROR_INVALID_CONTEXT). The runs on the host (offload disabled)
// give the same results.
//
// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic --check-prefixes=CHECK,DEVICE
// RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic | %fcheck-generic --check-prefixes=CHECK,HOST

#include <omp.h>
#include <stdio.h>

int main(void) {
  // Unbuffered, so that the lines before a failure of the runtime are not lost
  setvbuf(stdout, NULL, _IONBF, 0);

  // One thread creates the task, and waits for it with taskwait: the region ran once.
  int count = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp target map(tofrom : count) nowait
    count += 1;
#pragma omp taskwait
  }
  // CHECK: single 1
  printf("single %d\n", count);

  // A chain through depend: each region runs once, after the one before.
  int v = 1;
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp target map(tofrom : v) depend(inout : v) nowait
    v = v + 1;
#pragma omp target map(tofrom : v) depend(inout : v) nowait
    v = v * 10;
#pragma omp target map(tofrom : v) depend(inout : v) nowait
    v = v - 3;
#pragma omp taskwait
  }
  // CHECK: chain 17
  printf("chain %d\n", v);

  // Many in a row on one variable: a region that ran twice would be seen in the count.
  count = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
  {
    for (int i = 0; i < 8; ++i) {
#pragma omp target map(tofrom : count) depend(inout : count) nowait
      count += 1;
    }
#pragma omp taskwait
  }
  // CHECK: many 8
  printf("many %d\n", count);

  // enter data, the region and exit data as deferred tasks ordered by depend.
  int a[4] = {1, 2, 3, 4};
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp target enter data map(to : a) depend(out : a) nowait
#pragma omp target map(tofrom : a) depend(inout : a) nowait
    for (int i = 0; i < 4; ++i)
      a[i] *= 2;
#pragma omp target exit data map(from : a) depend(inout : a) nowait
#pragma omp taskwait
  }
  // CHECK: data 2 4 6 8
  printf("data %d %d %d %d\n", a[0], a[1], a[2], a[3]);

  // update to and from as deferred tasks, between enter data and exit data.
  int b[4] = {5, 6, 7, 8};
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp target enter data map(alloc : b) depend(out : b) nowait
#pragma omp target update to(b) depend(inout : b) nowait
#pragma omp target map(tofrom : b) depend(inout : b) nowait
    for (int i = 0; i < 4; ++i)
      b[i] += 1;
#pragma omp target update from(b) depend(inout : b) nowait
#pragma omp target exit data map(delete : b) depend(inout : b) nowait
#pragma omp taskwait
  }
  // CHECK: update 6 7 8 9
  printf("update %d %d %d %d\n", b[0], b[1], b[2], b[3]);

  // Every thread of the team creates its own target task.
  int slot[8] = {0};
  int threads = 0;
#pragma omp parallel num_threads(4)
  {
    int t = omp_get_thread_num();
#pragma omp single
    threads = omp_get_num_threads();
#pragma omp target map(tofrom : slot[t : 1]) nowait
    slot[t] += 1;
#pragma omp taskwait
  }
  int total = 0, most = 0;
  for (int i = 0; i < 8; ++i) {
    total += slot[i];
    most = slot[i] > most ? slot[i] : most;
  }
  // CHECK: threads ok 1 most 1
  printf("threads ok %d most %d\n", total == threads, most);

  // A false if clause runs the region on the host (the task may still be deferred, OpenMP 5.2 13.8): once, and by the
  // taskwait.
  count = 0;
  int initial = -1;
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp target if (0) map(tofrom : count) map(from : initial) nowait
    {
      count += 1;
      initial = omp_is_initial_device();
    }
#pragma omp taskwait
    // CHECK: if false 1 1
    printf("if false %d %d\n", count, initial);
#pragma omp target if (1) map(tofrom : count) map(from : initial) nowait
    {
      count += 10;
      initial = omp_is_initial_device();
    }
#pragma omp taskwait
  }
  // DEVICE: if true 11 0
  // HOST: if true 11 1
  printf("if true %d %d\n", count, initial);

  // The result of a map(from) is there at the end of the taskgroup.
  int r = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp taskgroup
    {
#pragma omp target map(from : r) nowait
      r = 42;
    }
  }
  // CHECK: taskgroup 42
  printf("taskgroup %d\n", r);

  // A target nowait inside an explicit task.
  r = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
  {
#pragma omp task shared(r)
    {
#pragma omp target map(from : r) nowait
      r = 7;
#pragma omp taskwait
    }
#pragma omp taskwait
  }
  // CHECK: in task 7
  printf("in task %d\n", r);

  // Outside any parallel region, in the initial thread (this always worked).
  r = 0;
#pragma omp target map(from : r) nowait
  r = 9;
#pragma omp taskwait
  // CHECK: initial thread 9
  printf("initial thread %d\n", r);
  return 0;
}
