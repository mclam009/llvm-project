// The map types and map-type modifiers with an iterator modifier on the same map clause, in either order (clang
// accepted the iterator modifier only first, and mapped one item for the whole expansion): to, from, alloc with
// updates, release and delete (reference counts and presence of every item), and always, close and present. They need
// separate device memory, so there is no run on the host.
//
// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4

static int *rows[N];
static int seen[N];

static void init(int v) {
  for (int i = 0; i < N; ++i)
    rows[i][0] = v + i;
}
static int sum(void) {
  int s = 0;
  for (int i = 0; i < N; ++i)
    s += rows[i][0];
  return s;
}
static int present(void) {
  int n = 0;
  for (int i = 0; i < N; ++i)
    n += omp_target_is_present(rows[i], omp_get_default_device());
  return n;
}
static void show(const char *what) {
  printf("%s %d %d %d %d\n", what, seen[0], seen[1], seen[2], seen[3]);
}

int main(void) {
  for (int i = 0; i < N; ++i)
    rows[i] = malloc(2 * sizeof(int));

  // to: the device sees the host's values; what it writes does not come back.
  init(10);
#pragma omp target map(iterator(it = 0 : N), to : rows[it][:1]) map(to : rows) map(from : seen)
  for (int k = 0; k < N; ++k) {
    seen[k] = rows[k][0];
    rows[k][0] = 100;
  }
  // CHECK: to 10 11 12 13
  show("to");
  // CHECK: to host 46
  printf("to host %d\n", sum());

  // from: what the device writes comes back; its memory starts out undefined.
  init(10);
#pragma omp target map(iterator(it = 0 : N), from : rows[it][:1]) map(to : rows)
  for (int k = 0; k < N; ++k)
    rows[k][0] = 7 * (k + 1);
  // CHECK: from host 70
  printf("from host %d\n", sum());

  // alloc maps without copying: the host's values reach the device with update to, and come back with update from.
  init(1);
#pragma omp target enter data map(iterator(it = 0 : N), alloc : rows[it][:1]) map(to : rows)
  // CHECK: alloc present 4
  printf("alloc present %d\n", present());
#pragma omp target update to(iterator(it = 0 : N) : rows[it][:1])
#pragma omp target
  for (int k = 0; k < N; ++k)
    rows[k][0] += 1000;
#pragma omp target update from(iterator(it = 0 : N) : rows[it][:1])
  // CHECK: alloc host 4010
  printf("alloc host %d\n", sum());
#pragma omp target exit data map(iterator(it = 0 : N), delete : rows[it][:1]) map(delete : rows)
  // CHECK: delete present 0
  printf("delete present %d\n", present());

  // Reference counts: two enter data (to) need two exit data (release).
#pragma omp target enter data map(iterator(it = 0 : N), to : rows[it][:1]) map(to : rows)
#pragma omp target enter data map(iterator(it = 0 : N), to : rows[it][:1]) map(to : rows)
#pragma omp target exit data map(iterator(it = 0 : N), release : rows[it][:1]) map(release : rows)
  // CHECK: release once 4
  printf("release once %d\n", present());
#pragma omp target exit data map(iterator(it = 0 : N), release : rows[it][:1]) map(release : rows)
  // CHECK: release twice 0
  printf("release twice %d\n", present());

  // always: the data is mapped already, and only a map with always copies again. Both orders of the modifiers.
  init(10);
#pragma omp target enter data map(iterator(it = 0 : N), to : rows[it][:1]) map(to : rows)
  init(50);
#pragma omp target map(iterator(it = 0 : N), to : rows[it][:1]) map(to : rows) map(from : seen)
  for (int k = 0; k < N; ++k)
    seen[k] = rows[k][0];
  // CHECK: plain 10 11 12 13
  show("plain");
#pragma omp target map(always, iterator(it = 0 : N), to : rows[it][:1]) map(to : rows) map(from : seen)
  for (int k = 0; k < N; ++k)
    seen[k] = rows[k][0];
  // CHECK: always first 50 51 52 53
  show("always first");
  init(70);
#pragma omp target map(iterator(it = 0 : N), always, to : rows[it][:1]) map(to : rows) map(from : seen)
  for (int k = 0; k < N; ++k)
    seen[k] = rows[k][0];
  // CHECK: always last 70 71 72 73
  show("always last");
  init(90);
#pragma omp target map(always, close, iterator(it = 0 : N), to : rows[it][:1]) map(to : rows) map(from : seen)
  for (int k = 0; k < N; ++k)
    seen[k] = rows[k][0];
  // CHECK: always close 90 91 92 93
  show("always close");
  init(110);
#pragma omp target map(close, iterator(it = 0 : N), always, to : rows[it][:1]) map(to : rows) map(from : seen)
  for (int k = 0; k < N; ++k)
    seen[k] = rows[k][0];
  // CHECK: close always 110 111 112 113
  show("close always");

  // always from: without it the values stay on the device (the data is mapped by the enter data too); with it they
  // come back at the end of the region.
  init(0);
#pragma omp target map(iterator(it = 0 : N), from : rows[it][:1]) map(to : rows)
  for (int k = 0; k < N; ++k)
    rows[k][0] = 200 + k;
  // CHECK: from plain 6
  printf("from plain %d\n", sum());
#pragma omp target map(always, iterator(it = 0 : N), from : rows[it][:1]) map(to : rows)
  for (int k = 0; k < N; ++k)
    rows[k][0] = 200 + k;
  // CHECK: always from 806
  printf("always from %d\n", sum());
  init(0);
#pragma omp target map(iterator(it = 0 : N), always, from : rows[it][:1]) map(to : rows)
  for (int k = 0; k < N; ++k)
    rows[k][0] = 300 + k;
  // CHECK: from always 1206
  printf("from always %d\n", sum());

  // present: the data is mapped (the device holds 300 to 303 from the last region), so the construct runs, with the
  // modifier on either side of the iterator; the device's copy is not refreshed from the host (no always).
  init(20);
#pragma omp target map(present, iterator(it = 0 : N), tofrom : rows[it][:1]) map(present, to : rows) map(from : seen)
  for (int k = 0; k < N; ++k) {
    seen[k] = rows[k][0];
    rows[k][0] += 1;
  }
  // CHECK: present first 300 301 302 303
  show("present first");
#pragma omp target map(iterator(it = 0 : N), present, tofrom : rows[it][:1]) map(present, to : rows) map(from : seen)
  for (int k = 0; k < N; ++k)
    seen[k] = rows[k][0];
  // CHECK: present last 301 302 303 304
  show("present last");
#pragma omp target exit data map(iterator(it = 0 : N), delete : rows[it][:1]) map(delete : rows)
  // CHECK: done present 0
  printf("done present %d\n", present());
  return 0;
}
