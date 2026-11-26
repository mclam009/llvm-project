// A kernel indexes an array section with the indices of the host array: map(tofrom: a[START:LENGTH]) and a[i] for i from
// START. The base pointer that the kernel gets lies START elements before the device copy of the section. A runtime
// that binds every pointer argument of a kernel to the allocation that contains it (CuMetal does) needs that pointer to
// lie inside an allocation, so the offload runtime on macOS reserves the distance in front of the section.
//
// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic | %fcheck-generic

#include <stdio.h>
#include <stdlib.h>

#define N (1 << 22)
#define ROWS 600
#define COLS 500

static int a[N];
static int b[N];
static float m[ROWS][COLS];

static void reset(void) {
  for (int i = 0; i < N; ++i)
    a[i] = i;
}
// How many elements differ from what a section [start, start + len) with 5 added should hold
static int wrong(int start, int len) {
  int bad = 0;
  for (int i = 0; i < N; ++i)
    bad += a[i] != ((i >= start && i < start + len) ? i + 5 : i);
  return bad;
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  static const int starts[] = {1, 7, 4096, 100000, 1 << 20, N - 1000};

  // sections starting from near the beginning to near the end, and an index that does not start the section
  for (unsigned k = 0; k < sizeof(starts) / sizeof(starts[0]); ++k) {
    int start = starts[k], len = 1000;
    reset();
#pragma omp target map(tofrom : a[start : len])
    for (int i = start; i < start + len; ++i)
      a[i] += 5;
    // CHECK: section 1 ok
    // CHECK: section 7 ok
    // CHECK: section 4096 ok
    // CHECK: section 100000 ok
    // CHECK: section 1048576 ok
    // CHECK: section 4193304 ok
    printf("section %d %s\n", start, wrong(start, len) ? "FAIL" : "ok");
  }

  // the same in a parallel loop on the device
  reset();
  int start = 1 << 21, len = 4096;
#pragma omp target teams distribute parallel for map(tofrom : a[start : len])
  for (int i = start; i < start + len; ++i)
    a[i] += 5;
  // CHECK: parallel loop ok
  printf("parallel loop %s\n", wrong(start, len) ? "FAIL" : "ok");

  // sections of two arrays in one construct (two sections of one array would not be addressable by one name), and a
  // section through a pointer
  reset();
  for (int i = 0; i < N; ++i)
    b[i] = 2 * i;
#pragma omp target map(tofrom : a[10 : 10]) map(tofrom : b[2000000 : 10])
  {
    for (int i = 10; i < 20; ++i)
      a[i] += 5;
    for (int i = 2000000; i < 2000010; ++i)
      b[i] += 5;
  }
  int bad = 0;
  for (int i = 0; i < N; ++i) {
    bad += a[i] != ((i >= 10 && i < 20) ? i + 5 : i);
    bad += b[i] != ((i >= 2000000 && i < 2000010) ? 2 * i + 5 : 2 * i);
  }
  // CHECK: two arrays ok
  printf("two arrays %s\n", bad ? "FAIL" : "ok");

  reset();
  int *p = a;
  start = 3000000;
#pragma omp target map(tofrom : p[start : 100])
  for (int i = start; i < start + 100; ++i)
    p[i] += 5;
  // CHECK: pointer section ok
  printf("pointer section %s\n", wrong(start, 100) ? "FAIL" : "ok");

  // rows of a matrix: the section starts at a row, and the kernel indexes with the row number
  for (int i = 0; i < ROWS; ++i)
    for (int j = 0; j < COLS; ++j)
      m[i][j] = i * 1000 + j;
#pragma omp target map(tofrom : m[500 : 3][0 : COLS])
  for (int i = 500; i < 503; ++i)
    for (int j = 0; j < COLS; ++j)
      m[i][j] += 0.5f;
  bad = 0;
  for (int i = 0; i < ROWS; ++i)
    for (int j = 0; j < COLS; ++j) {
      float want = i * 1000 + j + ((i >= 500 && i < 503) ? 0.5f : 0.0f);
      bad += m[i][j] != want;
    }
  // CHECK: matrix rows ok
  printf("matrix rows %s\n", bad ? "FAIL" : "ok");

  // map(to:) and map(from:) of sections: what the kernel reads, and what it writes into the section
  reset();
  start = 1 << 20;
  int sum = 0;
#pragma omp target map(to : a[start : 10]) map(from : sum)
  {
    sum = 0;
    for (int i = start; i < start + 10; ++i)
      sum += a[i];
  }
  // CHECK: to section 10485805
  printf("to section %d\n", sum);
  return 0;
}
