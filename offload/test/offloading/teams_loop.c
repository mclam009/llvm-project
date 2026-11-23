// A standalone 'teams loop' is a teams construct with a distribute parallel for; clang 23.1.6 emitted it as a plain
// 'distribute', which every thread of a team ran in full (a target region with 'teams loop' is an SPMD kernel), so an
// in-place body such as c[i] = c[i] * c[i] was applied several times: wrong results on every GPU. A 'target teams
// loop' that cannot be a parallel for (its body calls a function that is not an OpenMP routine) is still a plain
// distribute, and then must neither use the bounds of a distribute parallel for nor the fused reduction schedule (which
// ran one iteration per chunk). Every result is compared with the serial loop; the runs on the host give the same.
//
// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic | %fcheck-generic

#include <omp.h>
#include <stdio.h>

#define N 96
#define ROWS 8
#define COLS 12
#define BIG 5000

static void fill(int *c, int n) {
  for (int i = 0; i < n; ++i)
    c[i] = i % 7 + 2;
}
static void same(const char *what, const int *got, const int *want, int n) {
  int bad = 0;
  for (int i = 0; i < n; ++i)
    bad += got[i] != want[i];
  if (bad)
    printf("%s FAIL %d of %d differ\n", what, bad, n);
  else
    printf("%s ok\n", what);
}
static void same_value(const char *what, long got, long want) {
  if (got != want)
    printf("%s FAIL got %ld want %ld\n", what, got, want);
  else
    printf("%s ok\n", what);
}
// not an OpenMP routine: a call to it keeps a 'target teams loop' from being a parallel for
static int square(int x) { return x * x; }

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int c[N], want[N];

  // The standalone teams loop in a target region, in place.
  fill(c, N);
  for (int i = 0; i < N; ++i)
    want[i] = (i % 7 + 2) * (i % 7 + 2);
#pragma omp target map(tofrom : c)
#pragma omp teams loop
  for (int i = 0; i < N; ++i)
    c[i] = c[i] * c[i];
  // CHECK: standalone ok
  same("standalone", c, want, N);

  // With num_teams and thread_limit.
  fill(c, N);
#pragma omp target map(tofrom : c)
#pragma omp teams loop num_teams(4) thread_limit(8)
  for (int i = 0; i < N; ++i)
    c[i] = c[i] * c[i];
  // CHECK: num_teams ok
  same("num_teams", c, want, N);

  // A reduction.
  fill(c, N);
  long sum = 0, want_sum = 0;
  for (int i = 0; i < N; ++i)
    want_sum += c[i] * c[i];
#pragma omp target map(tofrom : sum) map(to : c)
#pragma omp teams loop reduction(+ : sum)
  for (int i = 0; i < N; ++i)
    sum += c[i] * c[i];
  // CHECK: reduction ok
  same_value("reduction", sum, want_sum);

  // collapse(2): a 2-D loop nest over the same storage.
  fill(c, N);
#pragma omp target map(tofrom : c)
#pragma omp teams loop collapse(2)
  for (int i = 0; i < ROWS; ++i)
    for (int j = 0; j < COLS; ++j)
      c[i * COLS + j] = c[i * COLS + j] * c[i * COLS + j];
  // CHECK: collapse ok
  same("collapse", c, want, N);

  // private and firstprivate (lastprivate is limited to the loop variable on a loop construct).
  fill(c, N);
  int t = -1, base = 5;
  for (int i = 0; i < N; ++i)
    want[i] = (i % 7 + 2) * (i % 7 + 2) + 5;
#pragma omp target map(tofrom : c)
#pragma omp teams loop private(t) firstprivate(base)
  for (int i = 0; i < N; ++i) {
    t = c[i] * c[i];
    c[i] = t + base;
  }
  // CHECK: data sharing ok
  same("data sharing", c, want, N);

  // A loop construct with bind(teams) in a teams region.
  fill(c, N);
  for (int i = 0; i < N; ++i)
    want[i] = (i % 7 + 2) * (i % 7 + 2);
#pragma omp target teams map(tofrom : c)
#pragma omp loop bind(teams)
  for (int i = 0; i < N; ++i)
    c[i] = c[i] * c[i];
  // CHECK: bind teams ok
  same("bind teams", c, want, N);

  // The combined construct, which can be a parallel for.
  fill(c, N);
#pragma omp target teams loop map(tofrom : c)
  for (int i = 0; i < N; ++i)
    c[i] = c[i] * c[i];
  // CHECK: combined ok
  same("combined", c, want, N);

  // The combined construct with a reduction.
  fill(c, N);
  sum = 0;
#pragma omp target teams loop map(tofrom : sum) map(to : c) reduction(+ : sum)
  for (int i = 0; i < N; ++i)
    sum += c[i] * c[i];
  // CHECK: combined reduction ok
  same_value("combined reduction", sum, want_sum);

  // The combined construct with collapse(2).
  fill(c, N);
#pragma omp target teams loop collapse(2) map(tofrom : c)
  for (int i = 0; i < ROWS; ++i)
    for (int j = 0; j < COLS; ++j)
      c[i * COLS + j] = c[i * COLS + j] * c[i * COLS + j];
  // CHECK: combined collapse ok
  same("combined collapse", c, want, N);

  // The combined construct that cannot be a parallel for (the body calls a function): a plain distribute.
  fill(c, N);
#pragma omp target teams loop map(tofrom : c)
  for (int i = 0; i < N; ++i)
    c[i] = square(c[i]);
  // CHECK: distribute ok
  same("distribute", c, want, N);

  // The same with a reduction (the schedule that fuses distribute and the worksharing loop ran one iteration per
  // chunk). The number of teams is given: a reduction over the teams of a kernel in generic mode with the default
  // number of teams is wrong with the CUDA plugin on CuMetal, for 'target teams distribute' as well.
  fill(c, N);
  sum = 0;
#pragma omp target teams loop num_teams(4) map(tofrom : sum) map(to : c) reduction(+ : sum)
  for (int i = 0; i < N; ++i)
    sum += square(c[i]);
  // CHECK: distribute reduction ok
  same_value("distribute reduction", sum, want_sum);

  // With collapse(2) and with several teams.
  fill(c, N);
#pragma omp target teams loop collapse(2) num_teams(3) map(tofrom : c)
  for (int i = 0; i < ROWS; ++i)
    for (int j = 0; j < COLS; ++j)
      c[i * COLS + j] = square(c[i * COLS + j]);
  // CHECK: distribute collapse ok
  same("distribute collapse", c, want, N);

  // A longer loop, several iterations per thread, with and without the call.
  static int big[BIG];
  long big_sum = 0, big_want = 0;
  for (int i = 0; i < BIG; ++i)
    big_want += (long)(i % 7 + 2) * (i % 7 + 2);
  fill(big, BIG);
#pragma omp target teams loop map(tofrom : big_sum) map(to : big) reduction(+ : big_sum)
  for (int i = 0; i < BIG; ++i)
    big_sum += big[i] * big[i];
  // CHECK: long ok
  same_value("long", big_sum, big_want);
  big_sum = 0;
#pragma omp target teams loop num_teams(4) map(tofrom : big_sum) map(to : big) reduction(+ : big_sum)
  for (int i = 0; i < BIG; ++i)
    big_sum += square(big[i]);
  // CHECK: long distribute ok
  same_value("long distribute", big_sum, big_want);

  // Fewer iterations than threads, and none.
  fill(c, N);
  for (int i = 0; i < N; ++i)
    want[i] = (i % 7 + 2) * (i % 7 + 2);
#pragma omp target map(tofrom : c)
#pragma omp teams loop
  for (int i = 0; i < 3; ++i)
    c[i] = c[i] * c[i];
  for (int i = 3; i < N; ++i)
    want[i] = c[i];
  // CHECK: short ok
  same("short", c, want, N);
  int none = 0;
  sum = 5;
#pragma omp target map(tofrom : sum)
#pragma omp teams loop reduction(+ : sum)
  for (int i = 0; i < none; ++i)
    sum += 1;
  // CHECK: empty ok
  same_value("empty", sum, 5);
  return 0;
}
