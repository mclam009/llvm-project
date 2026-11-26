// A reduction across the teams of a kernel: the device runtime counts the teams that have stored their partial result
// with a wrapping atomic increment (atom.inc), and the last team to arrive combines them. CuMetal has no atom.inc, and
// the runtime builds it from a compare-and-swap loop with fences around it (Metal atomics are relaxed), which has to
// count every team exactly once for any number of teams, so that the last one combines all the partial results. The
// results are compared with the serial loops.
//
// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic | %fcheck-generic

#include <stdio.h>

#define M 200000

static int value(int i) { return (i * 7) % 1009 + (i % 3); }

#define PRAGMA(x) _Pragma(#x)
#define SUM_TEAMS(NT)                                                                                                  \
  do {                                                                                                                 \
    long sum = 0;                                                                                                      \
    PRAGMA(omp target teams distribute parallel for num_teams(NT) reduction(+ : sum))                                  \
    for (int i = 0; i < M; ++i)                                                                                        \
      sum += value(i);                                                                                                 \
    printf("sum %s %s\n", #NT, sum == want_sum ? "ok" : "FAIL");                                                       \
  } while (0)

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  long want_sum = 0;
  int want_max = 0, want_min = 1 << 30;
  for (int i = 0; i < M; ++i) {
    want_sum += value(i);
    want_max = value(i) > want_max ? value(i) : want_max;
    want_min = value(i) < want_min ? value(i) : want_min;
  }

  // Different numbers of teams: fewer than the loop's chunks, many more than the cores, and over a thousand.
  // CHECK: sum 1 ok
  SUM_TEAMS(1);
  // CHECK: sum 2 ok
  SUM_TEAMS(2);
  // CHECK: sum 7 ok
  SUM_TEAMS(7);
  // CHECK: sum 64 ok
  SUM_TEAMS(64);
  // CHECK: sum 256 ok
  SUM_TEAMS(256);
  // CHECK: sum 1000 ok
  SUM_TEAMS(1000);
  // CHECK: sum 1500 ok
  SUM_TEAMS(1500);

  // The default number of teams, and the other reduction operators.
  long sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (int i = 0; i < M; ++i)
    sum += value(i);
  // CHECK: sum default ok
  printf("sum default %s\n", sum == want_sum ? "ok" : "FAIL");

  int max = 0;
#pragma omp target teams distribute parallel for num_teams(64) reduction(max : max)
  for (int i = 0; i < M; ++i)
    max = value(i) > max ? value(i) : max;
  // CHECK: max ok
  printf("max %s\n", max == want_max ? "ok" : "FAIL");

  int min = 1 << 30;
#pragma omp target teams distribute parallel for num_teams(64) reduction(min : min)
  for (int i = 0; i < M; ++i)
    min = value(i) < min ? value(i) : min;
  // CHECK: min ok
  printf("min %s\n", min == want_min ? "ok" : "FAIL");

  // Several reductions in one construct, and one construct after the other with the same teams.
  long a = 0, b = 0;
#pragma omp target teams distribute parallel for num_teams(128) reduction(+ : a, b)
  for (int i = 0; i < M; ++i) {
    a += value(i);
    b += 2;
  }
  // CHECK: two reductions ok
  printf("two reductions %s\n", (a == want_sum && b == 2L * M) ? "ok" : "FAIL");

  int bad = 0;
  for (int rep = 0; rep < 20; ++rep) {
    long s = 0;
#pragma omp target teams distribute parallel for num_teams(200) reduction(+ : s)
    for (int i = 0; i < M; ++i)
      s += value(i);
    bad += s != want_sum;
  }
  // CHECK: repeated ok
  printf("repeated %s\n", bad ? "FAIL" : "ok");
  return 0;
}
