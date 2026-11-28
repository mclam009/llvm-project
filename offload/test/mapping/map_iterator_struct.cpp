// The iterator modifier with members of structures reached through pointers, with a section whose length depends on
// the iteration, in a function template, next to a plain map of the same base, and with list items whose type has a
// declare mapper (the default one, and a named one given by the mapper modifier before or after the iterator: the
// iterator's map function calls the mapper of every element).
//
// RUN: %libomptarget-compilexx-generic -fopenmp-version=51
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compilexx-generic -fopenmp-version=52
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compilexx-generic -fopenmp-version=60
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic

#include <cstdio>
#include <cstdlib>

struct S {
  int n;
  int *p;
};
#pragma omp declare mapper(S s) map(tofrom : s.p[ : s.n])

struct T {
  int n;
  int *q;
};
#pragma omp declare mapper(tm : T t) map(tofrom : t.q[ : t.n])

struct V {
  int n;
  int *p;
};

template <int M> int sum_rows(int *(&r)[M]) {
#pragma omp target map(iterator(i = 0 : M), tofrom : r[i][ : 1])
  for (int k = 0; k < M; ++k)
    r[k][0] *= 3;
  int s = 0;
  for (int k = 0; k < M; ++k)
    s += r[k][0];
  return s;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  // Members of structures through pointers: the section of every element.
  struct M {
    int n;
    int *p;
  } m[4];
  for (int i = 0; i < 4; ++i) {
    m[i].n = 2;
    m[i].p = (int *)std::malloc(2 * sizeof(int));
    m[i].p[0] = i;
    m[i].p[1] = 10 * i;
  }
#pragma omp target map(iterator(i = 0 : 4), tofrom : m[i].p[ : 2])
  for (int k = 0; k < 4; ++k) {
    m[k].p[0] += 1;
    m[k].p[1] += 1;
  }
  int t = 0;
  for (int i = 0; i < 4; ++i)
    t += m[i].p[0] + m[i].p[1];
  // CHECK: struct members 74
  std::printf("struct members %d\n", t);

  // The length of the section depends on the iteration: element k maps k + 1 of its four integers.
  V v[4];
  for (int i = 0; i < 4; ++i) {
    v[i].n = i + 1;
    v[i].p = (int *)std::malloc(4 * sizeof(int));
    for (int j = 0; j < 4; ++j)
      v[i].p[j] = 1;
  }
#pragma omp target map(iterator(i = 0 : 4), tofrom : v[i].p[ : v[i].n])
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < v[k].n; ++j)
      v[k].p[j] += 100;
  t = 0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      t += v[i].p[j];
  // CHECK: varying length 1016
  std::printf("varying length %d\n", t);

  // A function template.
  int *r[5];
  for (int i = 0; i < 5; ++i) {
    r[i] = (int *)std::malloc(sizeof(int));
    *r[i] = i;
  }
  // CHECK: template 30
  std::printf("template %d\n", sum_rows<5>(r));

  // An iterator map next to a plain one on the same base.
  for (int i = 0; i < 5; ++i)
    *r[i] = 1;
#pragma omp target map(tofrom : r[0][ : 1]) map(iterator(i = 1 : 5), tofrom : r[i][ : 1])
  for (int k = 0; k < 5; ++k)
    r[k][0] += k;
  t = 0;
  for (int i = 0; i < 5; ++i)
    t += *r[i];
  // CHECK: mixed clauses 15
  std::printf("mixed clauses %d\n", t);

  // The default mapper of S maps the data of every element's pointer.
  S a[4];
  for (int i = 0; i < 4; ++i) {
    a[i].n = 2;
    a[i].p = (int *)std::malloc(2 * sizeof(int));
    a[i].p[0] = i;
    a[i].p[1] = i;
  }
#pragma omp target map(iterator(i = 0 : 4), tofrom : a[i])
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < a[k].n; ++j)
      a[k].p[j] += 10 * k + j;
  t = 0;
  for (int i = 0; i < 4; ++i)
    t += a[i].p[0] + a[i].p[1];
  // CHECK: default mapper 136
  std::printf("default mapper %d\n", t);

  // A named mapper, the modifier before the iterator and after it.
  T b[4];
  for (int i = 0; i < 4; ++i) {
    b[i].n = 2;
    b[i].q = (int *)std::malloc(2 * sizeof(int));
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 2; ++j)
      b[i].q[j] = i + j + 1;
#pragma omp target map(mapper(tm), iterator(i = 0 : 4), tofrom : b[i])
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < b[k].n; ++j)
      b[k].q[j] *= 2;
  t = 0;
  for (int i = 0; i < 4; ++i)
    t += b[i].q[0] + b[i].q[1];
  // CHECK: mapper first 48
  std::printf("mapper first %d\n", t);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 2; ++j)
      b[i].q[j] = i + j + 1;
#pragma omp target map(iterator(i = 0 : 4), mapper(tm), tofrom : b[i])
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < b[k].n; ++j)
      b[k].q[j] *= 2;
  t = 0;
  for (int i = 0; i < 4; ++i)
    t += b[i].q[0] + b[i].q[1];
  // CHECK: mapper last 48
  std::printf("mapper last %d\n", t);
  return 0;
}
