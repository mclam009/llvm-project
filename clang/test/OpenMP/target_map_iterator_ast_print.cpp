// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -include-pch %t -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -ast-print %s | FileCheck --check-prefix=SIMD %s
// expected-no-diagnostics

// The iterator modifier of the map clause is printed, and read back from a
// precompiled header, where it stood among the modifiers of the clause.

#ifndef HEADER
#define HEADER

int a[10], b[10];
int n = 4;

void test() {
#pragma omp target map(iterator(int it = 0:10), tofrom: a[it])
  ;
// CHECK: #pragma omp target map(iterator(int it = 0:10),tofrom: a[it])
// SIMD: #pragma omp target map(iterator(int it = 0:10),tofrom: a[it])

#pragma omp target map(always, iterator(int it = 0:n), close, to: a[it])
  ;
// CHECK: #pragma omp target map(always,iterator(int it = 0:n),close,to: a[it])

#pragma omp target map(close, present, iterator(int it = 0:10:2), from: a[it], b[it])
  ;
// CHECK: #pragma omp target map(close,present,iterator(int it = 0:10:2),from: a[it],b[it])

#pragma omp target data map(iterator(int it = 0:10, int jt = 0:2), tofrom: a[it + jt])
  ;
// CHECK: #pragma omp target data map(iterator(int it = 0:10, int jt = 0:2),tofrom: a[it + jt])

#pragma omp target enter data map(iterator(int it = 0:10), alloc: a[it])
// CHECK: #pragma omp target enter data map(iterator(int it = 0:10),alloc: a[it])
#pragma omp target exit data map(iterator(int it = 0:10), delete: a[it])
// CHECK: #pragma omp target exit data map(iterator(int it = 0:10),delete: a[it])
#pragma omp target update to(iterator(int it = 0:10): a[it]) from(present, iterator(int jt = 0:10): b[jt])
// CHECK: #pragma omp target update to(iterator(int it = 0:10): a[it]) from(present, iterator(int jt = 0:10): b[jt])
}

template <typename T, int N> void tmpl(T *p) {
#pragma omp target map(iterator(int it = 0:N), always, tofrom: p[it])
  ;
// CHECK: template <typename T, int N> void tmpl(T *p) {
// CHECK: #pragma omp target map(iterator(int it = 0:N),always,tofrom: p[it])
// CHECK: template<> void tmpl<int, 4>(int *p) {
// CHECK: #pragma omp target map(iterator(int it = 0:4),always,tofrom: p[it])
}

// The iteration count that is a template parameter is substituted in the instantiation
void use() { tmpl<int, 4>(a); }

#endif
