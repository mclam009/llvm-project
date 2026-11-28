// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -triple x86_64-unknown-unknown -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -triple x86_64-unknown-unknown -include-pch %t -emit-llvm -x c /dev/null -o - | FileCheck %s

// OpenMP 5.2: a loop whose 'ordered' clause has no argument is a doacross loop
// of as many loops as 'collapse' says when stand-alone 'ordered' constructs
// bind to it, as if the clause were 'ordered(n)'.

// CHECK-LABEL: define {{.*}}void @one(
// CHECK: call void @__kmpc_doacross_init(ptr @{{.+}}, i32 %{{.+}}, i32 1, ptr %{{.+}})
// CHECK: call void @__kmpc_doacross_wait(
// CHECK: call void @__kmpc_doacross_post(
// CHECK: call void @__kmpc_doacross_fini(
// PRINT-LABEL: void one(
// PRINT: #pragma omp for ordered{{$}}
void one(int *a, int n) {
#pragma omp for ordered
  for (int i = 1; i < n; i++) {
#pragma omp ordered doacross(sink : i - 1)
    a[i] += a[i - 1];
#pragma omp ordered doacross(source :)
  }
}

// CHECK-LABEL: define {{.*}}void @two(
// CHECK: call void @__kmpc_doacross_init(ptr @{{.+}}, i32 %{{.+}}, i32 2, ptr %{{.+}})
// CHECK: call void @__kmpc_doacross_wait(
// CHECK: call void @__kmpc_doacross_post(
// CHECK: call void @__kmpc_doacross_fini(
// PRINT-LABEL: void two(
// PRINT: #pragma omp for ordered collapse(2){{$}}
void two(int *a, int n) {
#pragma omp for ordered collapse(2)
  for (int i = 1; i < n; i++)
    for (int j = 1; j < n; j++) {
#pragma omp ordered doacross(sink : omp_cur_iteration - 1)
      a[i * n + j] += a[i * n + j - 1];
#pragma omp ordered doacross(source : omp_cur_iteration)
    }
}

// A block-associated 'ordered' construct keeps an ordered loop.
// CHECK-LABEL: define {{.*}}void @block(
// CHECK-NOT: __kmpc_doacross_init
// CHECK: call void @__kmpc_ordered(
void foo(void);
void block(int n) {
#pragma omp for ordered
  for (int i = 0; i < n; i++) {
#pragma omp ordered
    foo();
  }
}
