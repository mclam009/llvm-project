// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY %s
// expected-no-diagnostics

// A list item with an iterator modifier in the contexts of C++ and of the
// other constructs and clauses: every one of them evaluates the iterations
// into a heap buffer, passes it with the iterator mapper, and frees it
// exactly once on every path that goes on to run the construct.

// SIMD-ONLY-NOT: {{__kmpc|__tgt|omp_iterator_mapper}}

struct S { int *p; int n; };
S objs[4];
int *rows[8];
int x;

#pragma omp declare mapper(S s) map(tofrom: s.p[:s.n])

// The instantiations of a template, with a dependent element type and a
// dependent iteration count
template <typename T, int N>
void tmpl(T **p) {
#pragma omp target map(iterator(it = 0:N), tofrom: p[it][:2])
  x = 1;
}
void use_tmpl(double **d) {
  tmpl<int, 4>(rows);
  tmpl<double, 6>(d);
}
// CHECK-LABEL: define {{.*}}void @_Z8use_tmplPPd(
// CHECK: call void @_Z4tmplIiLi4EEvPPT_(
// CHECK: call void @_Z4tmplIdLi6EEvPPT_(
// CHECK-LABEL: define {{.*}}void @_Z4tmplIiLi4EEvPPT_(
// ints: 8 bytes of pointer and 4 bytes of section per element are in the records, 4 iterations
// CHECK: call ptr @malloc(i64 192)
// CHECK: store ptr @.omp_iterator_mapper, ptr
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: call void @free(
// CHECK-LABEL: define {{.*}}void @_Z4tmplIdLi6EEvPPT_(
// CHECK: call ptr @malloc(i64 288)
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: call void @free(

// The if clause: the buffer exists only when the condition is true, and is
// freed once, after the launch, also when the launch failed and the host
// fallback ran; with a false condition there is no buffer and no free
void if_clause(int c, int n) {
#pragma omp target if(c) map(iterator(it = 0:n), tofrom: rows[it][:2])
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @_Z9if_clauseii(
// CHECK: omp_if.then:
// CHECK: call ptr @malloc(
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: omp_offload.failed:
// CHECK-NEXT: call void @__omp_offloading_{{.*}}if_clause{{.*}}(
// CHECK: omp_offload.cont:
// CHECK: call void @free(
// CHECK: omp_if.else:
// CHECK-NEXT: call void @__omp_offloading_{{.*}}if_clause{{.*}}(
// CHECK-NOT: @free
// CHECK: omp_if.end:

// A declared mapper of the element type is called by the iterator mapper for
// every iteration (the declared mapper's function is the component's mapper)
void declared_mapper(int n) {
#pragma omp target map(iterator(it = 0:n), tofrom: objs[it])
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @_Z15declared_mapperi(
// CHECK: store ptr [[ITER_MAPPER:@.omp_iterator_mapper[.0-9]*]], ptr
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: define internal void [[ITER_MAPPER]](
// CHECK-NOT: call void @__tgt_push_mapper_component(
// CHECK: call void @.omp_mapper._ZTS1S.default(ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}}, ptr null)
// CHECK: define internal void @.omp_mapper._ZTS1S.default(

// A construct with an iterator inside a lambda body and through a reference
void in_lambda(int *(&r)[8], int n) {
  auto f = [&r, n]() {
#pragma omp target map(iterator(it = 0:n), tofrom: r[it][:2])
    x = 1;
  };
  f();
}
// CHECK-LABEL: define {{.*}}void @"_ZZ9in_lambdaRA8_PiiENK3$_0clEv"(
// CHECK: call ptr @malloc(
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: call void @free(

// The combined constructs that contain a target
void combined(int n) {
#pragma omp target teams map(iterator(it = 0:n), tofrom: rows[it][:2])
  x = 1;
#pragma omp target parallel map(iterator(it = 0:n), tofrom: rows[it][:2])
  x = 2;
#pragma omp target teams distribute parallel for simd map(iterator(it = 0:n), tofrom: rows[it][:2])
  for (int i = 0; i < 4; ++i) x = i;
#pragma omp target simd map(iterator(it = 0:n), tofrom: rows[it][:2])
  for (int i = 0; i < 4; ++i) x = i;
}
// CHECK-LABEL: define {{.*}}void @_Z8combinedi(
// CHECK: call ptr @malloc(
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: call void @free(
// CHECK: call ptr @malloc(
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: call void @free(
// CHECK: call ptr @malloc(
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: call void @free(
// CHECK: call ptr @malloc(
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: call void @free(

// A deferred target task owns the buffer: it is freed in the task, after the
// device call, and not in the function that creates the task
void nowait_depend(int n) {
#pragma omp target map(iterator(it = 0:n), tofrom: rows[it][:2]) nowait depend(out: x)
  x = 1;
#pragma omp target enter data map(iterator(it = 0:n), to: rows[it][:2]) nowait depend(out: x)
#pragma omp target exit data map(iterator(it = 0:n), from: rows[it][:2]) nowait depend(in: x)
}
// CHECK-LABEL: define {{.*}}void @_Z13nowait_dependi(
// CHECK: call ptr @malloc(
// CHECK: call ptr @__kmpc_omp_target_task_alloc(
// CHECK-NOT: call void @free(
// CHECK: call ptr @malloc(
// CHECK: call ptr @__kmpc_omp_target_task_alloc(
// CHECK-NOT: call void @free(
// CHECK: call ptr @malloc(
// CHECK: call ptr @__kmpc_omp_target_task_alloc(
// CHECK-NOT: call void @free(
// CHECK: ret void
