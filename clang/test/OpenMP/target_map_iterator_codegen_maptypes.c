// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY %s
// expected-no-diagnostics

// The map types of the list items with an iterator modifier. The entry that the
// construct passes to the runtime carries only the TO and FROM bits of the
// list item; every other bit (always 0x4, delete 0x8, close 0x400, present
// 0x1000, attach 0x4000) is a bit of a component that the mapper function
// pushes for each iteration, whatever the position of the iterator modifier
// among the other map-type-modifiers.

// SIMD-ONLY-NOT: {{__kmpc|__tgt|omp_iterator_mapper}}

struct S { int *p; int n; int v[4]; };
struct S arr[8];
int *rows[8];
int mat[4][8];
int x;

// Entries: the captured x (implicit literal), to, from, dyn_ptr
// CHECK-DAG: @.offload_maptypes = private unnamed_addr constant [4 x i64] [i64 800, i64 1, i64 2, i64 288]
// alloc, release and delete have no TO or FROM bit
// CHECK-DAG: @.offload_maptypes.2 = private unnamed_addr constant [1 x i64] zeroinitializer
// CHECK-DAG: @.offload_maptypes.4 = private unnamed_addr constant [1 x i64] zeroinitializer
// CHECK-DAG: @.offload_maptypes.6 = private unnamed_addr constant [1 x i64] zeroinitializer
// tofrom for the modifier tests, with the entry's own bits only
// CHECK-DAG: @.offload_maptypes.9 = private unnamed_addr constant [3 x i64] [i64 800, i64 3, i64 288]
// CHECK-DAG: @.offload_maptypes.12 = private unnamed_addr constant [3 x i64] [i64 800, i64 3, i64 288]
// CHECK-DAG: @.offload_maptypes.15 = private unnamed_addr constant [3 x i64] [i64 800, i64 3, i64 288]
// Two iterators, one entry; negative step, to only
// CHECK-DAG: @.offload_maptypes.18 = private unnamed_addr constant [3 x i64] [i64 800, i64 3, i64 288]
// CHECK-DAG: @.offload_maptypes.21 = private unnamed_addr constant [3 x i64] [i64 800, i64 1, i64 288]
// Two list items, two entries
// CHECK-DAG: @.offload_maptypes.24 = private unnamed_addr constant [4 x i64] [i64 800, i64 3, i64 3, i64 288]
// x (from), local (tofrom), the entry (to): the iterator entries follow the others
// CHECK-DAG: @.offload_maptypes.28 = private unnamed_addr constant [4 x i64] [i64 34, i64 35, i64 1, i64 288]
// target update: x (from, present) and the entry (to); the entry alone (from)
// CHECK-DAG: @.offload_maptypes.31 = private unnamed_addr constant [2 x i64] [i64 4098, i64 1]
// CHECK-DAG: @.offload_maptypes.33 = private unnamed_addr constant [1 x i64] [i64 2]
// target data
// CHECK-DAG: @.offload_maptypes.35 = private unnamed_addr constant [1 x i64] [i64 3]

// to and from in two clauses of one construct: two entries, two mappers
void tofrom_split(int n) {
#pragma omp target map(iterator(it = 0:n), to: rows[it][:2]) map(iterator(jt = 0:n), from: rows[jt][:2])
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @tofrom_split(
// CHECK: store ptr [[TO_MAPPER:@.omp_iterator_mapper]], ptr
// CHECK: store ptr [[FROM_MAPPER:@.omp_iterator_mapper.1]], ptr
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: define internal void [[TO_MAPPER]](
// CHECK: or i64 1, %
// CHECK: or i64 16384, %
// CHECK: define internal void [[FROM_MAPPER]](
// CHECK: or i64 2, %
// CHECK: or i64 16384, %

void alloc_release_delete(int n) {
#pragma omp target enter data map(iterator(it = 0:n), alloc: rows[it][:2])
#pragma omp target exit data map(iterator(it = 0:n), release: rows[it][:2])
#pragma omp target exit data map(iterator(it = 0:n), delete: rows[it][:2])
}
// CHECK-LABEL: define {{.*}}void @alloc_release_delete(
// CHECK: call void @__tgt_target_data_begin_mapper({{.*}}ptr @.offload_maptypes.2, ptr null, ptr %.offload_mappers)
// CHECK: call void @__tgt_target_data_end_mapper({{.*}}ptr @.offload_maptypes.4, ptr null, ptr %.offload_mappers{{.*}})
// CHECK: call void @__tgt_target_data_end_mapper({{.*}}ptr @.offload_maptypes.6, ptr null, ptr %.offload_mappers{{.*}})
// As for a list item without an iterator ([0, 16384], [0] and [8] for the same three directives): alloc has no bit
// besides the attachment's, release and delete leave the attachment out, delete has bit 0x8
// CHECK: define internal void @.omp_iterator_mapper.3(
// CHECK: or i64 0, %
// CHECK: or i64 16384, %
// CHECK: define internal void @.omp_iterator_mapper.5(
// CHECK: or i64 0, %
// CHECK-NOT: or i64 16384, %
// CHECK: define internal void @.omp_iterator_mapper.7(
// CHECK: or i64 8, %
// CHECK-NOT: or i64 16384, %

// The iterator modifier first, last, and between the other modifiers: the same
// component bits (present 0x1000, close 0x400, always 0x4, tofrom 3)
void modifiers_first(int n) {
#pragma omp target map(iterator(it = 0:n), always, close, present, tofrom: rows[it][:2])
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @modifiers_first(
// CHECK: define internal void @.omp_iterator_mapper.10(
// CHECK: or i64 5127, %
// CHECK: or i64 16384, %

void modifiers_last(int n) {
#pragma omp target map(always, close, present, iterator(it = 0:n), tofrom: rows[it][:2])
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @modifiers_last(
// CHECK: define internal void @.omp_iterator_mapper.13(
// CHECK: or i64 5127, %
// CHECK: or i64 16384, %

void modifiers_middle(int n) {
#pragma omp target map(always, iterator(it = 0:n), close, tofrom: rows[it][:2])
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @modifiers_middle(
// CHECK: define internal void @.omp_iterator_mapper.16(
// CHECK: or i64 1031, %
// CHECK: or i64 16384, %

// Two iterators make one list of n * m items; an array of arrays has no attachment
void two_iterators(int n, int m) {
#pragma omp target map(iterator(it = 0:n, jt = 0:m), tofrom: mat[it][jt:1])
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @two_iterators(
// CHECK: mul nuw i64 {{.+}}, {{.+}}
// CHECK: call ptr @malloc(
// CHECK: define internal void @.omp_iterator_mapper.19(
// CHECK: or i64 3, %
// CHECK-NOT: or i64 16384, %

void negative_step(int n) {
#pragma omp target map(iterator(it = n:0:-1), to: rows[it - 1][:1])
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @negative_step(
// CHECK: call ptr @malloc(
// CHECK: define internal void @.omp_iterator_mapper.22(
// CHECK: or i64 1, %
// CHECK: or i64 16384, %

// Members of the elements of an array of structures: the component of the
// structure comes first, and the member is MEMBER_OF it, counted from the
// components that the mapper has pushed so far
void members(int n) {
#pragma omp target map(iterator(it = 0:n), tofrom: arr[it].v[:4], arr[it].n)
  x = 1;
}
// CHECK-LABEL: define {{.*}}void @members(
// CHECK: define internal void @.omp_iterator_mapper.25(
// CHECK: [[NUM:%.+]] = call i64 @__tgt_mapper_num_components(
// CHECK: [[SHIFTED:%.+]] = shl i64 [[NUM]], 48
// CHECK: or i64 0, %
// CHECK: call void @__tgt_push_mapper_component(
// CHECK: add nuw i64 281474976710659, [[SHIFTED]]
// CHECK: call void @__tgt_push_mapper_component(
// CHECK: define internal void @.omp_iterator_mapper.26(
// CHECK: or i64 3, %

// Items without an iterator keep their entries and come first
void mixed(int n) {
  int local[10];
#pragma omp target map(tofrom: local) map(iterator(it = 0:n), to: rows[it][:2]) map(from: x)
  x = local[0];
}
// CHECK-LABEL: define {{.*}}void @mixed(
// CHECK: define internal void @.omp_iterator_mapper.29(
// CHECK: or i64 1, %
// CHECK: or i64 16384, %

// The motion clauses: the iterator modifier before or after the other
// motion-modifier, and the present bit in the component
void update_forms(int n) {
#pragma omp target update to(iterator(it = 0:n): rows[it][:2]) from(present: x)
#pragma omp target update from(present, iterator(it = 0:n): rows[it][:2])
}
// CHECK-LABEL: define {{.*}}void @update_forms(
// CHECK: call void @__tgt_target_data_update_mapper({{.*}}ptr @.offload_maptypes.31, ptr null, ptr %.offload_mappers)
// CHECK: call void @__tgt_target_data_update_mapper({{.*}}ptr @.offload_maptypes.33, ptr null, ptr %.offload_mappers{{.*}})
// CHECK: define internal void @.omp_iterator_mapper.32(
// CHECK: or i64 1, %
// CHECK: define internal void @.omp_iterator_mapper.34(
// CHECK: or i64 4098, %

void data_forms(int n) {
#pragma omp target data map(iterator(it = 0:n), tofrom: rows[it][:2])
  ;
}
// CHECK-LABEL: define {{.*}}void @data_forms(
// CHECK: call void @__tgt_target_data_begin_mapper({{.*}}ptr @.offload_maptypes.35, ptr null, ptr %.offload_mappers)
// CHECK: call void @__tgt_target_data_end_mapper({{.*}}ptr @.offload_maptypes.35, ptr null, ptr %.offload_mappers)
// CHECK: define internal void @.omp_iterator_mapper.36(
// CHECK: or i64 3, %
// CHECK: or i64 16384, %
