// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=x86_64-pc-linux-gnu -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY %s
// expected-no-diagnostics

// A list item with an iterator modifier stands for one list item per
// iteration. It gets one map entry, whose mapper function pushes the
// components of all iterations: their base, begin and size are evaluated
// before the construct into a heap buffer, freed after the runtime call.

// SIMD-ONLY-NOT: {{__kmpc|__tgt|omp_iterator_mapper}}

int *rows[8];

// rows (implicit), the entry of rows[it][:2] (TO | FROM), dyn_ptr
// CHECK-DAG: @.offload_maptypes = private unnamed_addr constant [3 x i64] [i64 547, i64 3, i64 288]
// The entry of rows[it][:1] alone: TO, ALLOC (enter data), FROM (update)
// CHECK-DAG: @.offload_maptypes.1 = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK-DAG: @.offload_maptypes.3 = private unnamed_addr constant [1 x i64] zeroinitializer
// CHECK-DAG: @.offload_maptypes.5 = private unnamed_addr constant [1 x i64] [i64 2]

void map_target(int n) {
#pragma omp target map(iterator(it = 0:n), tofrom: rows[it][:2])
  rows[0][0] = 1;
}

// CHECK-LABEL: define {{.*}}void @map_target(
// CHECK: [[RECORDS:%.+]] = alloca ptr
// CHECK: [[COUNTER:%.+]] = alloca i64
// Two components per iteration (the section and its attachment), 24 bytes each
// CHECK: [[N:%.+]] = select i1 %{{.+}}, i32 0, i32 %{{.+}}
// CHECK: [[N64:%.+]] = sext i32 [[N]] to i64
// CHECK: [[ITERS:%.+]] = mul nuw i64 1, [[N64]]
// CHECK: [[BYTES:%.+]] = mul nuw i64 [[ITERS]], 48
// CHECK: [[BUF:%.+]] = call ptr @malloc(i64 [[BYTES]])
// CHECK: store ptr [[BUF]], ptr [[RECORDS]]
// CHECK: store i64 0, ptr [[COUNTER]]
// CHECK: iter.body:
// CHECK: [[PTR_ADDR:%.+]] = getelementptr inbounds [8 x ptr], ptr @rows, i64 0, i64 %{{.+}}
// CHECK: [[BEGIN:%.+]] = getelementptr inbounds nuw i32, ptr %{{.+}}, i64 0
// CHECK: [[I:%.+]] = load i64, ptr [[COUNTER]]
// CHECK: [[RECS:%.+]] = load ptr, ptr [[RECORDS]]
// CHECK: [[FIRST:%.+]] = mul nuw i64 [[I]], 2
// The section: base rows[it], begin &rows[it][0], 8 bytes
// CHECK: getelementptr inbounds { ptr, ptr, i64 }, ptr [[RECS]], i64 %{{.+}}
// CHECK: store ptr %{{.+}}, ptr %{{.+}}
// CHECK: store ptr [[BEGIN]], ptr %{{.+}}
// CHECK: store i64 8, ptr %{{.+}}
// The attachment: base &rows[it], begin &rows[it][0], the size of a pointer
// CHECK: getelementptr inbounds { ptr, ptr, i64 }, ptr [[RECS]], i64 %{{.+}}
// CHECK: store ptr [[PTR_ADDR]], ptr %{{.+}}
// CHECK: store ptr [[BEGIN]], ptr %{{.+}}
// CHECK: store i64 8, ptr %{{.+}}
// CHECK: [[NEXT:%.+]] = add nuw i64 [[I]], 1
// CHECK: store i64 [[NEXT]], ptr [[COUNTER]]
// CHECK: iter.exit:
// CHECK: [[ENTRY:%.+]] = load ptr, ptr [[RECORDS]]
// CHECK: [[COUNT:%.+]] = load i64, ptr [[COUNTER]]
// CHECK: [[BP1:%.+]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr [[ENTRY]], ptr [[BP1]]
// CHECK: [[P1:%.+]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK: store ptr [[ENTRY]], ptr [[P1]]
// CHECK: [[S1:%.+]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 1
// CHECK: store i64 [[COUNT]], ptr [[S1]]
// CHECK: [[M1:%.+]] = getelementptr inbounds [3 x ptr], ptr %.offload_mappers, i64 0, i64 1
// CHECK: store ptr [[MAPPER:@.omp_iterator_mapper]], ptr [[M1]]
// CHECK: call i32 @__tgt_target_kernel(
// CHECK: omp_offload.cont:
// CHECK: [[FREE_ADDR:%.+]] = getelementptr ptr, ptr %{{.+}}, i64 1
// CHECK: [[FREE:%.+]] = load ptr, ptr [[FREE_ADDR]]
// CHECK: call void @free(ptr [[FREE]])

// CHECK: define internal void [[MAPPER]](ptr [[HANDLE:%.+]], ptr %{{.+}}, ptr [[BUFFER:%.+]], i64 [[ITERATIONS:%.+]], i64 [[TYPE:%.+]], ptr %{{.+}})
// The TO and FROM bits of the map type are kept, and its MEMBER_OF field.
// CHECK: [[TOFROM:%.+]] = and i64 [[TYPE]], 3
// CHECK: [[KEEP:%.+]] = or i64 [[TOFROM]], -4
// CHECK: [[MEMBEROF:%.+]] = and i64 [[TYPE]], -281474976710656
// CHECK: icmp eq i64 [[ITERATIONS]], 0
// CHECK: [[K:%.+]] = phi i64
// CHECK: call i64 @__tgt_mapper_num_components(ptr [[HANDLE]])
// CHECK: mul nuw i64 [[K]], 2
// CHECK: getelementptr inbounds { ptr, ptr, i64 }, ptr [[BUFFER]], i64 %{{.+}}
// CHECK: [[T0:%.+]] = or i64 3, [[MEMBEROF]]
// CHECK: [[T0K:%.+]] = and i64 [[T0]], [[KEEP]]
// CHECK: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr %{{.+}}, ptr %{{.+}}, i64 %{{.+}}, i64 [[T0K]], ptr null)
// CHECK: [[T1:%.+]] = or i64 16384, [[MEMBEROF]]
// CHECK: [[T1K:%.+]] = and i64 [[T1]], [[KEEP]]
// CHECK: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr %{{.+}}, ptr %{{.+}}, i64 %{{.+}}, i64 [[T1K]], ptr null)
// CHECK: [[KNEXT:%.+]] = add nuw i64 [[K]], 1
// CHECK: icmp eq i64 [[KNEXT]], [[ITERATIONS]]
// CHECK: ret void

void map_target_data(int c) {
#pragma omp target data map(iterator(it = 0:8), to: rows[it][:1]) if(c)
  ;
}

// The buffer exists only if the if clause was true.
// CHECK-LABEL: define {{.*}}void @map_target_data(
// CHECK: br i1 [[C:%.+]], label %omp_if.then, label %omp_if.else
// CHECK: omp_if.then:
// CHECK: call ptr @malloc(i64 384)
// CHECK: store ptr @.omp_iterator_mapper.2, ptr
// CHECK: call void @__tgt_target_data_begin_mapper({{.*}}, ptr @.offload_maptypes.1, ptr null, ptr %.offload_mappers)
// CHECK: call void @__tgt_target_data_end_mapper({{.*}}, ptr @.offload_maptypes.1, ptr null, ptr %.offload_mappers)
// CHECK: br i1 [[C]], label %omp.iter.map.free, label %omp.iter.map.cont
// CHECK: omp.iter.map.free:
// CHECK: [[DATA_ADDR:%.+]] = getelementptr ptr, ptr %.offload_baseptrs, i64 0
// CHECK: [[DATA_BUF:%.+]] = load ptr, ptr [[DATA_ADDR]]
// CHECK: call void @free(ptr [[DATA_BUF]])
// CHECK: omp.iter.map.cont:

void map_enter_update_nowait(void) {
#pragma omp target enter data map(iterator(it = 0:8:2), alloc: rows[it][:1])
#pragma omp target update from(iterator(it = 0:8): rows[it][:1]) nowait
}

// Four iterations of two components; then eight of one (no attachment in a
// motion clause), freed in the target task.
// CHECK-LABEL: define {{.*}}void @map_enter_update_nowait(
// CHECK: call ptr @malloc(i64 192)
// CHECK: call void @__tgt_target_data_begin_mapper({{.*}}, ptr @.offload_maptypes.3, ptr null, ptr %.offload_mappers)
// CHECK: call void @free(
// CHECK: call ptr @malloc(i64 192)
// CHECK: call ptr @__kmpc_omp_target_task_alloc(
// CHECK-NOT: call void @free(
// CHECK: ret void

// CHECK: define internal i32 @.omp_task_entry.(
// CHECK: call void @__tgt_target_data_update_nowait_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[TASK_BPS:%.+]], ptr %{{.+}}, ptr %{{.+}}, ptr @.offload_maptypes.5, ptr null, ptr %{{.+}}, i32 0, ptr null, i32 0, ptr null)
// CHECK-NEXT: [[TASK_BUF:%.+]] = load ptr, ptr [[TASK_BPS]]
// CHECK-NEXT: call void @free(ptr [[TASK_BUF]])
