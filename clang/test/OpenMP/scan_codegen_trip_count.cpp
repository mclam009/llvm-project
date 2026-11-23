// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-unknown -emit-llvm -O0 %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-unknown -emit-llvm -O0 %s -o - | FileCheck --check-prefix SIMD-ONLY %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -triple x86_64-unknown-unknown -emit-llvm -O0 %s -o - | FileCheck %s
// expected-no-diagnostics

// A worksharing loop with an inscan reduction combines the partial results
// of its iterations in ceil(log2(n)) rounds, and copies the last element of
// its buffer to the original variable. For at most one iteration there is
// nothing to combine (a loop that tests its exit at the end would never stop),
// and for none there is no last element to copy: both are guarded at run time
// when the trip count is not a constant, and fold away when it is.

// SIMD-ONLY-NOT: {{__kmpc|omp.outer.log.scan}}

void use(long);

// The trip count is a variable: the copy of the last element is skipped for
// no iterations, and the combine loop is skipped for at most one.
void inclusive_parallel_for(long n, long *in, long *out) {
  long x = 0;
#pragma omp parallel for reduction(inscan, +: x)
  for (long i = 0; i < n; ++i) {
    x += in[i];
#pragma omp scan inclusive(x)
    out[i] = x;
  }
  use(x);
}
// CHECK-LABEL: define {{.*}}void @_Z22inclusive_parallel_forlPlS_(
// CHECK: [[NONEMPTY:%.+]] = icmp ne i64 %{{.+}}, 0
// CHECK-NEXT: br i1 [[NONEMPTY]], label %[[COPY:omp.scan.finals.copy]], label %[[DONE:omp.scan.finals.done]]
// CHECK: [[COPY]]:
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK: call void @_Z3usel(
// CHECK-LABEL: define internal void @_Z22inclusive_parallel_forlPlS_.omp_outlined(
// CHECK: [[MORE_THAN_ONE:%.+]] = icmp ugt i64 %{{.+}}, 1
// CHECK-NEXT: br i1 [[MORE_THAN_ONE]], label %omp.outer.log.scan.body, label %omp.outer.log.scan.exit
// CHECK: omp.outer.log.scan.body:
// CHECK: omp.inner.log.scan.body:
// CHECK: omp.outer.log.scan.exit:

void exclusive_parallel_for(long n, long *in, long *out) {
  long y = 0;
#pragma omp parallel for reduction(inscan, +: y)
  for (long i = 0; i < n; ++i) {
    out[i] = y;
#pragma omp scan exclusive(y)
    y += in[i];
  }
  use(y);
}
// CHECK-LABEL: define {{.*}}void @_Z22exclusive_parallel_forlPlS_(
// CHECK: [[NONEMPTY:%.+]] = icmp ne i64 %{{.+}}, 0
// CHECK-NEXT: br i1 [[NONEMPTY]], label %[[COPY:omp.scan.finals.copy]], label %[[DONE:omp.scan.finals.done]]
// CHECK: [[COPY]]:
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK-LABEL: define internal void @_Z22exclusive_parallel_forlPlS_.omp_outlined(
// CHECK: [[MORE_THAN_ONE:%.+]] = icmp ugt i64 %{{.+}}, 1
// CHECK-NEXT: br i1 [[MORE_THAN_ONE]], label %omp.outer.log.scan.body, label %omp.outer.log.scan.exit

void inclusive_parallel_for_simd(int n, int *in, int *out) {
  int z = 0;
#pragma omp parallel for simd reduction(inscan, +: z)
  for (int i = 0; i < n; ++i) {
    z += in[i];
#pragma omp scan inclusive(z)
    out[i] = z;
  }
  use(z);
}
// CHECK-LABEL: define {{.*}}void @_Z27inclusive_parallel_for_simdiPiS_(
// CHECK: [[NONEMPTY:%.+]] = icmp ne i64 %{{.+}}, 0
// CHECK-NEXT: br i1 [[NONEMPTY]], label %omp.scan.finals.copy, label %omp.scan.finals.done
// CHECK-LABEL: define internal void @_Z27inclusive_parallel_for_simdiPiS_.omp_outlined(
// CHECK: [[MORE_THAN_ONE:%.+]] = icmp ugt i64 %{{.+}}, 1
// CHECK-NEXT: br i1 [[MORE_THAN_ONE]], label %omp.outer.log.scan.body, label %omp.outer.log.scan.exit

// A constant trip count folds the guards: no iterations, one iteration (only
// the copy is kept) and several (as before)
static long cx;
long cin[4], cout[4];
void orphaned_no_iterations() {
#pragma omp for reduction(inscan, +: cx)
  for (int i = 0; i < 0; ++i) {
    cx += cin[i];
#pragma omp scan inclusive(cx)
    cout[i] = cx;
  }
}
// CHECK-LABEL: define {{.*}}void @_Z22orphaned_no_iterationsv(
// CHECK: br i1 false, label %omp.scan.finals.copy, label %omp.scan.finals.done

void orphaned_one_iteration() {
#pragma omp for reduction(inscan, +: cx)
  for (int i = 0; i < 1; ++i) {
    cx += cin[i];
#pragma omp scan inclusive(cx)
    cout[i] = cx;
  }
}
// CHECK-LABEL: define {{.*}}void @_Z22orphaned_one_iterationv(
// CHECK: br i1 false, label %omp.outer.log.scan.body, label %omp.outer.log.scan.exit
// CHECK: br i1 true, label %omp.scan.finals.copy, label %omp.scan.finals.done

void orphaned_four_iterations() {
#pragma omp for reduction(inscan, +: cx)
  for (int i = 0; i < 4; ++i) {
    cx += cin[i];
#pragma omp scan inclusive(cx)
    cout[i] = cx;
  }
}
// CHECK-LABEL: define {{.*}}void @_Z24orphaned_four_iterationsv(
// CHECK: br i1 true, label %omp.outer.log.scan.body, label %omp.outer.log.scan.exit
// CHECK: br i1 true, label %omp.scan.finals.copy, label %omp.scan.finals.done

void orphaned_for_simd_no_iterations() {
#pragma omp for simd reduction(inscan, +: cx)
  for (int i = 0; i < 0; ++i) {
    cx += cin[i];
#pragma omp scan inclusive(cx)
    cout[i] = cx;
  }
}
// CHECK-LABEL: define {{.*}}void @_Z31orphaned_for_simd_no_iterationsv(
// CHECK: br i1 false, label %omp.scan.finals.copy, label %omp.scan.finals.done
