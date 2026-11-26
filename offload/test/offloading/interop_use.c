// '#pragma omp interop use(obj)' waits for the stream of a targetsync object, and destroy releases it. The CUDA plugin
// did not implement the barrier that use needs, so every use was fatal ("sync_barrier not supported"). A use after work
// was queued on the stream returns after that work has finished.
//
// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic

#include <omp.h>
#include <stdio.h>

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int dev = omp_get_default_device();

  omp_interop_t obj = omp_interop_none;
#pragma omp interop init(targetsync : obj) device(dev)
  // CHECK: init object
  printf("init %s\n", obj != omp_interop_none ? "object" : "none");

  int rc = -1;
  void *stream = omp_get_interop_ptr(obj, omp_ipr_targetsync, &rc);
  // CHECK: stream 1 0
  printf("stream %d %d\n", stream != NULL, rc);

  // use with nothing queued, and again
#pragma omp interop use(obj)
#pragma omp interop use(obj)
  // CHECK: use done
  printf("use done\n");

  // with a target construct in between
  int v = 1;
#pragma omp target map(tofrom : v)
  v += 41;
#pragma omp interop use(obj)
  // CHECK: after target 42
  printf("after target %d\n", v);

#pragma omp interop destroy(obj)
  // CHECK: destroyed
  printf("destroyed\n");

  // a second object on the same device, used with a dependence
  omp_interop_t second = omp_interop_none;
  int dep = 0;
#pragma omp interop init(targetsync : second) device(dev)
  int rc2 = -1;
  void *stream2 = omp_get_interop_ptr(second, omp_ipr_targetsync, &rc2);
#pragma omp interop use(second) depend(in : dep)
#pragma omp interop destroy(second) depend(in : dep)
  // CHECK: second 1 0
  printf("second %d %d\n", stream2 != NULL, rc2);
  return 0;
}
