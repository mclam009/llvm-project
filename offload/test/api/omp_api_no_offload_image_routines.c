// A program without any offload image that links libomptarget has no plugin manager: every device routine that the
// plugin manager served crashed on the null pointer. With no devices the initial device is device 0 and the routines
// take their host paths; a routine that is given a device number that does not exist reports it and stops, as it does
// when there are devices (a message instead of a crash, which the second group of runs checks). The routines are those
// that the change touched: the allocation and mapping routines, the copy and set routines (also the asynchronous and
// rectangular ones), the unique ids of devices, the limit of group-private memory, the interop directives and the dump
// of the mapping tables.
//
// RUN: %clang %openmp_flags %flags %s -o %t -lomptarget
// RUN: %t | %fcheck-generic
// RUN: %not --crash %t is_present 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=is_present
// RUN: %not --crash %t is_accessible 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=is_accessible
// RUN: %not --crash %t memcpy_dst 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=memcpy_dst
// RUN: %not --crash %t memcpy_src 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=memcpy_src
// RUN: %not --crash %t memset 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=memset
// RUN: %not --crash %t associate 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=associate
// RUN: %not --crash %t disassociate 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=disassociate
// RUN: %not --crash %t uid 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=uid
// RUN: %not --crash %t gprivate 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=gprivate
// RUN: %not --crash %t alloc 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=alloc
// RUN: %not --crash %t free 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=free
// RUN: %not --crash %t lock 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=lock
// RUN: %not --crash %t unlock 2>&1 | %fcheck-generic --check-prefix=FATAL -DMODE=unlock

#include <omp.h>
#include <stdio.h>
#include <string.h>

extern void ompx_dump_mapping_tables(void);
extern void *llvm_omp_target_lock_mem(void *, size_t, int);
extern void llvm_omp_target_unlock_mem(void *, int);

static void expect(const char *what, long got, long want) {
  if (got == want)
    printf("%s ok\n", what);
  else
    printf("%s FAIL got %ld want %ld\n", what, got, want);
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int init = omp_get_initial_device();
  char buf[16] = {0}, buf2[16] = {0};

  if (argc > 1) {
    // The device number 1 does not exist: each routine says so and stops.
    const char *mode = argv[1];
    printf("calling %s\n", mode);
    if (!strcmp(mode, "is_present"))
      return omp_target_is_present(buf, 1);
    if (!strcmp(mode, "is_accessible"))
      return omp_target_is_accessible(buf, 8, 1);
    if (!strcmp(mode, "memcpy_dst"))
      return omp_target_memcpy(buf2, buf, 8, 0, 0, 1, init);
    if (!strcmp(mode, "memcpy_src"))
      return omp_target_memcpy(buf2, buf, 8, 0, 0, init, 1);
    if (!strcmp(mode, "memset"))
      return omp_target_memset(buf, 1, 8, 1) != NULL;
    if (!strcmp(mode, "associate"))
      return omp_target_associate_ptr(buf, buf2, 8, 0, 1);
    if (!strcmp(mode, "disassociate"))
      return omp_target_disassociate_ptr(buf, 1);
    if (!strcmp(mode, "uid"))
      return omp_get_uid_from_device(1) != NULL;
    if (!strcmp(mode, "gprivate"))
      return omp_get_gprivate_limit(1, omp_access_cgroup) != 0;
    if (!strcmp(mode, "alloc"))
      return omp_target_alloc(8, 1) != NULL;
    if (!strcmp(mode, "free")) {
      omp_target_free(buf, 1);
      return 0;
    }
    if (!strcmp(mode, "lock"))
      return llvm_omp_target_lock_mem(buf, 8, 1) != NULL;
    if (!strcmp(mode, "unlock")) {
      llvm_omp_target_unlock_mem(buf, 1);
      return 0;
    }
    return 2;
  }

  // CHECK: devices 0 initial 0 device_num 0 is_initial 1 default 0
  printf("devices %d initial %d device_num %d is_initial %d default %d\n", omp_get_num_devices(), init,
         omp_get_device_num(), omp_is_initial_device(), omp_get_default_device());

  // The unique id of the initial device, and the lookup in both directions.
  const char *uid = omp_get_uid_from_device(init);
  // CHECK: uid ok
  expect("uid", uid != NULL && strlen(uid) > 0, 1);
  // CHECK: uid back ok
  expect("uid back", omp_get_device_from_uid(uid), init);
  // CHECK: unknown uid ok
  expect("unknown uid", omp_get_device_from_uid("no such device uid"), omp_invalid_device);
  // CHECK: null uid ok
  expect("null uid", omp_get_device_from_uid(NULL), omp_invalid_device);
  // CHECK: invalid device uid ok
  expect("invalid device uid", omp_get_uid_from_device(omp_invalid_device) == NULL, 1);
  // CHECK: gprivate ok
  expect("gprivate", (long)omp_get_gprivate_limit(init, omp_access_cgroup), 0);

  // Allocation and release on the initial device, by its number and as omp_initial_device.
  void *p = omp_target_alloc(16, init);
  void *q = omp_target_alloc(16, omp_initial_device);
  // CHECK: alloc ok
  expect("alloc", p != NULL && q != NULL, 1);
  omp_target_free(p, init);
  omp_target_free(q, omp_initial_device);

  // Set and copy: host to host, plain, rectangular and asynchronous.
  void *r = omp_target_memset(buf, 0x5a, 8, init);
  // CHECK: memset ok
  expect("memset", r == buf && (unsigned char)buf[7] == 0x5a, 1);
  // CHECK: memcpy ok
  expect("memcpy", omp_target_memcpy(buf2, buf, 8, 0, 0, init, init), 0);
  size_t volume[1] = {8}, offset[1] = {0}, dims[1] = {16};
  // CHECK: memcpy_rect ok
  expect("memcpy_rect", omp_target_memcpy_rect(buf2, buf, 1, 1, volume, offset, offset, dims, dims, init, init), 0);
  // CHECK: memcpy_async ok
  expect("memcpy_async", omp_target_memcpy_async(buf2, buf, 8, 0, 0, init, init, 0, NULL), 0);
  // CHECK: memset_async ok
  expect("memset_async", omp_target_memset_async(buf, 0, 4, init, 0, NULL) == buf, 1);

  // Mapping queries: everything is present on the initial device and nothing can be associated with it.
  // CHECK: associate ok
  expect("associate", omp_target_associate_ptr(buf, buf2, 8, 0, init) != 0, 1);
  // CHECK: disassociate ok
  expect("disassociate", omp_target_disassociate_ptr(buf, init) != 0, 1);
  // CHECK: present ok
  expect("present", omp_target_is_present(buf, init), 1);
  // CHECK: present null ok
  expect("present null", omp_target_is_present(NULL, init), 0);
  // CHECK: accessible ok
  expect("accessible", omp_target_is_accessible(buf, 8, init), 1);
  // CHECK: mapped ptr ok
  expect("mapped ptr", omp_get_mapped_ptr(buf, init) == buf, 1);
  // CHECK: mapped ptr no device ok
  expect("mapped ptr no device", omp_get_mapped_ptr(buf, 1) == NULL, 1);

  // No mapping tables: nothing is printed.
  // CHECK: before dump
  // CHECK-NEXT: after dump
  printf("before dump\n");
  ompx_dump_mapping_tables();
  printf("after dump\n");

  // The interop directives give omp_interop_none, as when the offload is disabled, and use and destroy accept it.
  omp_interop_t obj = omp_interop_none;
#pragma omp interop init(targetsync : obj)
  // CHECK: interop init ok
  expect("interop init", obj == omp_interop_none, 1);
#pragma omp interop use(obj)
#pragma omp interop destroy(obj)
  // CHECK: interop use destroy ok
  expect("interop use destroy", obj == omp_interop_none, 1);
  return 0;
}

// FATAL: calling [[MODE]]
// FATAL: device number '1' out of range, no devices available: the program has registered no offload image
