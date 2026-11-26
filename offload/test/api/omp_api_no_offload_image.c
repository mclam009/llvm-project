// A program without any offload image that calls the device routines of
// libomptarget: the plugin manager is never created, so there are no devices
// and every routine acts on the initial device (these calls used to crash).
//
// RUN: %clang %openmp_flags %flags %s -o %t -lomptarget
// RUN: %t | %fcheck-generic

#include <omp.h>
#include <stdio.h>

int main(void) {
  int Initial = omp_get_initial_device();
  // CHECK: devices 0 initial 0 device_num 0
  printf("devices %d initial %d device_num %d\n", omp_get_num_devices(),
         Initial, omp_get_device_num());

  int Host[4] = {1, 2, 3, 4};
  int *P = (int *)omp_target_alloc(sizeof(Host), Initial);
  int Rc = omp_target_memcpy(P, Host, sizeof(Host), 0, 0, Initial, Initial);
  // CHECK: alloc 1 memcpy 0 value 4
  printf("alloc %d memcpy %d value %d\n", P != NULL, Rc, P ? P[3] : -1);
  omp_target_free(P, Initial);

  // CHECK: present 1 accessible 1 mapped 1
  printf("present %d accessible %d mapped %d\n",
         omp_target_is_present(Host, Initial),
         omp_target_is_accessible(Host, sizeof(Host), Initial),
         omp_get_mapped_ptr(Host, Initial) == Host);

  omp_interop_t Obj = omp_interop_none;
#pragma omp interop init(targetsync : Obj)
  // CHECK: interop none
  printf("interop %s\n", Obj == omp_interop_none ? "none" : "object");
  return 0;
}
