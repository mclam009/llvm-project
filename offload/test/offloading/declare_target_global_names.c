// declare target variables of several types, whose names begin with an underscore or two (and one that begins like the
// runtime's own kernels, __omp_offloading_): their initial values on the device, a change on the device, and
// target update in both directions. The CUDA plugin on macOS loads the image as PTX text, reads the initialized globals
// that it needs from the text, renames the symbols that begin with __omp_offloading_ for CuMetal, which cannot find a
// symbol that begins with two underscores, and looks every global up by its name for target update.
//
// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic | %fcheck-generic
// RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic | %fcheck-generic

#include <stdio.h>

#pragma omp begin declare target
int _g_scalar = 7;
int __g_double = 11;
unsigned long long __g_u64 = 0x0102030405060708ULL;
short __g_arr16[4] = {1, 2, 3, 4};
unsigned char _g_bytes[3] = {250, 251, 252};
int __g_zero[2];
int __omp_offloading_user = 9;
#pragma omp end declare target

static void show(const char *what, const int *out) {
  printf("%s %d %d %d %d %d %d %d %d\n", what, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int out[8];

  // The values that the device image initializes.
#pragma omp target map(from : out)
  {
    out[0] = _g_scalar;
    out[1] = __g_double;
    out[2] = (int)(__g_u64 & 0xff) + (int)(__g_u64 >> 56);
    out[3] = __g_arr16[0] + __g_arr16[3];
    out[4] = _g_bytes[0] + _g_bytes[2];
    out[5] = __g_zero[0] + __g_zero[1];
    out[6] = __omp_offloading_user;
    out[7] = _g_bytes[1];
  }
  // CHECK: initial 7 11 9 5 502 0 9 251
  show("initial", out);

  // Changed on the device, and brought back by target update from.
#pragma omp target
  {
    _g_scalar = 70;
    __g_double = 110;
    __g_u64 = 0x1112131415161718ULL;
    __g_arr16[1] = 20;
    _g_bytes[1] = 99;
    __g_zero[1] = 42;
    __omp_offloading_user = 90;
  }
#pragma omp target update from(_g_scalar, __g_double, __g_u64, __g_arr16, _g_bytes, __g_zero, __omp_offloading_user)
  out[0] = _g_scalar;
  out[1] = __g_double;
  out[2] = (int)(__g_u64 & 0xff) + (int)(__g_u64 >> 56);
  out[3] = __g_arr16[1];
  out[4] = _g_bytes[1];
  out[5] = __g_zero[1];
  out[6] = __omp_offloading_user;
  out[7] = _g_bytes[1];
  // CHECK: from device 70 110 41 20 99 42 90 99
  show("from device", out);

  // Changed on the host, and sent with target update to.
  _g_scalar = 3;
  __g_double = 4;
  __g_u64 = 0x0A0B0C0D0E0F1011ULL;
  __g_arr16[0] = 5;
  __g_arr16[3] = 6;
  _g_bytes[0] = 1;
  _g_bytes[2] = 2;
  __g_zero[0] = 8;
  __omp_offloading_user = 12;
#pragma omp target update to(_g_scalar, __g_double, __g_u64, __g_arr16, _g_bytes, __g_zero, __omp_offloading_user)
#pragma omp target map(from : out)
  {
    out[0] = _g_scalar;
    out[1] = __g_double;
    out[2] = (int)(__g_u64 & 0xff) + (int)(__g_u64 >> 56);
    out[3] = __g_arr16[0] + __g_arr16[3];
    out[4] = _g_bytes[0] + _g_bytes[2];
    out[5] = __g_zero[0] + __g_zero[1];
    out[6] = __omp_offloading_user;
    out[7] = _g_bytes[1];
  }
  // CHECK: to device 3 4 27 11 3 50 12 99
  show("to device", out);
  return 0;
}
