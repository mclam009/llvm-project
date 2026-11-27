// REQUIRES: darwin, librt_has_emupac
// UNSUPPORTED: ios
// RUN: llvm-objdump --macho --arch=all --private-headers "%osx_builtins" | FileCheck %s
// CHECK-LABEL: (emupac.cpp.o)
// CHECK:       cmd LC_BUILD_VERSION
// CHECK:       platform macos
// CHECK-NOT:   .o)
// CHECK:       minos 14.0
