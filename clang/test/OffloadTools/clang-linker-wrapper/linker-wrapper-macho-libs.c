// UNSUPPORTED: system-windows
// REQUIRES: aarch64-registered-target
// REQUIRES: nvptx-registered-target

// The command line of the linker wrapper on a Mach-O host (macOS), as the Darwin driver gives it:
//  - ld64's "-lto_library <path>" is an option with a value, not the library "to_library" (it is passed on),
//  - a library that the search cannot find is not an error: Darwin links against the SDK's stubs (-lSystem, -lc++, ...),
//    which carry no device code and which the search does not know,
//  - libraries are found as lib<name>.dylib, and a dynamic library is not searched for offload binaries.
// On an ELF host a library that is not found is still an error.

// RUN: %clang -cc1 %s -triple arm64-apple-macosx -emit-obj -o %t.dev.o
// RUN: llvm-offload-binary -o %t.out --image=file=%t.dev.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple arm64-apple-macosx -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: yaml2obj %S/Inputs/macho-dylib.yaml -o %t.dir/libfoo.dylib

// RUN: clang-linker-wrapper --dry-run --host-triple=arm64-apple-macosx --linker-path=/usr/bin/ld %t.o \
// RUN:   -lto_library /nonexistent/libLTO.dylib -o a.out 2>&1 | FileCheck %s --check-prefix=LTO
// LTO: "/usr/bin/ld" {{.*}} -lto_library /nonexistent/libLTO.dylib {{.*}}-o a.out
// LTO-NOT: to_library

// RUN: clang-linker-wrapper --dry-run --host-triple=arm64-apple-macosx --linker-path=/usr/bin/ld %t.o \
// RUN:   -lSystem -lc++ -o a.out 2>&1 | FileCheck %s --check-prefix=STUBS
// STUBS: "/usr/bin/ld" {{.*}}-l System -l c++ {{.*}}-o a.out

// RUN: clang-linker-wrapper --dry-run --host-triple=arm64-apple-macosx --linker-path=/usr/bin/ld %t.o \
// RUN:   -L%t.dir -lfoo -lnothere -o a.out 2>&1 | FileCheck %s --check-prefix=DYLIB
// DYLIB: "/usr/bin/ld" {{.*}}-l foo -l nothere {{.*}}-o a.out

// RUN: not clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld %t.o \
// RUN:   -lSystem -o a.out 2>&1 | FileCheck %s --check-prefix=ELF
// ELF: error: unable to find library -lSystem
