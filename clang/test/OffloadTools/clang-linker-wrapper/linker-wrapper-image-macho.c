// UNSUPPORTED: system-windows
// REQUIRES: aarch64-registered-target
// REQUIRES: nvptx-registered-target

// The linker wrapper on a Mach-O host (macOS): it finds the offload binary of the host object in the section
// __LLVM,__offloading (an object file, and the metadata of a bitcode file), and the wrapper module that it makes puts
// the device image in __DATA (Apple's linker rejects a relocation against a symbol in a custom segment, and drops the
// contents of __LLVM), reaches the entries through the symbols section$start$__DATA$offload_entries and
// section$end$__DATA$offload_entries that the linker provides, makes no dummy entry (the object writer gives a
// zero-length variable one byte, which shifted every real entry), and registers the image from a __TEXT,__StaticInit
// function.

// RUN: %clang -cc1 %s -triple arm64-apple-macosx -emit-obj -o %t.dev.o
// RUN: llvm-offload-binary -o %t.out --image=file=%t.dev.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple arm64-apple-macosx -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: %clang -cc1 %s -triple arm64-apple-macosx -emit-llvm-bc -o %t.bc -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=arm64-apple-macosx \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=OPENMP,OPENMP-LOAD \
// RUN:   --implicit-check-not=__dummy
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=arm64-apple-macosx \
// RUN:   --linker-path=/usr/bin/ld %t.bc -o a.out 2>&1 | FileCheck %s --check-prefixes=OPENMP,OPENMP-LOAD \
// RUN:   --implicit-check-not=__dummy
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run -r --host-triple=arm64-apple-macosx \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=OPENMP,OPENMP-REL \
// RUN:   --implicit-check-not=__dummy

//      OPENMP: @"\01section$start$__DATA$offload_entries" = external hidden constant [0 x %struct.__tgt_offload_entry]
// OPENMP-NEXT: @"\01section$end$__DATA$offload_entries" = external hidden constant [0 x %struct.__tgt_offload_entry]

//      OPENMP-LOAD: @.omp_offloading.device_image = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"\10\FF\10\AD{{.*}}", section "__DATA,__offloading", align 8
//      OPENMP-REL: @.omp_offloading.device_image = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"\10\FF\10\AD{{.*}}", section "__DATA,__offload_rel", align 8
// OPENMP-NEXT: @.omp_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { ptr getelementptr ([[[IMG_OFF:[0-9]+]] x
// OPENMP-NEXT: @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @"\01section$start$__DATA$offload_entries", ptr @"\01section$end$__DATA$offload_entries" }
// OPENMP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.omp_offloading.descriptor_reg, ptr null }]

//      OPENMP: define internal void @.omp_offloading.descriptor_reg() section "__TEXT,__StaticInit" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_register_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   %0 = call i32 @atexit(ptr @.omp_offloading.descriptor_unreg)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

//      OPENMP: define internal void @.omp_offloading.descriptor_unreg() section "__TEXT,__StaticInit" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_unregister_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }
