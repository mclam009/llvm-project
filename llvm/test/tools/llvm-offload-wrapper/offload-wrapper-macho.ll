; The wrapper modules for a Mach-O target (macOS), for every kind of offload binary. A Mach-O section name needs a
; segment and a section. The entries are reached through the symbols section$start$__DATA$offload_entries and
; section$end$__DATA$offload_entries, which the linker provides when an object defines the section (the host objects
; always do), so the wrapper makes no dummy entry: the object writer gives a zero-length variable one byte, which
; shifted every real entry. The OpenMP device image is in __DATA,__offloading (Apple's linker rejects a relocation
; against a symbol in a custom segment, and drops the contents of __LLVM), the relocatable one in __DATA,__offload_rel,
; and the SYCL binary in __DATA,__offloading.

; RUN: llvm-offload-wrapper --triple=arm64-apple-macosx -kind=openmp %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=OMP --implicit-check-not=__dummy

;      OMP: @"\01section$start$__DATA$offload_entries" = external hidden constant [0 x %struct.__tgt_offload_entry]
; OMP-NEXT: @"\01section$end$__DATA$offload_entries" = external hidden constant [0 x %struct.__tgt_offload_entry]
; OMP-NEXT: @.omp_offloading.device_image = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"{{.*}}", section "__DATA,__offloading", align 8
; OMP-NEXT: @.omp_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { ptr @.omp_offloading.device_image, ptr getelementptr ([[[SIZE]] x i8], ptr
; OMP-NEXT: @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @"\01section$start$__DATA$offload_entries", ptr @"\01section$end$__DATA$offload_entries" }
; OMP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.omp_offloading.descriptor_reg, ptr null }]

;      OMP: define internal void @.omp_offloading.descriptor_reg() section "__TEXT,__StaticInit" {
; OMP-NEXT: entry:
; OMP-NEXT:   call void @__tgt_register_lib(ptr @.omp_offloading.descriptor)
; OMP-NEXT:   %0 = call i32 @atexit(ptr @.omp_offloading.descriptor_unreg)
; OMP-NEXT:   ret void
; OMP-NEXT: }

;      OMP: define internal void @.omp_offloading.descriptor_unreg() section "__TEXT,__StaticInit" {
; OMP-NEXT: entry:
; OMP-NEXT:   call void @__tgt_unregister_lib(ptr @.omp_offloading.descriptor)
; OMP-NEXT:   ret void
; OMP-NEXT: }

; RUN: llvm-offload-wrapper --triple=arm64-apple-macosx -kind=cuda %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=CUDA --implicit-check-not=__dummy

;      CUDA: @"\01section$start$__DATA$offload_entries" = external hidden constant [0 x %struct.__tgt_offload_entry]
; CUDA-NEXT: @"\01section$end$__DATA$offload_entries" = external hidden constant [0 x %struct.__tgt_offload_entry]
; CUDA-NEXT: @.fatbin_image = internal constant {{.*}}, section "__NV_CUDA,__nv_fatbin"
; CUDA-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1180844977, i32 1, ptr @.fatbin_image, ptr null }, section "__NV_CUDA,__fatbin", no_sanitize_address, no_sanitize_hwaddress, align 8
; CUDA-NEXT: @.cuda.binary_handle = internal global ptr null
; CUDA-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.cuda.fatbin_reg, ptr null }]

;      CUDA: define internal void @.cuda.fatbin_reg() section "__TEXT,__StaticInit" {
;      CUDA: define internal void @.cuda.fatbin_unreg() section "__TEXT,__StaticInit" {

; RUN: llvm-offload-wrapper --triple=arm64-apple-macosx -kind=sycl %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=SYCL

;      SYCL: @.sycl_offloading.binary = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"{{.*}}", section "__DATA,__offloading"
; SYCL-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @sycl.descriptor_reg, ptr null }]
; SYCL-NEXT: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @sycl.descriptor_unreg, ptr null }]

;      SYCL: define internal void @sycl.descriptor_reg() section "__TEXT,__StaticInit" {
;      SYCL: define internal void @sycl.descriptor_unreg() section "__TEXT,__StaticInit" {
