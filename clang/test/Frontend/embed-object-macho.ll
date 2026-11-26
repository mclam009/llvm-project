; REQUIRES: aarch64-registered-target

; An embedded offload object in a Mach-O module lives in the section __LLVM,__offloading: a Mach-O section name needs a
; segment and a section, which the ELF name ".llvm.offloading" is not, and the object extraction for the offload tools
; looks for the same name (OffloadBinary.cpp).
; RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h -x ir %s -o - \
; RUN:    | FileCheck %s

; CHECK: @[[OBJECT_1:.+]] = private constant [0 x i8] zeroinitializer, section "__LLVM,__offloading", align 8, !exclude
; CHECK: @[[OBJECT_2:.+]] = private constant [0 x i8] zeroinitializer, section "__LLVM,__offloading", align 8, !exclude
; CHECK: @llvm.compiler.used = appending global [3 x ptr] [ptr @x, ptr @[[OBJECT_1]], ptr @[[OBJECT_2]]], section "llvm.metadata"

@x = private constant i8 1
@llvm.compiler.used = appending global [1 x ptr] [ptr @x], section "llvm.metadata"

define i32 @foo() {
  ret i32 0
}

; CHECK: !llvm.embedded.objects = !{![[METADATA_1:[0-9]+]], ![[METADATA_2:[0-9]+]]}
; CHECK: ![[METADATA_1]] = !{ptr @[[OBJECT_1]], !"__LLVM,__offloading"}
; CHECK: ![[METADATA_2]] = !{ptr @[[OBJECT_2]], !"__LLVM,__offloading"}

; The object file has the section, and the Mach-O assembler accepts the name.
; RUN: %clang_cc1 -triple arm64-apple-macosx -emit-obj \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h -x ir %s -o %t.o
; RUN: llvm-objdump --section-headers %t.o | FileCheck %s --check-prefix=OBJECT
; OBJECT: __offloading
