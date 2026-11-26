// The offloading entries of a Mach-O host: their names are in __DATA,offload_names and the entries in
// __DATA,offload_entries (Mach-O section names need a segment and a section, and Apple's linker drops the contents of
// the __LLVM segment), and every entry is in llvm.used, as the linker dead-strips an atom that nothing references and
// the entries are only reached through the symbols that mark the start and the end of their section. Entries of an ELF
// host are in llvm_offload_entries and not in llvm.used.
//
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-macosx10.15.0 -fgpu-rdc --offload-new-driver -emit-llvm -o - \
// RUN:   -x hip %s | FileCheck --check-prefix=MACHO %s
// RUN: %clang_cc1 -std=c++11 -triple aarch64-apple-macosx14.0 -fgpu-rdc --offload-new-driver -emit-llvm -o - \
// RUN:   -x cuda %s | FileCheck --check-prefix=MACHO %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fgpu-rdc --offload-new-driver -emit-llvm -o - \
// RUN:   -x hip %s | FileCheck --check-prefix=ELF %s

#include "Inputs/cuda.h"

__global__ void kernel() {}
__device__ int var;

// MACHO-DAG: @.offloading.entry_name{{(\.[0-9]+)?}} = internal unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}", section "__DATA,offload_names"
// MACHO-DAG: @.offloading.entry_name{{(\.[0-9]+)?}} = internal unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}", section "__DATA,offload_names"
// MACHO-DAG: @.offloading.entry.{{.*}}kernel{{.*}} = weak constant %struct.__tgt_offload_entry {{.*}}, section "__DATA,offload_entries"
// MACHO-DAG: @.offloading.entry.var = weak constant %struct.__tgt_offload_entry {{.*}}, section "__DATA,offload_entries"
// MACHO-DAG: @llvm.used = appending global [2 x ptr] [ptr @.offloading.entry.{{.*}}, ptr @.offloading.entry.var], section "llvm.metadata"
// MACHO-NOT: llvm_offload_entries
// MACHO-NOT: .llvm.rodata.offloading
// MACHO-NOT: __LLVM,offload_entries

// ELF-DAG: @.offloading.entry.{{.*}}kernel{{.*}} = weak constant %struct.__tgt_offload_entry {{.*}}, section "llvm_offload_entries"
// ELF-DAG: @.offloading.entry.var = weak constant %struct.__tgt_offload_entry {{.*}}, section "llvm_offload_entries"
// ELF-NOT: @llvm.used = appending global {{.*}}offloading.entry
// ELF-NOT: __DATA,offload
