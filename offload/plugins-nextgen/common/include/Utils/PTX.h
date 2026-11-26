//===- Utils/PTX.h - Helpers for PTX text device images ---------*- C++ -*-===//
//
// macOS port (see docs/bug-reports/llvm-offload-macos-wip.patch in omp-examples).
//
// The CUDA plugin normally loads cubins (ELF). On macOS the "CUDA driver" is CuMetal, which JIT-compiles PTX text to
// Metal and has no ptxas/nvlink that produce cubins, so the device image is the PTX text emitted by the LTO step of
// clang-nvlink-wrapper (--lto-emit-asm). These helpers recognize such an image, read its `.target sm_XX`, and read the
// initialized globals the runtime needs before launch (the `<kernel>_kernel_environment` structures) from the text.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_UTILS_PTX_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_UTILS_PTX_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace utils {
namespace ptx {

/// True if \p Buffer looks like a PTX module: a `.version` directive and a `.target sm_` directive.
inline bool isPTX(llvm::StringRef Buffer) {
  return Buffer.contains("\n.version ") && Buffer.contains("\n.target sm_");
}

/// The numeric `sm_XX` value of the `.target` directive (80 for `.target sm_80`), if there is one.
inline std::optional<unsigned> getTargetSM(llvm::StringRef Buffer) {
  size_t Pos = Buffer.find("\n.target sm_");
  if (Pos == llvm::StringRef::npos)
    return std::nullopt;
  llvm::StringRef Rest = Buffer.drop_front(Pos + sizeof("\n.target sm_") - 1);
  unsigned SM = 0;
  if (Rest.take_while([](char C) { return C >= '0' && C <= '9'; }).getAsInteger(10, SM))
    return std::nullopt;
  return SM;
}

/// Read the initialized global `Name` (`.u64`, `.u32`, `.u16` or `.b8` array or scalar, e.g.
/// `.weak .global .align 8 .u64 NAME[5] = {1, 2, 0, generic(sym), generic(sym2)};`) as little-endian bytes. Pointer
/// entries (`generic(sym)`) are zero: they are device addresses the host copy does not need.
inline std::optional<std::vector<uint8_t>> readGlobal(llvm::StringRef PTX, llvm::StringRef Name) {
  // Find the declaration `<type> Name` followed by `[`, ` =` or `;`.
  size_t Pos = 0;
  while ((Pos = PTX.find(Name, Pos)) != llvm::StringRef::npos) {
    llvm::StringRef After = PTX.drop_front(Pos + Name.size());
    bool NameEnds = !After.empty() && (After[0] == '[' || After[0] == ' ' || After[0] == ';');
    bool NameStarts = Pos > 0 && PTX[Pos - 1] == ' ';
    if (!NameEnds || !NameStarts) {
      Pos += Name.size();
      continue;
    }
    llvm::StringRef Before = PTX.take_front(Pos - 1);
    llvm::StringRef Line = Before.substr(Before.rfind('\n') == llvm::StringRef::npos ? 0 : Before.rfind('\n') + 1);
    size_t ElemSize = 0;
    if (Line.ends_with(".u64") || Line.ends_with(".b64") || Line.ends_with(".s64"))
      ElemSize = 8;
    else if (Line.ends_with(".u32") || Line.ends_with(".b32") || Line.ends_with(".s32") || Line.ends_with(".f32"))
      ElemSize = 4;
    else if (Line.ends_with(".u16") || Line.ends_with(".b16") || Line.ends_with(".s16"))
      ElemSize = 2;
    else if (Line.ends_with(".u8") || Line.ends_with(".b8") || Line.ends_with(".s8"))
      ElemSize = 1;
    if (!ElemSize || !Line.contains(".global")) {
      Pos += Name.size();
      continue;
    }

    // Element count: `[N]` (absent means one element).
    size_t Count = 1;
    if (After[0] == '[') {
      size_t Close = After.find(']');
      if (Close == llvm::StringRef::npos || After.slice(1, Close).getAsInteger(10, Count))
        return std::nullopt;
      After = After.drop_front(Close + 1);
    }

    std::vector<uint8_t> Bytes(Count * ElemSize, 0);
    size_t Eq = After.find('=');
    size_t Semi = After.find(';');
    if (Eq == llvm::StringRef::npos || Eq > Semi)
      return Bytes; // zero-initialized
    llvm::StringRef Init = After.slice(Eq + 1, Semi).trim();
    if (Init.consume_front("{"))
      Init = Init.take_until([](char C) { return C == '}'; });

    size_t Index = 0;
    while (!Init.empty() && Index < Count) {
      size_t Comma = 0;
      int Depth = 0;
      for (; Comma < Init.size(); ++Comma) {
        if (Init[Comma] == '(')
          ++Depth;
        else if (Init[Comma] == ')')
          --Depth;
        else if (Init[Comma] == ',' && Depth == 0)
          break;
      }
      llvm::StringRef Item = Init.take_front(Comma).trim();
      Init = Comma < Init.size() ? Init.drop_front(Comma + 1) : llvm::StringRef();
      uint64_t Value = 0;
      if (!Item.starts_with("generic(") && !Item.starts_with("0f") && !Item.starts_with("0d") &&
          Item.getAsInteger(0, Value))
        return std::nullopt;
      for (size_t B = 0; B < ElemSize; ++B)
        Bytes[Index * ElemSize + B] = static_cast<uint8_t>(Value >> (8 * B));
      ++Index;
    }
    return Bytes;
  }
  return std::nullopt;
}

} // namespace ptx
} // namespace utils

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_UTILS_PTX_H
