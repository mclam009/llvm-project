//===--- cuda/dynamic_cuda/cuda.pp ------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement subset of cuda api by calling into cuda library via dlopen
// Does the dlopen/dlsym calls as part of the call to cuInit
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"

#include "Shared/Debug.h"

#include "DLWrap.h"
#include "cuda.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>

#ifdef __APPLE__
#include <dlfcn.h>
#endif

using namespace llvm::offload::debug;

DLWRAP_INITIALIZE()

DLWRAP_INTERNAL(cuInit, 1)

DLWRAP(cuCtxGetDevice, 1)
DLWRAP(cuDeviceGet, 2)
DLWRAP(cuDeviceGetAttribute, 3)
DLWRAP(cuDeviceGetCount, 1)
DLWRAP(cuFuncGetAttribute, 3)
DLWRAP(cuFuncSetAttribute, 3)

// Device info
DLWRAP(cuDeviceGetName, 3)
DLWRAP(cuDeviceGetUuid, 2)
DLWRAP(cuDeviceTotalMem, 2)
DLWRAP(cuDriverGetVersion, 1)

DLWRAP(cuGetErrorString, 2)
DLWRAP(cuLaunchKernelEx, 4)
DLWRAP(cuLaunchHostFunc, 3)

DLWRAP(cuMemAlloc, 2)
DLWRAP(cuMemAllocHost, 2)
DLWRAP(cuMemAllocManaged, 3)
DLWRAP(cuMemAllocAsync, 3)

DLWRAP(cuMemcpyDtoDAsync, 4)
DLWRAP(cuMemcpyDtoH, 3)
DLWRAP(cuMemcpyDtoHAsync, 4)
DLWRAP(cuMemcpyHtoD, 3)
DLWRAP(cuMemcpyHtoDAsync, 4)

DLWRAP(cuMemsetD8Async, 4)
DLWRAP(cuMemsetD16Async, 4)
DLWRAP(cuMemsetD32Async, 4)
DLWRAP(cuMemsetD2D8Async, 6)
DLWRAP(cuMemsetD2D16Async, 6)
DLWRAP(cuMemsetD2D32Async, 6)

DLWRAP(cuMemFree, 1)
DLWRAP(cuMemFreeHost, 1)
DLWRAP(cuMemFreeAsync, 2)

DLWRAP(cuMemPrefetchAsync, 4)
DLWRAP(cuPointerGetAttribute, 3)

DLWRAP(cuModuleGetFunction, 3)
DLWRAP(cuModuleGetGlobal, 4)

DLWRAP(cuModuleUnload, 1)
DLWRAP(cuStreamCreate, 2)
DLWRAP(cuStreamDestroy, 1)
DLWRAP(cuStreamSynchronize, 1)
DLWRAP(cuStreamQuery, 1)
DLWRAP(cuStreamAddCallback, 4)
DLWRAP(cuCtxSetCurrent, 1)
DLWRAP(cuDevicePrimaryCtxRelease, 1)
DLWRAP(cuDevicePrimaryCtxGetState, 3)
DLWRAP(cuDevicePrimaryCtxSetFlags, 2)
DLWRAP(cuDevicePrimaryCtxRetain, 2)
DLWRAP(cuModuleLoadDataEx, 5)
DLWRAP(cuOccupancyMaxPotentialBlockSize, 6)
DLWRAP(cuOccupancyMaxActiveBlocksPerMultiprocessor, 4)

DLWRAP(cuDeviceCanAccessPeer, 3)
DLWRAP(cuCtxEnablePeerAccess, 2)
DLWRAP(cuMemcpyPeerAsync, 6)

DLWRAP(cuCtxGetLimit, 2)
DLWRAP(cuCtxSetLimit, 2)

DLWRAP(cuEventCreate, 2)
DLWRAP(cuEventRecord, 2)
DLWRAP(cuEventQuery, 1)
DLWRAP(cuStreamWaitEvent, 3)
DLWRAP(cuEventSynchronize, 1)
DLWRAP(cuEventElapsedTime, 3)
DLWRAP(cuEventDestroy, 1)

DLWRAP(cuMemUnmap, 2)
DLWRAP(cuMemRelease, 1)
DLWRAP(cuMemAddressFree, 2)
DLWRAP(cuMemGetInfo, 2)
DLWRAP(cuMemAddressReserve, 5)
DLWRAP(cuMemMap, 5)
DLWRAP(cuMemCreate, 4)
DLWRAP(cuMemSetAccess, 4)
DLWRAP(cuMemGetAllocationGranularity, 3)

DLWRAP_FINALIZE()

#ifndef DYNAMIC_CUDA_PATH
#ifdef __APPLE__
// macOS port: the "CUDA driver" is CuMetal (https://github.com/Lulzx/cuda-metal), which runs the PTX on the Metal GPU
#define DYNAMIC_CUDA_PATH "libcumetal.dylib"
#else
#define DYNAMIC_CUDA_PATH "libcuda.so"
#endif
#endif

#ifdef __APPLE__
// Driver functions this plugin binds but CuMetal does not export. The plugin reads the stack/heap size limits at
// device initialization (and sets them only when the user asks), so those two get harmless stubs; stream-ordered
// allocation and managed-memory prefetch report CUDA_ERROR_NOT_SUPPORTED, which the plugin handles. Every other
// missing function still makes the library unusable.
static CUresult unsupportedDriverFunction() { return CUDA_ERROR_NOT_SUPPORTED; }
static CUresult stubCtxGetLimit(size_t *Value, int) {
  if (Value)
    *Value = 8u << 20; // the CUDA default heap size; the stack size query only needs some value
  return CUDA_SUCCESS;
}
static CUresult stubCtxSetLimit(int, size_t) { return CUDA_SUCCESS; }

static void *optionalDriverFunctionStub(const char *Sym) {
  std::string Name(Sym);
  if (Name == "cuCtxGetLimit")
    return reinterpret_cast<void *>(&stubCtxGetLimit);
  if (Name == "cuCtxSetLimit")
    return reinterpret_cast<void *>(&stubCtxSetLimit);
  if (Name == "cuMemAllocAsync" || Name == "cuMemFreeAsync" || Name == "cuMemPrefetchAsync")
    return reinterpret_cast<void *>(&unsupportedDriverFunction);
  return nullptr;
}
#endif

#ifndef TARGET_NAME
#define TARGET_NAME CUDA
#endif
#ifndef DEBUG_PREFIX
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"
#endif

static bool checkForCUDA() {
  // return true if dlopen succeeded and all functions found

  // Prefer _v2 versions of functions if found in the library
  std::unordered_map<std::string, const char *> TryFirst = {
      {"cuMemAlloc", "cuMemAlloc_v2"},
      {"cuMemFree", "cuMemFree_v2"},
      {"cuMemcpyDtoH", "cuMemcpyDtoH_v2"},
      {"cuMemcpyHtoD", "cuMemcpyHtoD_v2"},
      {"cuStreamDestroy", "cuStreamDestroy_v2"},
      {"cuModuleGetGlobal", "cuModuleGetGlobal_v2"},
      {"cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"},
      {"cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2"},
      {"cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"},
      {"cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2"},
      {"cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags_v2"},
  };

  const char *CudaLib = DYNAMIC_CUDA_PATH;
  std::string ErrMsg;
#ifdef __APPLE__
  // dlopen() does not search Homebrew's lib directory: honor CUMETAL_LIBRARY, then a libcumetal.dylib installed next to
  // this runtime (a self-contained toolchain prefix), then the Homebrew prefix, then the default search path
  std::string CumetalPath;
  if (const char *Env = std::getenv("CUMETAL_LIBRARY")) {
    CumetalPath = Env;
  } else {
    Dl_info Info;
    if (dladdr(reinterpret_cast<void *>(&checkForCUDA), &Info) && Info.dli_fname) {
      std::string Sibling(Info.dli_fname);
      Sibling.erase(Sibling.find_last_of('/') + 1);
      Sibling += "libcumetal.dylib";
      if (llvm::sys::fs::exists(Sibling))
        CumetalPath = Sibling;
    }
    if (CumetalPath.empty() && llvm::sys::fs::exists("/opt/homebrew/opt/cumetal/lib/libcumetal.dylib"))
      CumetalPath = "/opt/homebrew/opt/cumetal/lib/libcumetal.dylib";
  }
  if (!CumetalPath.empty())
    CudaLib = CumetalPath.c_str();
  // OpenMP kernels follow device pointers stored in memory (the reduction buffer inside the kernel launch
  // environment, mapped structs), which CuMetal's Metal buffers are only guaranteed to keep resident when this mode is
  // on; without it cross-team reductions over 1024 teams lost partial results. Respect an explicit user setting.
  ::setenv("CUMETAL_USE_METAL_DEVICE_ADDRESSES", "1", /*overwrite=*/0);
#endif
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(CudaLib, &ErrMsg));
  if (!DynlibHandle->isValid()) {
    ODBG(OLDT_Init) << "Unable to load library ' " << CudaLib << "': " << ErrMsg
                    << "!";
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    auto It = TryFirst.find(Sym);
    if (It != TryFirst.end()) {
      const char *First = It->second;
      void *P = DynlibHandle->getAddressOfSymbol(First);
      if (P) {
        ODBG(OLDT_Init) << "Implementing " << Sym << " with dlsym(" << First
                        << ") -> " << P;
        *dlwrap::pointer(I) = P;
        continue;
      }
    }

    void *P = DynlibHandle->getAddressOfSymbol(Sym);
#ifdef __APPLE__
    // cuda.h renames some functions to <name>_v2; CuMetal exports a few of them only under the plain name
    if (P == nullptr) {
      std::string SymName(Sym);
      if (SymName.size() > 3 && SymName.compare(SymName.size() - 3, 3, "_v2") == 0) {
        P = DynlibHandle->getAddressOfSymbol(SymName.substr(0, SymName.size() - 3).c_str());
        if (P)
          ODBG(OLDT_Init) << "Implementing " << Sym << " with dlsym(" << SymName.substr(0, SymName.size() - 3)
                          << ") -> " << P;
      }
    }
    if (P == nullptr) {
      if (void *Stub = optionalDriverFunctionStub(Sym)) {
        ODBG(OLDT_Init) << "'" << Sym << "' is not in '" << CudaLib << "': using a stub";
        *dlwrap::pointer(I) = Stub;
        continue;
      }
    }
#endif
    if (P == nullptr) {
      ODBG(OLDT_Init) << "Unable to find '" << Sym << "' in '" << CudaLib
                      << "'!";
      return false;
    }
    ODBG(OLDT_Init) << "Implementing " << Sym << " with dlsym(" << Sym
                    << ") -> " << P;

    *dlwrap::pointer(I) = P;
  }

  return true;
}

CUresult cuInit(unsigned X) {
  // Note: Called exactly once from cuda rtl.cpp in a global constructor so
  // does not need to handle being called repeatedly or concurrently
  if (!checkForCUDA()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return dlwrap_cuInit(X);
}
