// REQUIRES: x86-registered-target
// Test that by default -fnoopenmp-use-tls is passed to frontend.
//
// RUN: %clang %s -### -o %t.o 2>&1 -fopenmp=libomp | FileCheck --check-prefix=CHECK-DEFAULT %s

// CHECK-DEFAULT: "-cc1"
// CHECK-DEFAULT-NOT: -fnoopenmp-use-tls
//
// RUN: %clang %s -### -o %t.o 2>&1 -fopenmp=libomp -fnoopenmp-use-tls | FileCheck --check-prefix=CHECK-NO-TLS %s

// CHECK-NO-TLS: "-cc1"
// CHECK-NO-TLS-SAME: -fnoopenmp-use-tls
//
// RUN: %clang %s -c -E -dM -fopenmp=libomp | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=60 | FileCheck --check-prefix=CHECK-60-VERSION %s

// A version that clang does not know is an error, not the default version.
// RUN: not %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=1 2>&1 | FileCheck --check-prefix=CHECK-BAD-1 %s
// RUN: not %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=0 2>&1 | FileCheck --check-prefix=CHECK-BAD-0 %s
// RUN: not %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=100 2>&1 | FileCheck --check-prefix=CHECK-BAD-100 %s
// RUN: not %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=5.1 2>&1 | FileCheck --check-prefix=CHECK-BAD-DOT %s
// RUN: not %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=1 2>&1 | FileCheck --check-prefix=CHECK-BAD-1 %s
// RUN: not %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=0 2>&1 | FileCheck --check-prefix=CHECK-BAD-0 %s
// RUN: not %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=100 2>&1 | FileCheck --check-prefix=CHECK-BAD-100 %s
// CHECK-BAD-1: error: invalid value '1' in '-fopenmp-version=1', expected one of: 31, 40, 45, 50, 51, 52, 60, 61
// CHECK-BAD-0: error: invalid value '0' in '-fopenmp-version=0', expected one of: 31, 40, 45, 50, 51, 52, 60, 61
// CHECK-BAD-100: error: invalid value '100' in '-fopenmp-version=100', expected one of: 31, 40, 45, 50, 51, 52, 60, 61
// CHECK-BAD-DOT: error: invalid value '5.1' in '-fopenmp-version=5.1', expected one of: 31, 40, 45, 50, 51, 52, 60, 61
// RUN: %clang %s -c -E -dM -fopenmp=libomp | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s

// CHECK-DEFAULT-VERSION: #define _OPENMP 202011
// CHECK-60-VERSION: #define _OPENMP 202411

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=31 | FileCheck --check-prefix=CHECK-31-VERSION %s
// CHECK-31-VERSION: #define _OPENMP 201107

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=40 | FileCheck --check-prefix=CHECK-40-VERSION %s
// CHECK-40-VERSION: #define _OPENMP 201307

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 | FileCheck --check-prefix=CHECK-45-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 -fopenmp-simd | FileCheck --check-prefix=CHECK-45-VERSION %s
// CHECK-45-VERSION: #define _OPENMP 201511
// CHECK-45-VERSION2: #define _OPENMP 201511

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=50 | FileCheck --check-prefix=CHECK-50-VERSION %s
// CHECK-50-VERSION: #define _OPENMP 201811

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=60 | FileCheck --check-prefix=CHECK-60-VERSION %s

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=61 | FileCheck --check-prefix=CHECK-61-VERSION %s
// CHECK-61-VERSION: #define _OPENMP 202611

// RUN: %clang %s -c -E -dM -fopenmp=libomp | FileCheck --check-prefix=CHECK-51-VERSION %s
// CHECK-51-VERSION: #define _OPENMP 202011

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=52 | FileCheck --check-prefix=CHECK-52-VERSION %s
// CHECK-52-VERSION: #define _OPENMP 202111

// RUN: %clang %s -c -E -dM -fopenmp-version=1 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=31 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=40 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fno-openmp-simd -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=31 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=40 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=60 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd | FileCheck --check-prefix=CHECK-VERSION %s

// CHECK-VERSION-NOT: #define _OPENMP

