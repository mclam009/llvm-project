// -fopenmp-version takes the decimal spellings of the versions that clang
// knows, 31, 40, 45, 50, 51, 52, 60 and 61, and nothing else. The frontend
// reads its argument with the radix of C (a leading 0 is octal, 0x hexadecimal),
// so a spelling that the driver let through could mean another version than it
// names: 061 is the octal 49, which clang does not know and takes as the
// default, OpenMP 5.1.

// Every version that is known, with and without the spelling's leading text
// RUN: %clang %s -fopenmp -fopenmp-version=31 -E -dM | FileCheck --check-prefix=V31 %s
// RUN: %clang %s -fopenmp -fopenmp-version=40 -E -dM | FileCheck --check-prefix=V40 %s
// RUN: %clang %s -fopenmp -fopenmp-version=45 -E -dM | FileCheck --check-prefix=V45 %s
// RUN: %clang %s -fopenmp -fopenmp-version=50 -E -dM | FileCheck --check-prefix=V50 %s
// RUN: %clang %s -fopenmp -fopenmp-version=51 -E -dM | FileCheck --check-prefix=V51 %s
// RUN: %clang %s -fopenmp -fopenmp-version=52 -E -dM | FileCheck --check-prefix=V52 %s
// RUN: %clang %s -fopenmp -fopenmp-version=60 -E -dM | FileCheck --check-prefix=V60 %s
// RUN: %clang %s -fopenmp -fopenmp-version=61 -E -dM | FileCheck --check-prefix=V61 %s
// V31: #define _OPENMP 201107
// V40: #define _OPENMP 201307
// V45: #define _OPENMP 201511
// V50: #define _OPENMP 201811
// V51: #define _OPENMP 202011
// V52: #define _OPENMP 202111
// V60: #define _OPENMP 202411
// V61: #define _OPENMP 202611

// The version that another number would name in C is not accepted
// RUN: not %clang %s -fopenmp -fopenmp-version=061 -E -dM 2>&1 | FileCheck --check-prefix=BAD-061 %s
// RUN: not %clang %s -fopenmp -fopenmp-version=051 -E -dM 2>&1 | FileCheck --check-prefix=BAD-051 %s
// RUN: not %clang %s -fopenmp -fopenmp-version=0052 -E -dM 2>&1 | FileCheck --check-prefix=BAD-0052 %s
// RUN: not %clang %s -fopenmp -fopenmp-version=0x34 -E -dM 2>&1 | FileCheck --check-prefix=BAD-0x34 %s
// RUN: not %clang %s -fopenmp -fopenmp-version=0b110100 -E -dM 2>&1 | FileCheck --check-prefix=BAD-0b110100 %s
// BAD-061: error: invalid value '061' in '-fopenmp-version=061', expected one of: 31, 40, 45, 50, 51, 52, 60, 61
// BAD-051: error: invalid value '051' in '-fopenmp-version=051', expected one of: 31, 40, 45, 50, 51, 52, 60, 61
// BAD-0052: error: invalid value '0052' in '-fopenmp-version=0052', expected one of: 31, 40, 45, 50, 51, 52, 60, 61
// BAD-0x34: error: invalid value '0x34' in '-fopenmp-version=0x34', expected one of: 31, 40, 45, 50, 51, 52, 60, 61
// BAD-0b110100: error: invalid value '0b110100' in '-fopenmp-version=0b110100', expected one of: 31, 40, 45, 50, 51, 52, 60, 61

// Nothing but the digits
// RUN: not %clang %s -fopenmp -fopenmp-version=52x -E -dM 2>&1 | FileCheck --check-prefix=BAD-52X %s
// RUN: not %clang %s -fopenmp "-fopenmp-version= 52" -E -dM 2>&1 | FileCheck --check-prefix=BAD-SPACE %s
// RUN: not %clang %s -fopenmp "-fopenmp-version=52 " -E -dM 2>&1 | FileCheck --check-prefix=BAD-TRAILING %s
// RUN: not %clang %s -fopenmp -fopenmp-version= -E -dM 2>&1 | FileCheck --check-prefix=BAD-EMPTY %s
// RUN: not %clang %s -fopenmp -fopenmp-version=-52 -E -dM 2>&1 | FileCheck --check-prefix=BAD-NEG %s
// RUN: not %clang %s -fopenmp -fopenmp-version=5.2 -E -dM 2>&1 | FileCheck --check-prefix=BAD-DOT %s
// RUN: not %clang %s -fopenmp -fopenmp-version=99999999999999999999 -E -dM 2>&1 | FileCheck --check-prefix=BAD-HUGE %s
// RUN: not %clang %s -fopenmp -fopenmp-version=52.0 -E -dM 2>&1 | FileCheck --check-prefix=BAD-FRAC %s
// BAD-52X: error: invalid value '52x' in '-fopenmp-version=52x'
// BAD-SPACE: error: invalid value ' 52' in '-fopenmp-version= 52'
// BAD-TRAILING: error: invalid value '52 ' in '-fopenmp-version=52 '
// BAD-EMPTY: error: invalid value '' in '-fopenmp-version='
// BAD-NEG: error: invalid value '-52' in '-fopenmp-version=-52'
// BAD-DOT: error: invalid value '5.2' in '-fopenmp-version=5.2'
// BAD-HUGE: error: invalid value '99999999999999999999' in '-fopenmp-version=99999999999999999999'
// BAD-FRAC: error: invalid value '52.0' in '-fopenmp-version=52.0'

// A number that is not a version, below, between and above the known ones
// RUN: not %clang %s -fopenmp -fopenmp-version=62 -E -dM 2>&1 | FileCheck --check-prefix=BAD-62 %s
// RUN: not %clang %s -fopenmp -fopenmp-version=53 -E -dM 2>&1 | FileCheck --check-prefix=BAD-53 %s
// RUN: not %clang %s -fopenmp -fopenmp-version=30 -E -dM 2>&1 | FileCheck --check-prefix=BAD-30 %s
// RUN: not %clang %s -fopenmp -fopenmp-version=99 -E -dM 2>&1 | FileCheck --check-prefix=BAD-99 %s
// BAD-62: error: invalid value '62' in '-fopenmp-version=62'
// BAD-53: error: invalid value '53' in '-fopenmp-version=53'
// BAD-30: error: invalid value '30' in '-fopenmp-version=30'
// BAD-99: error: invalid value '99' in '-fopenmp-version=99'

// The same with the other ways to switch OpenMP on (the simd-only mode defines
// no _OPENMP), when the flag comes last (the last one counts), and in a
// compilation that offloads, which diagnoses it in each of its jobs
// RUN: not %clang %s -fopenmp=libomp -fopenmp-version=061 -E -dM 2>&1 | FileCheck --check-prefix=BAD-061 %s
// RUN: not %clang %s -fopenmp-simd -fopenmp-version=061 -E -dM 2>&1 | FileCheck --check-prefix=BAD-061 %s
// RUN: not %clang %s -fopenmp -fopenmp-version=52 -fopenmp-version=061 -E -dM 2>&1 | FileCheck --check-prefix=BAD-061 %s
// RUN: %clang %s -fopenmp -fopenmp-version=061 -fopenmp-version=52 -E -dM | FileCheck --check-prefix=V52 %s
// RUN: not %clang %s -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu -fopenmp-version=061 -c -o /dev/null 2>&1 | FileCheck --check-prefix=BAD-061 %s

// The flag means nothing without OpenMP, and is not checked then
// RUN: %clang %s -fopenmp-version=061 -E -dM | FileCheck --check-prefix=NO-OPENMP %s
// NO-OPENMP-NOT: #define _OPENMP
