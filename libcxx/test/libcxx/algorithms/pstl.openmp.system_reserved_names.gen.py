# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Test that the OpenMP backend of the parallel algorithms (_LIBCPP_PSTL_BACKEND_OPENMP), which is only included when
# that macro is defined and so is not covered by libcxx/test/libcxx/system_reserved_names.gen.py, is not tripped up by
# the surrounding code defining the macros that system headers define (such as the names of Microsoft's annotations,
# __in, __out, __part and __output). It reuses the tests of that generator for the headers that include the backend.

# RUN: %{python} %s %{libcxx-dir}/utils %{libcxx-dir}/test/libcxx/system_reserved_names.gen.py
# END.

import re
import subprocess
import sys

utils, generator = sys.argv[1], sys.argv[2]

# The public headers that include <__pstl/backend.h>.
headers = ["algorithm", "execution", "memory", "numeric"]

directives = """\
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-incomplete-pstl
// REQUIRES: has-fopenmp
// ADDITIONAL_COMPILE_FLAGS: -fopenmp -D_LIBCPP_PSTL_BACKEND_OPENMP
"""

generated = subprocess.run([sys.executable, generator, utils], check=True, capture_output=True, text=True).stdout
for block in re.split(r"(?m)^(?=//--- )", generated):
    match = re.match(r"//--- (\w+)\.compile\.pass\.cpp\n", block)
    if not match or match.group(1) not in headers:
        continue
    first_line, rest = block.split("\n", 1)
    print(first_line.replace(".compile.pass.cpp", ".openmp.compile.pass.cpp"))
    print(directives + rest, end="")
