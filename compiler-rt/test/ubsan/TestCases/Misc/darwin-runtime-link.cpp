// REQUIRES: darwin, ubsan-standalone
// UNSUPPORTED: darwin-remote
// RUN: %clangxx -c %s -o %t.o
// RUN: %darwin_runtime_link -dynamiclib %t.o -Werror -Wl,-fatal_warnings -o %t.dylib
// RUN: %darwin_runtime_link -dynamiclib %t.o -nodefaultlibs -lSystem \
// RUN:   -Werror -Wl,-fatal_warnings -o %t.dylib

void *allocate() { return new char; }
