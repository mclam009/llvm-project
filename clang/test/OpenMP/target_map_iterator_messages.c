// RUN: %clang_cc1 -verify=expected,lt51,lt52,lt60 -fopenmp -fno-openmp-extensions -fopenmp-version=50 -ferror-limit 200 %s
// RUN: %clang_cc1 -verify=expected,ge51,lt52,lt60,mid -fopenmp -fno-openmp-extensions -fopenmp-version=51 -ferror-limit 200 %s
// RUN: %clang_cc1 -verify=expected,ge51,ge52,lt60,mid -fopenmp -fno-openmp-extensions -fopenmp-version=52 -ferror-limit 200 %s
// RUN: %clang_cc1 -verify=expected,ge51,ge52,ge60 -fopenmp -fno-openmp-extensions -fopenmp-version=60 -ferror-limit 200 %s
// RUN: %clang_cc1 -verify=expected,ge51,lt52,lt60,mid -fopenmp-simd -fno-openmp-extensions -fopenmp-version=51 -ferror-limit 200 %s
// RUN: %clang_cc1 -verify=expected,ge51,ge52,ge60 -fopenmp-simd -fno-openmp-extensions -fopenmp-version=60 -ferror-limit 200 %s

// The iterator modifier of the map clause and of the to and from clauses is
// OpenMP 5.1: a modifier of the map clause that may stand anywhere among the
// other modifiers (from 5.2 they are separated by commas), and the same
// modifier of the motion clauses. It is an error before 5.1, and the iteration
// variable is not visible outside of the clause that declares it.

int a[10], b[10], n = 4;

void version_gate(void) {
  // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target map(iterator(it = 0:n), tofrom: a[it])
  ;
  // lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // lt51-error@+1 {{expected at least one 'map' clause for '#pragma omp target enter data'}}
#pragma omp target enter data map(iterator(it = 0:n), to: a[it])
  // lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // lt51-error@+1 {{expected at least one 'map' clause for '#pragma omp target exit data'}}
#pragma omp target exit data map(iterator(it = 0:n), from: a[it])
  // lt51-warning@+3 {{missing ':' after ) - ignoring}}
  // lt51-error@+2 {{use of undeclared identifier 'it'}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(iterator(it = 0:n): a[it])
  // lt51-warning@+3 {{missing ':' after ) - ignoring}}
  // lt51-error@+2 {{use of undeclared identifier 'it'}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(iterator(it = 0:n): a[it])
}

// The modifier first, between and after other modifiers
void positions(void) {
  // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target map(always, iterator(it = 0:n), tofrom: a[it])
  ;
  // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target map(close, always, iterator(it = 0:n), tofrom: a[it])
  ;
  // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target map(mapper(default), iterator(it = 0:n), tofrom: a[it])
  ;
  // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target map(iterator(it = 0:n), mapper(default), tofrom: a[it])
  ;
}

void missing_commas(void) {
  // The comma after an iterator modifier that comes first is required in every version that has it
  // expected-error@+2 {{expected ','}}
  // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target map(iterator(it = 0:n) tofrom: a[it])
  ;
  // After another modifier the comma is only required from 5.2 on
  // lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // ge52-error@+1 {{missing ',' after map type modifier}}
#pragma omp target map(always iterator(it = 0:n), tofrom: a[it])
  ;
}

void duplicates(void) {
  // lt51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // ge51-error@+1 {{same map type modifier has been specified more than once}}
#pragma omp target map(iterator(it = 0:n), always, iterator(jt = 0:n), tofrom: a[it])
  ;
  // lt51-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // lt51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // lt51-error@+2 {{same map type modifier has been specified more than once}}
  // ge51-error@+1 {{same map type modifier has been specified more than once}}
#pragma omp target map(always, iterator(it = 0:n), close, iterator(jt = 0:n), tofrom: a[it])
  ;
}

void scope_of_the_iterator(void) {
  // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target map(iterator(it = 0:n), to: a[it]) map(from: b[it]) // expected-error {{use of undeclared identifier 'it'}}
  ;
  // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target map(always, iterator(it = 0:n), to: a[it]) map(from: b[it]) // expected-error {{use of undeclared identifier 'it'}}
  ;
}

void map_type_first(void) {
  // From 6.0 the map type is a modifier that may stand first, as in map(tofrom, iterator(...): ...)
  // lt51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // mid-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
#pragma omp target map(tofrom, iterator(it = 0:n): a[it]) // lt60-error {{missing map type}}
  ;
}
