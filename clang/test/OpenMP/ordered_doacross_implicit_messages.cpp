// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 -ferror-limit 100 -Wno-deprecated -o - %s
// RUN: %clang_cc1 -verify=expected,omp52 -fopenmp -fopenmp-version=52 -ferror-limit 100 -Wno-deprecated -o - %s
// RUN: %clang_cc1 -verify=expected,omp52 -fopenmp -fopenmp-version=60 -ferror-limit 100 -Wno-deprecated -o - %s

// OpenMP 5.2 [4.4.4, ordered Clause]: "If n is not specified then the behavior
// is as if n is specified with the same value as is specified for the collapse
// clause on the construct." A loop with an 'ordered' clause without an argument
// is a doacross loop if a stand-alone 'ordered' construct binds to it, and then
// a block-associated 'ordered' construct must not bind to it. In OpenMP 5.1 a
// doacross loop needs the argument.

void foo();

template <typename T, int N>
void tmpl(T *a) {
#pragma omp parallel for ordered collapse(N)
  for (int i = 1; i < 10; ++i)
    for (int j = 1; j < 10; ++j) {
#pragma omp ordered depend(sink : i - 1, j) // omp51-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
      a[i] += a[i - 1];
#pragma omp ordered depend(source) // omp51-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
    }
}

int main(int argc, char **argv) {
  int a[10][10];
#pragma omp parallel for ordered
  for (int i = 1; i < 10; ++i) {
#pragma omp ordered depend(sink : i - 1) // omp51-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
    a[i][0] += a[i - 1][0];
#pragma omp ordered depend(source) // omp51-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
  }

#if _OPENMP >= 202111
#pragma omp for ordered
  for (int i = 1; i < 10; ++i) {
#pragma omp ordered doacross(sink : omp_cur_iteration - 1)
    a[i][0] += a[i - 1][0];
#pragma omp ordered doacross(source :)
  }

#pragma omp parallel for ordered collapse(2)
  for (int i = 1; i < 10; ++i)
    for (int j = 1; j < 10; ++j) {
#pragma omp ordered doacross(sink : i - 1, j) doacross(sink : i, j - 1)
      a[i][j] += a[i - 1][j] + a[i][j - 1];
#pragma omp ordered doacross(sink : i - 1) // omp52-error {{expected 'j' loop iteration variable}}
#pragma omp ordered doacross(sink : i - 1, j, j) // omp52-error {{unexpected expression: number of expressions is larger than the number of associated loops}}
#pragma omp ordered doacross(source : omp_cur_iteration)
    }

#pragma omp parallel for ordered
  for (int i = 1; i < 10; ++i) {
#pragma omp ordered // omp52-note {{previous 'ordered' directive used here}}
    foo();
#pragma omp ordered doacross(source :) // omp52-error {{'ordered' directive with 'doacross' or 'depend' clauses cannot bind to the same region as an 'ordered' directive without them}}
  }

#pragma omp parallel for ordered
  for (int i = 1; i < 10; ++i) {
#pragma omp ordered doacross(sink : i - 1) // omp52-note 2 {{previous 'ordered' directive used here}}
#pragma omp ordered doacross(source :)
#pragma omp ordered // omp52-error {{'ordered' directive without 'doacross' or 'depend' clauses cannot bind to the same region as an 'ordered' directive with them}}
    foo();
#pragma omp ordered threads // omp52-error {{'ordered' directive without 'doacross' or 'depend' clauses cannot bind to the same region as an 'ordered' directive with them}}
    foo();
  }
#endif

  // A block-associated 'ordered' construct alone keeps an 'ordered' loop.
#pragma omp parallel for ordered
  for (int i = 1; i < 10; ++i) {
#pragma omp ordered
    foo();
  }

  tmpl<int, 2>(&a[0][0]);
  return 0;
}
