#include <mkl_blas.h>
#include <mkl_lapack.h>

/**
 *  * Build like
 *   *
 *    * gcc myblas.c -o myblas.so -shared -fPIC -m64 -I${MKLROOT}/include -shared -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
 *     */

void my_dgemm(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
               double* A, int* lda, double* B, int* ldb, double* beta,
               double* C, int* ldc) {
	dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void my_dpotrf(char *uplo, int *n, double *A, int *lda, int *info) {
	dpotrf(uplo, n, A, lda, info);
}

void my_dtrsm(char* side, char *uplo, char* transa, char* diag,
                   int* m, int* n, double* alpha,
                   double *A, int *lda, double *B, int *ldb) {
	dtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void my_dsyrk(char *uplo, char* trans, int* n, int* k,
                   double* alpha, double *A, int *lda,
                   double* beta, double *C, int *ldc) {
	dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

 

