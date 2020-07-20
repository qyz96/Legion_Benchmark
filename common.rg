import "regent"

local common = {}
local blas = terralib.includecstring [[
extern void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
                   double* A, int* lda, double* B, int* ldb, double* beta,
                   double* C, int* ldc);

]]

if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  terralib.linklibrary("libblas.dylib")
  terralib.linklibrary("liblapack.dylib")
else
  terralib.linklibrary("libmkl_core.so")
  terralib.linklibrary("libmkl_sequential.so")
  terralib.linklibrary("libmkl_intel_lp64.so")
end

local blas = terralib.includecstring [[
extern void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
                   double* A, int* lda, double* B, int* ldb, double* beta,
                   double* C, int* ldc);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);

extern void dtrsm_(char* side, char *uplo, char* transa, char* diag,
                   int* m, int* n, double* alpha,
                   double *A, int *lda, double *B, int *ldb);

extern void dsyrk_(char *uplo, char* trans, int* n, int* k,
                   double* alpha, double *A, int *lda,
                   double* beta, double *C, int *ldc);

]]

local c = regentlib.c
local cstr = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)

-- declare fortran-order 2D indexspace
local struct __f2d { i : int, j : int }
local f2d = regentlib.index_type(__f2d, "f2d")
common.f2d = f2d

task make_random_matrix(matrix_size : int, rA : region(ispace(f2d), double), use_double : bool)
where reads writes(rA)
do
  for p in rA.ispace do
    if use_double then rA[p] = drand48()/[double](matrix_size)
    else rA[p] = [int](10.0*drand48()) end
  end
end

task make_zero_matrix(rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    rA[p] = 0
  end
end

terra find_ik(ik : int[2], z : int, n : int)
  var q = z / (n+1)
  if (z % (n + 1)) <= q then 
    ik[0] = q
    ik[1] = z % (n + 1)
  else 
    ik[0] = n - q 
    ik[1] = (q + 1) * (n + 1) - z - 1
  end
end

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

task make_pds_matrix(p : f2d, matrix_size : int, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  var is = rA.ispace
  for p in rA.ispace do
    if p.j <= p.i then
      rA[p] = [int](10.0 * drand48())
      if p.j == p.i then
        rA[p] += matrix_size * 10
      end
      rA[f2d { i = p.j, j = p.i }] = rA[p]
    end
  end
end

task transpose_copy(rSrc : region(ispace(f2d), double), rDst : region(ispace(f2d), double))
where reads(rSrc), writes(rDst)
do
  for p in rSrc.ispace do
    rDst[f2d { i = p.j, j = p.i }] = rSrc[p]
  end
end

local raw_ptr = raw_ptr_factory(double)

terra get_raw_ptr(i : int, j : int, matrix_size : int, block_size : int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  regentlib.assert(i * block_size >= 0, "row index (low) of region rectangle out of bounds")
  regentlib.assert(j * block_size >= 0, "col index (low) of region rectangle out of bounds")
  regentlib.assert((i+1) * block_size - 1< matrix_size, "row index (high) of region rectangle out of bounds")
  regentlib.assert((j+1) * block_size - 1< matrix_size, "col index (high) of region rectangle out of bounds")
  rect.lo.x[0] = i * block_size
  rect.lo.x[1] = j * block_size
  rect.hi.x[0] = (i + 1) * block_size - 1
  rect.hi.x[1] = (j + 1) * block_size - 1
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dpotrf_terra(j : int, matrix_size : int, block_size : int, val_A : double,
                   pr : c.legion_physical_region_t,
                   fld : c.legion_field_id_t)
  var uplo : rawstring = 'L'
  var matrix_size_ : int[1], block_size_ : int[1]
  matrix_size_[0], block_size_[0] = matrix_size, block_size
  var info : int[1]
  var rawA = get_raw_ptr(j, j, matrix_size, block_size, pr, fld)
  regentlib.assert(cmath.fabs(val_A - rawA.ptr[0]) == 0, "error reading matrix A in potrf!")
  blas.dpotrf_(uplo, block_size_, rawA.ptr, &(rawA.offset), info)
  regentlib.assert(info[0] == 0, "matrix not spd in potrf!")
end

task dpotrf(j : int, matrix_size : int, block_size : int, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  dpotrf_terra(j, matrix_size, block_size, rA[ f2d{ i = j * block_size, j = j * block_size }], __physical(rA)[0], __fields(rA)[0])
end

terra dtrsm_terra(j : int, i : int, matrix_size : int, block_size : int,
                  val_A : double, val_B :double,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)

  var side : rawstring = 'R'
  var uplo : rawstring = 'L'
  var transa : rawstring = 'T'
  var diag : rawstring = 'N'
  var matrix_size_ : int[1], block_size_ : int[1]
  matrix_size_[0], block_size_[0] = matrix_size, block_size
  var alpha : double[1] = array(1.0)
  var rawA = get_raw_ptr(i, j, matrix_size, block_size, prA, fldA)
  var rawB = get_raw_ptr(j, j, matrix_size, block_size, prB, fldB)
  regentlib.assert(cmath.fabs(val_A - rawA.ptr[0]) == 0, "error reading matrix A in trsm!")
  regentlib.assert(cmath.fabs(val_B - rawB.ptr[0]) == 0, "error reading matrix B in trsm!")
  blas.dtrsm_(side, uplo, transa, diag, block_size_, block_size_, alpha,
              rawB.ptr, &(rawB.offset), rawA.ptr, &(rawA.offset))
end

task dtrsm(j : int, i : int, matrix_size : int, block_size : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads writes(rA), reads(rB)
do
  dtrsm_terra(j, i, matrix_size, block_size, 
              rA[ f2d{ i = i * block_size, j = j * block_size }],
              rB[ f2d{ i = j * block_size, j = j * block_size }],
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dsyrk_terra(j : int, i : int, matrix_size : int, block_size : int,
                  val_A : double, val_B : double,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)

  var uplo : rawstring = 'L'
  var trans : rawstring = 'N'
  var matrix_size_ : int[1], block_size_ : int[1]
  matrix_size_[0], block_size_[0] = matrix_size, block_size
  var alpha : double[1] = array(-1.0)
  var beta : double[1] = array(1.0)
  var rawA = get_raw_ptr(i, i, matrix_size, block_size, prA, fldA)
  var rawB = get_raw_ptr(i, j, matrix_size, block_size, prB, fldB)
  regentlib.assert(cmath.fabs(val_A - rawA.ptr[0]) == 0, "error reading matrix A in syrk!")
  regentlib.assert(cmath.fabs(val_B - rawB.ptr[0]) == 0, "error reading matrix B in syrk!")
  blas.dsyrk_(uplo, trans, block_size_, block_size_,
              alpha, rawB.ptr, &(rawB.offset),
              beta, rawA.ptr, &(rawA.offset))
end

task dsyrk(j : int, i : int, matrix_size : int, block_size : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads writes(rA), reads(rB)
do
  dsyrk_terra(j, i, matrix_size, block_size, 
      rA[ f2d{ i = i * block_size, j = i * block_size }], 
      rB[ f2d{ i = i * block_size, j = j * block_size }], 
      __physical(rA)[0], __fields(rA)[0], __physical(rB)[0], __fields(rB)[0])
end


terra dgemm_terra(transa : bool, transb : bool, idx_a : int[2], idx_b : int[2], idx_c : int[2],
                  matrix_size : int, block_size : int,
                  alpha : double, beta :double,
                  val_A : double, val_B: double, val_C : double,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)
  var transa_ : rawstring
  var transb_ : rawstring
  if transa then transa_='T'
  else transa_= 'N' end
  if transb then transb_='T'
  else transb_ = 'N' end
  var matrix_size_ : int[1], block_size_ : int[1]
  matrix_size_[0], block_size_[0] = matrix_size, block_size
  var alpha_ : double[1] = array(alpha)
  var beta_ : double[1] = array(beta)
  var rawA = get_raw_ptr(idx_a[0], idx_a[1], matrix_size, block_size, prA, fldA)
  var rawB = get_raw_ptr(idx_b[0], idx_b[1], matrix_size, block_size, prB, fldB)
  var rawC = get_raw_ptr(idx_c[0], idx_c[1], matrix_size, block_size, prC, fldC)
  regentlib.assert(cmath.fabs(val_A - rawA.ptr[0]) == 0, "error reading matrix A in gemm!")
  regentlib.assert(cmath.fabs(val_B - rawB.ptr[0]) == 0, "error reading matrix B in gemm!")
  regentlib.assert(cmath.fabs(val_C - rawC.ptr[0]) == 0, "error reading matrix C in gemm!")
  blas.dgemm_(transa_, transb_, block_size_, block_size_, block_size_,
              alpha_, rawB.ptr, &(rawB.offset),
              rawC.ptr, &(rawC.offset),
              beta_, rawA.ptr, &(rawA.offset))

end

task dgemm(transa : bool, transb : bool, idx_a : int[2], idx_b : int[2], idx_c : int[2], matrix_size : int, block_size : int,
           alpha : double, beta : double,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reads writes(rA), reads(rB, rC)
do
  dgemm_terra(transa, transb, idx_a, idx_b, idx_c, matrix_size, block_size, alpha, beta, rA[ f2d{ i = idx_a[0] * block_size, j = idx_a[1] * block_size }], rB[ f2d{ i = idx_b[0] * block_size, j = idx_b[1] * block_size }], rC[ f2d{ i = idx_c[0] * block_size, j = idx_c[1] * block_size }], __physical(rA)[0], __fields(rA)[0],__physical(rB)[0], __fields(rB)[0],__physical(rC)[0], __fields(rC)[0])
end

task hand_dgemm(k : int, block_size : int, rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reads writes(rA), reads(rB, rC)
do
  for p in rA.ispace do
    for kk = 0, block_size do
      rA[p] = rA[p] + rB[f2d{i=p.i,j=k*block_size+kk}] * rC[f2d{i=k*block_size+kk, j=p.j}]
    end
  end
end

task verify_result_gemm(matrix_size : int,
                   org : region(ispace(f2d), double),
                   res : region(ispace(f2d), double),
                   use_double : bool)
where reads(org, res)
do
  c.printf("verifying results...\n")
  var max_error : double
  if use_double then max_error = 1e-14
  else max_error=0 end
  for i = 0, matrix_size do
    for j = 0, matrix_size do
      var v = res[f2d { i = i, j = j }]
      var sum = org[f2d{i=i,j=j}]
      if (cmath.fabs(sum - v)) > max_error then
        c.printf("error %e at (%d, %d) : %.3f, %.3f\n", cmath.fabs(sum-v), i, j, sum, v)
      end
    end
  end
end

task verify_result_cholesky(matrix_size : int,
                   org : region(ispace(f2d), double),
                   res : region(ispace(f2d), double))
where reads(org, res)
do
  c.printf("verifying results...\n")
  for j = 0, matrix_size do
    for i = j, matrix_size do
      var v = org[f2d { i = i, j = j }]
      var sum : double = 0
      for k = 0, j + 1 do
        sum += res[f2d { i = i, j = k }] * res[f2d { i = j, j = k }]
      end
      if cmath.fabs(sum - v) > 1e-6 then                                                                                                                                                                                  c.printf("error at (%d, %d) : %.3f, %.3f\n", i, j, sum, v)                                                                                                                                                      end                                                                                                                                                                                                             end                                                                                                                                                                                                             end                                                                                                                                                                                                             end


return common
