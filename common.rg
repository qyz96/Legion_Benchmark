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

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
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
  if transa then transa_='N'
  else transa_= 'L' end
  if transb then transb_='N'
  else transb_ = 'L' end
  var matrix_size_ : int[1], block_size_ : int[1]
  matrix_size_[0], block_size_[0] = matrix_size, block_size
  var alpha_ : double[1] = array(alpha)
  var beta_ : double[1] = array(beta)

  var rawA = get_raw_ptr(idx_a[0], idx_a[1], matrix_size, block_size, prA, fldA)
  var rawB = get_raw_ptr(idx_b[0], idx_b[1], matrix_size, block_size, prB, fldB)
  var rawC = get_raw_ptr(idx_c[0], idx_c[1], matrix_size, block_size, prC, fldC)

  regentlib.assert(cmath.fabs(val_A - rawA.ptr[0]) == 0, "error reading matrix A!")
  regentlib.assert(cmath.fabs(val_B - rawB.ptr[0]) == 0, "error reading matrix B!")
  regentlib.assert(cmath.fabs(val_C - rawC.ptr[0]) == 0, "error reading matrix C!")
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

task verify_result(matrix_size : int,
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


return common
