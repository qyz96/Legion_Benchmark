-- Copyright 2020 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- runs-with:
-- [
--   ["-verify", "-ll:cpu", "4", "-fflow", "0"],
--   ["-p", "1", "-verify", "-fflow", "0"]
-- ]

-- FIXME: Breaks RDIR

import "regent"

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

if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  terralib.linklibrary("libblas.dylib")
  terralib.linklibrary("liblapack.dylib")
else
  terralib.linklibrary("myblas.so")
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

task make_random_matrix(p : f2d, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    rA[p] = [int](10.0 * drand48())
  end
end

task transpose_copy(rSrc : region(ispace(f2d), double), rDst : region(ispace(f2d), double))
where reads(rSrc), writes(rDst)
do
  for p in rSrc.ispace do
    rDst[f2d { i = p.j, j = p.i }] = rSrc[p]
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

terra get_raw_ptr(i : int, j : int, block_size : int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = i * block_size
  rect.lo.x[1] = j * block_size
  rect.hi.x[0] = (i + 1) * block_size - 1
  rect.hi.x[1] = (j + 1) * block_size - 1
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dpotrf_terra(j : int, matrix_size : int, block_size : int,
                   pr : c.legion_physical_region_t,
                   fld : c.legion_field_id_t)
  var uplo : rawstring = 'L'
  var matrix_size_ : int[1], block_size_ : int[1]
  matrix_size_[0], block_size_[0] = matrix_size, block_size
  var info : int[1]

  var rawA = get_raw_ptr(j, j, block_size, pr, fld)

  blas.dpotrf_(uplo, block_size_, rawA.ptr, &(rawA.offset), info)
end

task dpotrf(j : int, matrix_size : int, block_size : int, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  dpotrf_terra(j, matrix_size, block_size, __physical(rA)[0], __fields(rA)[0])
end

terra dtrsm_terra(j : int, i : int, matrix_size : int, block_size : int,
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

  var rawA = get_raw_ptr(i, j, block_size, prA, fldA)
  var rawB = get_raw_ptr(j, j, block_size, prB, fldB)

  blas.dtrsm_(side, uplo, transa, diag, block_size_, block_size_, alpha,
              rawB.ptr, &(rawB.offset), rawA.ptr, &(rawA.offset))
end

task dtrsm(j : int, i : int, matrix_size : int, block_size : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads writes(rA), reads(rB)
do
  dtrsm_terra(j, i, matrix_size, block_size, __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dsyrk_terra(j : int, k : int, matrix_size : int, block_size : int,
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

  var rawA = get_raw_ptr(k, k, block_size, prA, fldA)
  var rawB = get_raw_ptr(k, j, block_size, prB, fldB)

  blas.dsyrk_(uplo, trans, block_size_, block_size_,
              alpha, rawB.ptr, &(rawB.offset),
              beta, rawA.ptr, &(rawA.offset))
end

task dsyrk(j : int, k : int, matrix_size : int, block_size : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads writes(rA), reads(rB)
do
  dsyrk_terra(j, k, matrix_size, block_size, __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dgemm_terra(j : int, i : int, k : int,
                  matrix_size : int, block_size : int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)

  var transa : rawstring = 'N'
  var transb : rawstring = 'T'
  var matrix_size_ : int[1], block_size_ : int[1]
  matrix_size_[0], block_size_[0] = matrix_size, block_size
  var alpha : double[1] = array(-1.0)
  var beta : double[1] = array(1.0)

  var rawA = get_raw_ptr(i, k, block_size, prA, fldA)
  var rawB = get_raw_ptr(i, j, block_size, prB, fldB)
  var rawC = get_raw_ptr(k, j, block_size, prC, fldC)

  blas.dgemm_(transa, transb, block_size_, block_size_, block_size_,
              alpha, rawB.ptr, &(rawB.offset),
              rawC.ptr, &(rawC.offset),
              beta, rawA.ptr, &(rawA.offset))
end


task dgemm(j : int, i : int, k : int, matrix_size : int, block_size : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reads writes(rA), reads(rB, rC)
do
  dgemm_terra(j, i, k, matrix_size, block_size,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0],
              __physical(rC)[0], __fields(rC)[0])
end

task verify_result(matrix_size : int,
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
      if cmath.fabs(sum - v) > 1e-6 then
        c.printf("error at (%d, %d) : %.3f, %.3f\n", i, j, sum, v)
      end
    end
  end
end

task cholesky(matrix_size : int, num_blocks : int, verify : bool)
  regentlib.assert(matrix_size % num_blocks == 0, "tile sizes should be uniform")
  var is = ispace(f2d, { j = matrix_size, i = matrix_size })
  var cs = ispace(f2d, { j = num_blocks, i = num_blocks })
  var rA = region(is, double)
  var rB = region(is, double)

  var pA = partition(equal, rA, cs)
  var pB = partition(equal, rB, cs)

  for j = 0, num_blocks do
    make_pds_matrix(f2d { i = j, j = j }, matrix_size, pA[f2d { i = j, j = j }])
    for i = j + 1, num_blocks do
      make_random_matrix(f2d { i = i, j = j }, pA[f2d { i = i, j = j }])
      var src = pA[f2d { i = i, j = j }]
      var dst = pA[f2d { i = j, j = i }]
      transpose_copy(src, dst)
    end
  end

  for c in cs do
    var src = pA[c]
    var dst = pB[c]
    copy(src, dst)
  end

  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()

  var block_size = matrix_size / num_blocks
  for j = 0, num_blocks do
    dpotrf(j, matrix_size, block_size, pB[f2d { i = j, j = j }])
    __demand(__index_launch)
    for i = j + 1, num_blocks do
      dtrsm(j, i, matrix_size, block_size, pB[f2d { i = i, j = j }], pB[f2d { i = j, j = j }])
    end
    __demand(__index_launch)
    for k = j + 1, num_blocks do
      dsyrk(j, k, matrix_size, block_size, pB[f2d { i = k, j = k }], pB[f2d { i = k, j = j }])
    end
    for k = j + 1, num_blocks do
      __demand(__index_launch)
      for i = k + 1, num_blocks do
        dgemm(j, i, k, matrix_size, block_size,
              pB[f2d { i = i, j = k }],
              pB[f2d { i = i, j = j }],
              pB[f2d { i = k, j = j }])
      end
    end
  end
  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("ELAPSED TIME = %7.3f ms\n", 1e-3 * (ts_end - ts_start))

  if verify then verify_result(matrix_size, rA, rB) end
end

task toplevel()
  var matrix_size = 8
  var num_blocks = 4
  var verify = false

  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstr.strcmp(args.argv[i], "-n") == 0 then
      matrix_size = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-p") == 0 then
      num_blocks = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-verify") == 0 then
      verify = true
    end
  end

  cholesky(matrix_size, num_blocks, verify)
end

regentlib.start(toplevel)
