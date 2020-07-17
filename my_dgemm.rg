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

import "regent"

local blas = terralib.includecstring [[
extern void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
                   double* A, int* lda, double* B, int* ldb, double* beta,
                   double* C, int* ldc);

]]

if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  terralib.linklibrary("libblas.dylib")
  terralib.linklibrary("liblapack.dylib")
else
--  terralib.linklibrary("libblas.so")
--  terralib.linklibrary("liblapack.so")
  terralib.linklibrary("libmkl_core.so")
  terralib.linklibrary("libmkl_sequential.so")
  terralib.linklibrary("libmkl_intel_lp64.so")
end

local c = regentlib.c
local cstr = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")

-- declare fortran-order 2D indexspace
local struct __f2d { i : int, j : int }
local f2d = regentlib.index_type(__f2d, "f2d")

task make_easy_matrix(rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    rA[p] = [double](p.j)
  end
end

task verify_easy_matrix(matrix_size : int,
                        computed    : region(ispace(f2d), double))
where reads(computed)
do
  c.printf("Verifying results...\n")
  var k = 0
  for i = 0, matrix_size do
    for j = 0, matrix_size do
      var cpt = computed[f2d{i=i,j=j}]
      var ref = j * (matrix_size*(matrix_size-1)/2)
      if cmath.fabs(cpt - ref) > 0.0 then
        c.printf("error at (%d, %d) : cpt = %.3f, ref = %.3f\n", i, j, cpt, ref)
      end
      k = k+1
    end
  end
  c.printf("Verified %d entires\n", k)
end

task make_random_matrix(rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    var ii : double = [double](p.i)
    var jj : double = [double](p.j)
    rA[p] = 1.0 + (ii + 2.0 * jj) % 47.0;
    -- c.printf("rA(%d,%d): %3.f\n", p.i, p.j, rA[p])
  end
end

task make_zero_matrix(rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    rA[p] = (0.0)
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

-- C[i,j] += A[i,k] * B[k,j]
terra dgemm_terra(i : int, j : int, k : int,
                  matrix_size : int, block_size : int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)

  var transa : rawstring = 'N'
  var transb : rawstring = 'N'
  var block_size_ : int[1]
  block_size_[0] = block_size
  var alpha : double[1] = array(1.0)
  var beta : double[1] = array(1.0)

  var rawA = get_raw_ptr(i, k, block_size, prA, fldA)
  var rawB = get_raw_ptr(k, j, block_size, prB, fldB)
  var rawC = get_raw_ptr(i, j, block_size, prC, fldC)

  blas.dgemm_(transa, transb, block_size_, block_size_, block_size_,
              alpha, rawA.ptr, &(rawA.offset),
              rawB.ptr, &(rawB.offset),
              beta, rawC.ptr, &(rawC.offset))
end

task dgemm(i : int, j : int, k : int, matrix_size : int, block_size : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reads writes(rC), reads(rA, rB)
do
  dgemm_terra(i, j, k, matrix_size, block_size, __physical(rA)[0], __fields(rA)[0],__physical(rB)[0], __fields(rB)[0],__physical(rC)[0], __fields(rC)[0])
end

task dgemm_verify(matrix_size : int,
                  rA : region(ispace(f2d), double),
                  rB : region(ispace(f2d), double),
                  rC : region(ispace(f2d), double))
where reads writes(rC), reads(rA, rB)
do
  for p in rC.ispace do
    rC[p] = 0.0
    for k = 0, matrix_size do
      rC[p] = rC[p] + rA[f2d{i=p.i,j=k}] * rB[f2d{i=k, j=p.j}]
    end
  end
end

task verify_result(matrix_size : int,
                   computed  : region(ispace(f2d), double),
                   reference : region(ispace(f2d), double))
where reads(computed, reference)
do
  c.printf("Verifying results...\n")
  var k = 0
  for i = 0, matrix_size do
    for j = 0, matrix_size do
      var cpt = computed[f2d{i=i,j=j}];
      var ref = reference[f2d{i=i,j=j}];
      if cmath.fabs(cpt - ref) > 0.0 then
        c.printf("error at (%d, %d) : cpt = %.3f, ref = %.3f\n", i, j, cpt, ref)
      end
      k = k+1
    end
  end
  c.printf("Verified %d entires\n", k)
end

task my_gemm(matrix_size : int, num_blocks : int, verify : bool)
  regentlib.assert(matrix_size % num_blocks == 0, "tile sizes should be uniform")
  var block_size = matrix_size / num_blocks
  var is = ispace(f2d, { i = matrix_size, j = matrix_size })
  var cs = ispace(f2d, { i = num_blocks,  j = num_blocks  })
  var ds = ispace(f2d, { i = 1,           j = 1           })
  var rA = region(is, double)
  var rB = region(is, double)
  var rC = region(is, double)
  var rD = region(is, double)
  var pA = partition(equal, rA, cs)
  var pB = partition(equal, rB, cs)
  var pC = partition(equal, rC, cs)
  for i = 0, num_blocks do
    for j = 0, num_blocks do
      make_random_matrix(pA[f2d{i=i,j=j}])
      make_random_matrix(pB[f2d{i=i,j=j}])
      -- make_easy_matrix(pA[f2d{i=i,j=j}])
      -- make_easy_matrix(pB[f2d{i=i,j=j}])
      make_zero_matrix(pC[f2d{i=i,j=j}])
    end
  end
  -- C[i,j] = sum_k A[i,k] * B[k,j]
  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()
  for k = 0, num_blocks do
    for i = 0, num_blocks do
      __demand(__index_launch)
      for j = 0, num_blocks do
        dgemm(i, j, k, matrix_size, block_size,
              pA[f2d { i=i, j=k }],
              pB[f2d { i=k, j=j }],
              pC[f2d { i=i, j=j }])
      end
    end
  end
  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("ELAPSED TIME = %7.3f ms\n", 1e-3 * (ts_end - ts_start))
  if verify then
    make_zero_matrix(rD)
    -- dgemm(0, 0, 0, matrix_size, matrix_size, rA, rB, rD)
    dgemm_verify(matrix_size, rA, rB, rD)
    verify_result(matrix_size, rC, rD) 
    -- verify_easy_matrix(matrix_size, rC)
  end
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

  my_gemm(matrix_size, num_blocks, verify)
end

regentlib.start(toplevel)
