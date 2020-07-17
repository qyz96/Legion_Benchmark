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
 extern void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* A, int* lda, double* B, int* ldb, double* beta,double* C, int* ldc);
//  extern void my_dgemm(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* A, int* lda, double* B, int* ldb, double* beta,double* C, int* ldc);

//  extern void GEMM(int n, double* A, double* B, double* C);

]]

if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  terralib.linklibrary("libblas.dylib")
  terralib.linklibrary("liblapack.dylib")
else
--  terralib.linklibrary("myeigen.so")
--  terralib.linklibrary("myblas.so")
  terralib.linklibrary("libblas.so") 
  terralib.linklibrary("liblapack.so")
end

local c = regentlib.c
local cstr = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)

-- declare fortran-order 2D indexspace
local struct __f2d { y : int, x : int }
local f2d = regentlib.index_type(__f2d, "f2d")

task make_random_matrix(n : int, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    var xx : double = [double](p.x)
    var yy : double = [double](p.y)
    rA[p] = (1+(p.x % 47) + (p.y % 57))/[double](n)
    rA[p] = [int](drand48())
--    c.printf("rA(%d,%d): %3.f\n", p.x, p.y, rA[p])
  end
end

task make_zero_matrix(p : f2d, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    rA[p] = [int](0.0)
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

terra get_raw_ptr(y : int, x : int, bn : int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = y * bn
  rect.lo.x[1] = x * bn
  rect.hi.x[0] = (y + 1) * bn - 1
  rect.hi.x[1] = (x + 1) * bn - 1
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dgemm_terra(x : int, y : int, k : int,
                  n : int, bn : int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)

  var transa : rawstring = 'N'
  var transb : rawstring = 'N'
  var n_ : int[1], bn_ : int[1]
  n_[0], bn_[0] = n, bn
  var alpha : double[1] = array(1.0)
  var beta : double[1] = array(1.0)

  var rawA = get_raw_ptr(x, y, bn, prA, fldA)
  var rawB = get_raw_ptr(x, y, bn, prB, fldB)
  var rawC = get_raw_ptr(x, y, bn, prC, fldC)
  blas.dgemm_(transa, transb, bn_, bn_, bn_,alpha, rawB.ptr, &(rawB.offset),rawC.ptr, &(rawC.offset),beta, rawA.ptr, &(rawA.offset))
end

task dgemm(x : int, y : int, k : int, n : int, bn : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reads writes(rA), reads(rB, rC)
do
  dgemm_terra(x, y, k, n, bn,__physical(rA)[0], __fields(rA)[0],__physical(rB)[0], __fields(rB)[0],__physical(rC)[0], __fields(rC)[0])
end

task verify_result(n : int,
                   org : region(ispace(f2d), double),
                   res : region(ispace(f2d), double))
where reads(org, res)
do
  c.printf("verifying results...\n")
  for x = 0, n do
    for y = 0, n do
      var v = res[f2d { x = x, y = y }]
      var sum = org[f2d{x=x,y=y}]
      if cmath.fabs(sum - v) > 1e-12 then
        c.printf("error at (%d, %d) : %.3f, %.3f\n", y, x, sum, v)
      end
    end
  end
end

task my_gemm(n : int, np : int, verify : bool)
  regentlib.assert(n % np == 0, "tile sizes should be uniform")
  var is = ispace(f2d, { x = n, y = n })
  var cs = ispace(f2d, { x = np, y = np })

  var rA = region(is, double)
  var rB = region(is, double)
  var rC = region(is, double)
  var rD = region(is, double)

  var pA = partition(equal, rA, cs)
  var pB = partition(equal, rB, cs)
  var pC = partition(equal, rC, cs)
  var launch_domain = rect2d { int2d {0, 0}, int2d {np - 1, np - 1} }
  make_random_matrix(n,rB)
  make_random_matrix(n,rC)
  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()
  var bn = n / np
  __demand(__index_launch)
  for p in cs do
    dgemm(p.x, p.y, 0, n, bn,
            pA[f2d { x = p.x, y = p.y }],
            pB[f2d { x = p.x, y = p.y }],
            pC[f2d { x = p.x, y = p.y }])
  end
  __fence(__execution, __block)

  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("ELAPSED TIME = %7.3f ms\n", 1e-3 * (ts_end - ts_start))
end

task toplevel()
  var n = 8
  var np = 4
  var verify = false

  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstr.strcmp(args.argv[i], "-n") == 0 then
      n = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-p") == 0 then
      np = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-verify") == 0 then
      verify = true
    end
  end

  my_gemm(n, np, verify)
end

regentlib.start(toplevel)
