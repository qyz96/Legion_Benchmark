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
  terralib.linklibrary("libmkl_core.so")
  terralib.linklibrary("libmkl_sequential.so")
  terralib.linklibrary("libmkl_intel_lp64.so")
end

local c = regentlib.c
local cstr = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
local common = require("common")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)

-- declare fortran-order 2D indexspace
local f2d = common.f2d


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

task my_gemm(matrix_size : int, num_blocks : int, verify : bool, use_double : bool)
  regentlib.assert(matrix_size % num_blocks == 0, "tile sizes should be uniform")
  var is = ispace(f2d, { i = matrix_size, j = matrix_size })
  var ds = ispace(f2d, { i = num_blocks, j = num_blocks * num_blocks })
  var cs = ispace(f2d, { i = num_blocks, j = num_blocks })

  var rA = region(is, double)
  var rB = region(is, double)
  var rC = region(is, double)
  var rD = region(is, double)

  var pA = partition(equal, rA, cs)
  var pB = partition(equal, rB, cs)
  var pC = partition(equal, rC, cs)
  var pD = partition(equal, rD, cs)
  for p in cs do
    make_zero_matrix(pA[p])       
    make_zero_matrix(pD[p]) 
    make_random_matrix(matrix_size, pB[p], use_double)
    make_random_matrix(matrix_size, pC[p], use_double)
  end
  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()
  var block_size = matrix_size / num_blocks
  for k = 0, num_blocks do
    __demand(__index_launch)
    for p in cs do
      var idx_a : int[2] = array(p.i, p.j) 
      var idx_b : int[2] = array(p.i, k)
      var idx_c : int[2] = array(k, p.j)
      dgemm(true, true, idx_a, idx_b, idx_c, matrix_size, block_size, 1.0, 1.0,
            pA[f2d { i = p.i, j = p.j }],
            pB[f2d { i = p.i, j = k }],
            pC[f2d { i = k, j = p.j }])
    end
  end
  __fence(__execution, __block)

  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("ELAPSED TIME = %7.3f ms\n", 1e-3 * (ts_end - ts_start))
  if verify then hand_dgemm(0, matrix_size,rD,rB,rC) end
  if verify then verify_result(matrix_size, rD, rA, use_double) end
end

task toplevel()
  var matrix_size = 8
  var num_block = 4
  var verify = false
  var use_double = false

  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstr.strcmp(args.argv[i], "-n") == 0 then
      matrix_size = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-p") == 0 then
      num_block = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-verify") == 0 then
      verify = true
    elseif cstr.strcmp(args.argv[i], "-use-double") == 0 then
      use_double = true
    end
  end

  my_gemm(matrix_size, num_block, verify, use_double)
end

regentlib.start(toplevel)
