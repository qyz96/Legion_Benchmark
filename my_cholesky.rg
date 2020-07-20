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

local c = regentlib.c
local cstr = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
local common = require("common")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)
-- declare fortran-order 2D indexspace
local f2d = common.f2d

task cholesky(matrix_size : int, num_blocks : int, verify : bool, use_double : bool)
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
      make_random_matrix(matrix_size, pA[f2d { i = i, j = j }], use_double)
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
    for i = j + 1, num_blocks do
      dsyrk(j, i, matrix_size, block_size, pB[f2d { i = i, j = i }], pB[f2d { i = i, j = j }])
    end
    for i = j + 1, num_blocks do
      __demand(__index_launch)
      for k = i + 1, num_blocks do
        var idx_Bki : int[2] = array(k,i)
        var idx_Bkj : int[2] = array(k,j)
        var idx_Bij : int[2] = array(i,j)
        dgemm(false, true, idx_Bki, idx_Bkj, idx_Bij, matrix_size, block_size, -1.0, 1.0, pB[f2d { i = k, j = i }], pB[f2d { i = k, j = j }], pB[f2d { i = i, j = j }])       
      end
    end
  end
  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("ELAPSED TIME = %7.3f ms\n", 1e-3 * (ts_end - ts_start))

  if verify then verify_result_cholesky(matrix_size, rA, rB) end
end

task toplevel()
  var matrix_size = 8
  var num_blocks = 4
  var verify = false
  var use_double = false

  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstr.strcmp(args.argv[i], "-n") == 0 then
      matrix_size = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-p") == 0 then
      num_blocks = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-verify") == 0 then
      verify = true
    elseif cstr.strcmp(args.argv[i], "-use-double") == 0 then
      use_double = true
    end
  end

  cholesky(matrix_size, num_blocks, verify, use_double)
end

regentlib.start(toplevel)
