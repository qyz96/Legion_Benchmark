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


import "regent"
local c = regentlib.c

-- declare fortran-order 2D indexspace
local struct __f2d { y : int, x : int }
local f2d = regentlib.index_type(__f2d, "f2d")

task make_zero_matrix(p : f2d, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    rA[p] = [int](0.0)
  end
end

task dgemm(x : int, y : int, k : int, n : int, bn : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reads writes(rA), reads(rB, rC)
do
  for p in rB.ispace do
--    c.printf("B(%d,%d)=%.3f\n", p.x, p.y, rB[p])
    rA[p] = rB[p] + rC[p]
  end 
end


task my_gemm(n : int, np : int)
  regentlib.assert(n % np == 0, "tile sizes should be uniform")
  var is = ispace(f2d, { x = n, y = n })
  var cs = ispace(f2d, { x = np, y = np })
  var rA = region(is, double)
  var rB = region(is, double)
  var rC = region(is, double)
  var pA = partition(equal, rA, cs)
  var pB = partition(equal, rB, cs)
  var pC = partition(equal, rC, cs)
  for x=0, np do
    for y=0, np do
      make_zero_matrix(f2d{x=x,y=y},pA[f2d{x=x,y=y}])
      make_zero_matrix(f2d{x=x,y=y}, pB[f2d{x=x,y=y}])
      make_zero_matrix(f2d{x=x,y=y}, pC[f2d{x=x,y=y}])
    end
  end
  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()
  var bn = n / np
  for k = 0, np do
    for x = 0, np do
      for y = 0, np do
        dgemm(x, y, k, n, bn,
              pA[f2d { x = x, y = y }],
              pB[f2d { x = x, y = k }],
              pC[f2d { x = k, y = y }])
      end
    end
  end    
  __fence(__execution, __block)

end

task toplevel()
  var n = 8192
  var np = 16

  my_gemm(n, np)
end

regentlib.start(toplevel)
