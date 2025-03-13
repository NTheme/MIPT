  .text
  .global longest_inc_subseq


longest_inc_subseq:
  mov x11, 0           // res
  mov x12, 1
  mov x13, 8

  mov x3, 0            // counter i
  mov x4, x0           // array num
  mov x5, x1           // array dp


cycle_elem_cond:
  cmp x3, x2
  b.lt cycle_elem_body
  mov x0, x11
  ret


cycle_elem_body:
  str x12, [x5]        // upd dp[i]

  mov x6, 0
  mov x7, x0           // array num
  mov x8, x1           // array dp


cycle_dp_cond:
  cmp x6, x3
  b.lt cycle_dp_body
  ldr x9, [x5]         // update max
  cmp x9, x11
  b.le cycle_elem_next
  mov x11, x9


cycle_dp_body:
  ldr x9, [x4]         // a[i]
  ldr x10, [x7]        // a[j]

  cmp x9, x10
  b.le cycle_dp_next

  ldr x9, [x5]         // dp[i]
  ldr x10, [x8]        // dp[j]
  cmp x9, x10
  b.gt cycle_dp_next

  add x9, x10, x12     // update dp 
  str x9, [x5]


cycle_dp_next:
  add x6, x6, x12      // step
  add x7, x7, x13
  add x8, x8, x13
  b cycle_dp_cond


cycle_elem_next:
  add x3, x3, x12      // step
  add x4, x4, x13
  add x5, x5, x13
  b cycle_elem_cond
