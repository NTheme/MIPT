    .intel_syntax noprefix

    .text
    .global dot_product

dot_product:
    pxor xmm0, xmm0

cycle_block_cond:
    cmp rdi, 4
    jl cycle_elem_cond
    jmp cycle_block_body

cycle_block_body:
    movups xmm1, xmmword ptr [rsi]
    movups xmm2, xmmword ptr [rdx]
    mulps xmm1, xmm2
    haddps xmm1, xmm1
    haddps xmm1, xmm1
    addss xmm0, xmm1

    add rsi, 16
    add rdx, 16
    sub rdi, 4
    jmp cycle_block_cond
      
cycle_elem_cond:
    cmp rdi, 1
    jl return
    jmp cycle_elem_body

cycle_elem_body:
    movss xmm1, [rsi]
    movss xmm2, [rdx]
    mulss xmm1, xmm2
    addss xmm0, xmm1

    add rsi, 4
    add rdx, 4
    sub rdi, 1
    jmp cycle_elem_cond

return:
    movd rax, xmm0
    ret
