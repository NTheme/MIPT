  .intel_syntax noprefix

  .text
  .global add_scanf

.scanf_format_string:
        .string "%Ld%Ld"

add_scanf:
        push    rbp
        mov     rbp, rsp
        sub     rsp, 16
        lea     rdx, [rbp-16]
        lea     rsi, [rbp-8]
        mov     rdi, OFFSET FLAT:.scanf_format_string
        call    scanf
        mov     rdx, QWORD PTR [rbp-8]
        mov     rax, QWORD PTR [rbp-16]
        add     rax, rdx
        mov     rsp, rbp
        pop     rbp
        ret
