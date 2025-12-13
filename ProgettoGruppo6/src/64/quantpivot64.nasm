default rel

section .text
global approx_distance_asm
global euclidean_distance_asm
global compute_lower_bound_asm

; ============================================================
; approx_distance_asm - VERSIONE AVX 64-BIT
;   RDI = vplus
;   RSI = vminus
;   RDX = wplus
;   RCX = wminus
;   R8D = D
;
; OTTIMIZZAZIONI AVX:
;   - Unrolling x4 (16 double/iterazione = 4 YMM registers)
;   - VMOVUPD per sicurezza (funziona anche con dati non allineati)
;   - Prefetching aggressivo
;   - Gestione robusta remainder (16 -> 8 -> 4 -> 1)
;
; Ritorno in XMM0 (double)
; ============================================================

approx_distance_asm:
    ; Reset accumulatori YMM (4 double ciascuno)
    vxorpd ymm0, ymm0, ymm0        ; sum_pp
    vxorpd ymm1, ymm1, ymm1        ; sum_mm
    vxorpd ymm2, ymm2, ymm2        ; sum_pm
    vxorpd ymm3, ymm3, ymm3        ; sum_mp

    ; Test edge case
    test r8d, r8d
    jz .return_zero

    ; ----------------------------------------
    ; Loop principale: 16 double/iterazione (4 YMM per array)
    ; ----------------------------------------
    mov eax, r8d
    shr eax, 4              ; eax = D/16
    jz .check8

.main_loop16:
    ; Prefetch aggressivo (512 byte = 64 double)
    prefetchnta [rdi + 512]
    prefetchnta [rsi + 512]
    prefetchnta [rdx + 512]
    prefetchnta [rcx + 512]

    ; ==== BLOCCO 1 (0-3) - 4 double ====
    vmovupd ymm4, [rdi]
    vmovupd ymm5, [rsi]
    vmovupd ymm6, [rdx]
    vmovupd ymm7, [rcx]

    vmulpd  ymm8, ymm4, ymm6
    vaddpd  ymm0, ymm0, ymm8

    vmulpd  ymm8, ymm5, ymm7
    vaddpd  ymm1, ymm1, ymm8

    vmulpd  ymm8, ymm4, ymm7
    vaddpd  ymm2, ymm2, ymm8

    vmulpd  ymm8, ymm5, ymm6
    vaddpd  ymm3, ymm3, ymm8

    ; ==== BLOCCO 2 (4-7) ====
    vmovupd ymm4, [rdi + 32]
    vmovupd ymm5, [rsi + 32]
    vmovupd ymm6, [rdx + 32]
    vmovupd ymm7, [rcx + 32]

    vmulpd  ymm8, ymm4, ymm6
    vaddpd  ymm0, ymm0, ymm8

    vmulpd  ymm8, ymm5, ymm7
    vaddpd  ymm1, ymm1, ymm8

    vmulpd  ymm8, ymm4, ymm7
    vaddpd  ymm2, ymm2, ymm8

    vmulpd  ymm8, ymm5, ymm6
    vaddpd  ymm3, ymm3, ymm8

    ; ==== BLOCCO 3 (8-11) ====
    vmovupd ymm4, [rdi + 64]
    vmovupd ymm5, [rsi + 64]
    vmovupd ymm6, [rdx + 64]
    vmovupd ymm7, [rcx + 64]

    vmulpd  ymm8, ymm4, ymm6
    vaddpd  ymm0, ymm0, ymm8

    vmulpd  ymm8, ymm5, ymm7
    vaddpd  ymm1, ymm1, ymm8

    vmulpd  ymm8, ymm4, ymm7
    vaddpd  ymm2, ymm2, ymm8

    vmulpd  ymm8, ymm5, ymm6
    vaddpd  ymm3, ymm3, ymm8

    ; ==== BLOCCO 4 (12-15) ====
    vmovupd ymm4, [rdi + 96]
    vmovupd ymm5, [rsi + 96]
    vmovupd ymm6, [rdx + 96]
    vmovupd ymm7, [rcx + 96]

    vmulpd  ymm8, ymm4, ymm6
    vaddpd  ymm0, ymm0, ymm8

    vmulpd  ymm8, ymm5, ymm7
    vaddpd  ymm1, ymm1, ymm8

    vmulpd  ymm8, ymm4, ymm7
    vaddpd  ymm2, ymm2, ymm8

    vmulpd  ymm8, ymm5, ymm6
    vaddpd  ymm3, ymm3, ymm8

    ; Avanza 16 double = 128 byte
    add rdi, 128
    add rsi, 128
    add rdx, 128
    add rcx, 128

    dec eax
    jnz .main_loop16

; ----------------------------------------
; Remainder: 8 double
; ----------------------------------------
.check8:
    mov eax, r8d
    shr eax, 3
    and eax, 1
    jz .check4

.main_loop8:
    ; Primo blocco di 4
    vmovupd ymm4, [rdi]
    vmovupd ymm5, [rsi]
    vmovupd ymm6, [rdx]
    vmovupd ymm7, [rcx]

    vmulpd  ymm8, ymm4, ymm6
    vaddpd  ymm0, ymm0, ymm8

    vmulpd  ymm8, ymm5, ymm7
    vaddpd  ymm1, ymm1, ymm8

    vmulpd  ymm8, ymm4, ymm7
    vaddpd  ymm2, ymm2, ymm8

    vmulpd  ymm8, ymm5, ymm6
    vaddpd  ymm3, ymm3, ymm8

    ; Secondo blocco di 4
    vmovupd ymm4, [rdi + 32]
    vmovupd ymm5, [rsi + 32]
    vmovupd ymm6, [rdx + 32]
    vmovupd ymm7, [rcx + 32]

    vmulpd  ymm8, ymm4, ymm6
    vaddpd  ymm0, ymm0, ymm8

    vmulpd  ymm8, ymm5, ymm7
    vaddpd  ymm1, ymm1, ymm8

    vmulpd  ymm8, ymm4, ymm7
    vaddpd  ymm2, ymm2, ymm8

    vmulpd  ymm8, ymm5, ymm6
    vaddpd  ymm3, ymm3, ymm8

    add rdi, 64
    add rsi, 64
    add rdx, 64
    add rcx, 64

; ----------------------------------------
; Remainder: 4 double
; ----------------------------------------
.check4:
    mov eax, r8d
    shr eax, 2
    and eax, 1
    jz .check1

.main_loop4:
    vmovupd ymm4, [rdi]
    vmovupd ymm5, [rsi]
    vmovupd ymm6, [rdx]
    vmovupd ymm7, [rcx]

    vmulpd ymm8, ymm4, ymm6
    vaddpd ymm0, ymm0, ymm8

    vmulpd ymm8, ymm5, ymm7
    vaddpd ymm1, ymm1, ymm8

    vmulpd ymm8, ymm4, ymm7
    vaddpd ymm2, ymm2, ymm8

    vmulpd ymm8, ymm5, ymm6
    vaddpd ymm3, ymm3, ymm8

    add rdi, 32
    add rsi, 32
    add rdx, 32
    add rcx, 32

; ----------------------------------------
; Remainder: 1-3 double (usa XMM, non YMM)
; ----------------------------------------
.check1:
    mov eax, r8d
    and eax, 3
    jz .reduce_all

.remainder_loop:
    vmovsd xmm4, [rdi]
    vmovsd xmm5, [rsi]
    vmovsd xmm6, [rdx]
    vmovsd xmm7, [rcx]

    vmulsd xmm8, xmm4, xmm6
    vaddsd xmm0, xmm0, xmm8

    vmulsd xmm8, xmm5, xmm7
    vaddsd xmm1, xmm1, xmm8

    vmulsd xmm8, xmm4, xmm7
    vaddsd xmm2, xmm2, xmm8

    vmulsd xmm8, xmm5, xmm6
    vaddsd xmm3, xmm3, xmm8

    add rdi, 8
    add rsi, 8
    add rdx, 8
    add rcx, 8

    dec eax
    jnz .remainder_loop

; ============================================================
; RIDUZIONE ORIZZONTALE YMM -> XMM -> scalar
; ============================================================
.reduce_all:
    ; Estrai upper 128bit e somma con lower 128bit
    vextractf128 xmm8, ymm0, 1
    vaddpd xmm0, xmm0, xmm8
    
    vextractf128 xmm8, ymm1, 1
    vaddpd xmm1, xmm1, xmm8
    
    vextractf128 xmm8, ymm2, 1
    vaddpd xmm2, xmm2, xmm8
    
    vextractf128 xmm8, ymm3, 1
    vaddpd xmm3, xmm3, xmm8

    ; Ora abbiamo 2 double in ciascun XMM
    ; Riduzione orizzontale XMM (2 double -> 1)
    vhaddpd xmm0, xmm0, xmm0
    vhaddpd xmm1, xmm1, xmm1
    vhaddpd xmm2, xmm2, xmm2
    vhaddpd xmm3, xmm3, xmm3

    ; Risultato finale: sum_pp + sum_mm - sum_pm - sum_mp
    vaddsd xmm0, xmm0, xmm1
    vaddsd xmm2, xmm2, xmm3
    vsubsd xmm0, xmm0, xmm2

    vzeroupper
    ret

.return_zero:
    vzeroupper
    ret


; ============================================================
; euclidean_distance_asm - VERSIONE AVX 64-BIT
;   RDI = v
;   RSI = w
;   EDX = D
;
; OTTIMIZZAZIONI AVX:
;   - Unrolling x4 (16 double/iterazione)
;   - VMOVUPD per sicurezza
;   - Prefetching
;   - Sqrt finale ottimizzata
; ============================================================

euclidean_distance_asm:
    vxorpd ymm0, ymm0, ymm0        ; sum_sq = 0

    ; Test edge case
    test edx, edx
    jz .return_zero

    ; ----------------------------------------
    ; Loop principale: 16 double/iterazione
    ; ----------------------------------------
    mov eax, edx
    shr eax, 4              ; eax = D/16
    jz .check8

.main_loop16:
    prefetchnta [rdi + 512]
    prefetchnta [rsi + 512]

    ; Blocco 1
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rsi]
    vsubpd  ymm1, ymm1, ymm2
    vmulpd  ymm1, ymm1, ymm1
    vaddpd  ymm0, ymm0, ymm1

    ; Blocco 2
    vmovupd ymm1, [rdi + 32]
    vmovupd ymm2, [rsi + 32]
    vsubpd  ymm1, ymm1, ymm2
    vmulpd  ymm1, ymm1, ymm1
    vaddpd  ymm0, ymm0, ymm1

    ; Blocco 3
    vmovupd ymm1, [rdi + 64]
    vmovupd ymm2, [rsi + 64]
    vsubpd  ymm1, ymm1, ymm2
    vmulpd  ymm1, ymm1, ymm1
    vaddpd  ymm0, ymm0, ymm1

    ; Blocco 4
    vmovupd ymm1, [rdi + 96]
    vmovupd ymm2, [rsi + 96]
    vsubpd  ymm1, ymm1, ymm2
    vmulpd  ymm1, ymm1, ymm1
    vaddpd  ymm0, ymm0, ymm1

    add rdi, 128
    add rsi, 128

    dec eax
    jnz .main_loop16

; ----------------------------------------
; Remainder: 8 double
; ----------------------------------------
.check8:
    mov eax, edx
    shr eax, 3
    and eax, 1
    jz .check4

.main_loop8:
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rsi]
    vsubpd  ymm1, ymm1, ymm2
    vmulpd  ymm1, ymm1, ymm1
    vaddpd  ymm0, ymm0, ymm1

    vmovupd ymm1, [rdi + 32]
    vmovupd ymm2, [rsi + 32]
    vsubpd  ymm1, ymm1, ymm2
    vmulpd  ymm1, ymm1, ymm1
    vaddpd  ymm0, ymm0, ymm1

    add rdi, 64
    add rsi, 64

; ----------------------------------------
; Remainder: 4 double
; ----------------------------------------
.check4:
    mov eax, edx
    shr eax, 2
    and eax, 1
    jz .check1

.main_loop4:
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rsi]
    vsubpd  ymm1, ymm1, ymm2
    vmulpd  ymm1, ymm1, ymm1
    vaddpd  ymm0, ymm0, ymm1

    add rdi, 32
    add rsi, 32

; ----------------------------------------
; Remainder: 1-3 double
; ----------------------------------------
.check1:
    mov eax, edx
    and eax, 3
    jz .reduce

.remainder_loop:
    vmovsd xmm1, [rdi]
    vmovsd xmm2, [rsi]
    vsubsd xmm1, xmm1, xmm2
    vmulsd xmm1, xmm1, xmm1
    vaddsd xmm0, xmm0, xmm1

    add rdi, 8
    add rsi, 8

    dec eax
    jnz .remainder_loop

; ============================================================
; RIDUZIONE E SQRT
; ============================================================
.reduce:
    ; Riduzione YMM -> XMM
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    
    ; Riduzione XMM (2 double -> 1)
    vhaddpd xmm0, xmm0, xmm0

    ; Sqrt finale
    vsqrtsd xmm0, xmm0, xmm0

    vzeroupper
    ret

.return_zero:
    vzeroupper
    ret


; ============================================================
; compute_lower_bound_asm - VERSIONE AVX 64-BIT
;   RDI = idx_v (array di h double)
;   RSI = qpivot (array di h double)
;   EDX = h
;
; Calcola: LB = max_j |idx_v[j] - qpivot[j]|
; ============================================================

compute_lower_bound_asm:
    ; Test edge case
    test edx, edx
    jz .return_zero

    ; Maschera per valore assoluto (azzera bit segno)
    mov rax, 0x7FFFFFFFFFFFFFFF
    vmovq xmm7, rax
    vbroadcastsd ymm7, xmm7    ; broadcast a tutti i 4 double in YMM

    vxorpd ymm0, ymm0, ymm0        ; max_LB = 0

    ; ----------------------------------------
    ; Loop principale: 4 double alla volta
    ; ----------------------------------------
    mov eax, edx
    shr eax, 2              ; eax = h/4
    jz .check1

.main_loop4:
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rsi]
    vsubpd  ymm1, ymm1, ymm2       ; diff
    vandpd  ymm1, ymm1, ymm7       ; |diff|
    vmaxpd  ymm0, ymm0, ymm1       ; max element-wise

    add rdi, 32
    add rsi, 32

    dec eax
    jnz .main_loop4

; ----------------------------------------
; Remainder: 1-3 elementi
; ----------------------------------------
.check1:
    mov eax, edx
    and eax, 3
    jz .reduce

.remainder_loop:
    vmovsd xmm1, [rdi]
    vmovsd xmm2, [rsi]
    vsubsd xmm1, xmm1, xmm2
    vandpd xmm1, xmm1, xmm7        ; abs
    vmaxsd xmm0, xmm0, xmm1

    add rdi, 8
    add rsi, 8

    dec eax
    jnz .remainder_loop

; ============================================================
; RIDUZIONE ORIZZONTALE (trova max tra i 4 elementi YMM)
; ============================================================
.reduce:
    ; Estrai upper 128bit e fai max con lower 128bit
    vextractf128 xmm1, ymm0, 1
    vmaxpd xmm0, xmm0, xmm1
    
    ; Ora abbiamo 2 double in XMM, trova il max
    vpermilpd xmm1, xmm0, 1        ; swap i due double
    vmaxsd xmm0, xmm0, xmm1

    vzeroupper
    ret

.return_zero:
    vzeroupper
    ret