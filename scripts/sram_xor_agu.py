import random
from pqc_reducer import Q_MLDSA, Q_MLKEM, unified_pqc_mul
from unified_ntt import bit_reverse, forward_ntt, inverse_ntt, mlkem_base_mul, ntt_poly_mul


def get_sram_bank_and_offset(logical_addr: int) -> tuple[int, int]:
    """
    Physical mapping for 256 words across 16 banks x 16 words/bank:
      - bank   = addr[3:0] ^ addr[7:4]
      - offset = addr[7:4]
    Bijective mapping: addr = (offset << 4) | (bank ^ offset).
    """
    bank = (logical_addr & 0xF) ^ (logical_addr >> 4)
    offset = logical_addr >> 4
    return bank, offset


def hw_agu(s: int, cyc: int, bfu: int) -> tuple[int, int]:
    """
    Hardware-friendly combinational AGU (replaces software 'visited' set).
      - s   in 0..7 : stage bit (stride = 1 << s)
      - cyc in 0..15: cycle counter within the stage
      - bfu in 0..7 : butterfly lane index
    """
    s_mod = s & 3
    b_bits = [b for b in range(4) if b != s_mod]
    used = set(b_bits) | {s}
    cyc_bits = [bit for bit in range(8) if bit not in used]

    idx_a = 0
    for i, pos in enumerate(cyc_bits):
        if (cyc >> i) & 1:
            idx_a |= (1 << pos)
    for i, pos in enumerate(b_bits):
        if (bfu >> i) & 1:
            idx_a |= (1 << pos)

    idx_b = idx_a | (1 << s)
    return idx_a, idx_b


def sram_load(poly: list[int]) -> list[list[int]]:
    sram = [[0] * 16 for _ in range(16)]
    for addr, val in enumerate(poly):
        bank, offset = get_sram_bank_and_offset(addr)
        sram[bank][offset] = val
    return sram


def sram_unload(sram: list[list[int]]) -> list[int]:
    poly = [0] * 256
    for addr in range(256):
        bank, offset = get_sram_bank_and_offset(addr)
        poly[addr] = sram[bank][offset]
    return poly


def sram_forward_ntt(poly: list[int], mode: str) -> list[int]:
    """Executes Forward NTT in-place on 16-bank SRAM using hw_agu."""
    if mode == "mldsa":
        modulus, zeta, stages, rev_bits = Q_MLDSA, 1753, range(7, -1, -1), 8
    else:
        modulus, zeta, stages, rev_bits = Q_MLKEM, 17, range(7, 0, -1), 7

    sram = sram_load(poly)

    for s in stages:
        for cyc in range(16):
            banks_used = []
            for bfu in range(8):
                idx_a, idx_b = hw_agu(s, cyc, bfu)
                bank_a, off_a = get_sram_bank_and_offset(idx_a)
                bank_b, off_b = get_sram_bank_and_offset(idx_b)
                banks_used.extend((bank_a, bank_b))

                k = (1 << (7 - s)) + (idx_a >> (s + 1))
                w = pow(zeta, bit_reverse(k, rev_bits), modulus)

                u = sram[bank_a][off_a]
                v = unified_pqc_mul(sram[bank_b][off_b], w, mode)
                add_res = u + v
                sub_res = u - v
                sram[bank_a][off_a] = add_res - modulus if add_res >= modulus else add_res
                sram[bank_b][off_b] = sub_res + modulus if sub_res < 0 else sub_res

            assert len(set(banks_used)) == 16, f"Bank conflict at forward s={s}, cyc={cyc}!"

    return sram_unload(sram)


def sram_inverse_ntt(poly_ntt: list[int], mode: str) -> list[int]:
    """Executes Inverse NTT in-place on 16-bank SRAM using hw_agu."""
    if mode == "mldsa":
        modulus, zeta, stages, rev_bits, inv_scale = (
            Q_MLDSA, 1753, range(0, 8), 8, pow(256, Q_MLDSA - 2, Q_MLDSA)
        )
    else:
        modulus, zeta, stages, rev_bits, inv_scale = (
            Q_MLKEM, 17, range(1, 8), 7, 3303
        )

    sram = sram_load(poly_ntt)

    for s in stages:
        for cyc in range(16):
            banks_used = []
            for bfu in range(8):
                idx_a, idx_b = hw_agu(s, cyc, bfu)
                bank_a, off_a = get_sram_bank_and_offset(idx_a)
                bank_b, off_b = get_sram_bank_and_offset(idx_b)
                banks_used.extend((bank_a, bank_b))

                k = (1 << (8 - s)) - 1 - (idx_a >> (s + 1))
                w = modulus - pow(zeta, bit_reverse(k, rev_bits), modulus)

                u = sram[bank_a][off_a]
                v = sram[bank_b][off_b]
                add_res = u + v
                diff = u - v
                if diff < 0:
                    diff += modulus
                sram[bank_a][off_a] = add_res - modulus if add_res >= modulus else add_res
                sram[bank_b][off_b] = unified_pqc_mul(diff, w, mode)

            assert len(set(banks_used)) == 16, f"Bank conflict at inverse s={s}, cyc={cyc}!"

    # Final scaling (16 reads/writes per cycle across all 16 banks)
    res = [unified_pqc_mul(x, inv_scale, mode) for x in sram_unload(sram)]
    return res


def sram_mlkem_basemul(a_ntt: list[int], b_ntt: list[int]) -> list[int]:
    """Executes ML-KEM BaseCaseMultiply using s=0 (pairs 2i, 2i+1) across 16 banks."""
    sram_a = sram_load(a_ntt)
    sram_b = sram_load(b_ntt)
    sram_c = [[0] * 16 for _ in range(16)]

    for cyc in range(16):
        banks_used = []
        for bfu in range(8):
            idx_a, idx_b = hw_agu(0, cyc, bfu)  # stride=1 -> (2i, 2i+1)
            bank_a, off_a = get_sram_bank_and_offset(idx_a)
            bank_b, off_b = get_sram_bank_and_offset(idx_b)
            banks_used.extend((bank_a, bank_b))

            pair_idx = idx_a >> 1
            gamma = pow(17, 2 * bit_reverse(pair_idx, 7) + 1, Q_MLKEM)
            c0, c1 = mlkem_base_mul(
                sram_a[bank_a][off_a], sram_a[bank_b][off_b],
                sram_b[bank_a][off_a], sram_b[bank_b][off_b],
                gamma,
            )
            sram_c[bank_a][off_a] = c0
            sram_c[bank_b][off_b] = c1

        assert len(set(banks_used)) == 16, f"Bank conflict at ML-KEM BaseMul cyc={cyc}!"

    return sram_unload(sram_c)


if __name__ == "__main__":
    # Verify bijection
    mapped = {get_sram_bank_and_offset(addr) for addr in range(256)}
    assert len(mapped) == 256

    # 1. ML-DSA full SRAM verification
    a_d = [random.randrange(Q_MLDSA) for _ in range(256)]
    assert sram_forward_ntt(a_d, "mldsa") == forward_ntt(a_d, "mldsa")
    assert sram_inverse_ntt(sram_forward_ntt(a_d, "mldsa"), "mldsa") == a_d
    print("[PASS] ML-DSA (8 stages): hw_agu + 16-bank SRAM exact match with unified_ntt.")

    # 2. ML-KEM full SRAM verification (7 NTT stages + BaseMul + INTT)
    a_k = [random.randrange(Q_MLKEM) for _ in range(256)]
    b_k = [random.randrange(Q_MLKEM) for _ in range(256)]
    a_k_ntt = sram_forward_ntt(a_k, "mlkem")
    b_k_ntt = sram_forward_ntt(b_k, "mlkem")
    assert a_k_ntt == forward_ntt(a_k, "mlkem")
    c_k_ntt = sram_mlkem_basemul(a_k_ntt, b_k_ntt)
    assert sram_inverse_ntt(c_k_ntt, "mlkem") == ntt_poly_mul(a_k, b_k, "mlkem")
    print("[PASS] ML-KEM (7 stages + BaseMul): hw_agu + 16-bank SRAM exact match with unified_ntt.")