import random
from pqc_reducer import Q_MLDSA, Q_MLKEM, unified_pqc_mul


def bit_reverse(val: int, bits: int) -> int:
    res = 0
    for _ in range(bits):
        res = (res << 1) | (val & 1)
        val >>= 1
    return res


def forward_ntt(poly: list[int], mode: str) -> list[int]:
    """
    Cooley-Tukey (DIT) Negacyclic Forward NTT over Z_q[X]/(X^256 + 1):
      - 'mldsa': 8 stages (half_step 128 -> 1), NIST FIPS 204 Alg 41, zeta = 1753
      - 'mlkem': 7 stages (half_step 128 -> 2), NIST FIPS 203 Alg 9,  zeta = 17
    """
    a = poly.copy()
    n = len(a)
    assert n == 256, "Both ML-DSA and ML-KEM operate on N = 256"

    if mode == "mldsa":
        modulus, zeta, min_half_step, rev_bits = Q_MLDSA, 1753, 1, 8
    elif mode == "mlkem":
        modulus, zeta, min_half_step, rev_bits = Q_MLKEM, 17, 2, 7
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    k = 0
    half_step = n // 2
    while half_step >= min_half_step:
        step = half_step * 2
        for group in range(0, n, step):
            k += 1
            w = pow(zeta, bit_reverse(k, rev_bits), modulus)
            for j in range(half_step):
                u = a[group + j]
                v = unified_pqc_mul(a[group + j + half_step], w, mode)
                add_res = u + v
                a[group + j] = add_res - modulus if add_res >= modulus else add_res
                sub_res = u - v
                a[group + j + half_step] = sub_res + modulus if sub_res < 0 else sub_res
        half_step //= 2

    return a


def inverse_ntt(poly_ntt: list[int], mode: str) -> list[int]:
    """
    Gentleman-Sande (DIF) Negacyclic Inverse NTT over Z_q[X]/(X^256 + 1):
      - 'mldsa': 8 stages (half_step 1 -> 128), scales by 256^-1 mod q
      - 'mlkem': 7 stages (half_step 2 -> 128), scales by 128^-1 mod q (= 3303)
    """
    a = poly_ntt.copy()
    n = len(a)
    assert n == 256, "Both ML-DSA and ML-KEM operate on N = 256"

    if mode == "mldsa":
        modulus, zeta, half_step, k, rev_bits, inv_scale = (
            Q_MLDSA, 1753, 1, 256, 8, pow(256, Q_MLDSA - 2, Q_MLDSA)
        )
    elif mode == "mlkem":
        modulus, zeta, half_step, k, rev_bits, inv_scale = (
            Q_MLKEM, 17, 2, 128, 7, 3303  # 128^-1 mod 3329
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    while half_step < n:
        step = half_step * 2
        for group in range(0, n, step):
            k -= 1
            w = modulus - pow(zeta, bit_reverse(k, rev_bits), modulus)
            for j in range(half_step):
                u = a[group + j]
                v = a[group + j + half_step]
                add_res = u + v
                a[group + j] = add_res - modulus if add_res >= modulus else add_res
                diff = u - v
                if diff < 0:
                    diff += modulus
                a[group + j + half_step] = unified_pqc_mul(diff, w, mode)
        half_step *= 2

    for i in range(n):
        a[i] = unified_pqc_mul(a[i], inv_scale, mode)

    return a


def mlkem_base_mul(a0: int, a1: int, b0: int, b1: int, gamma: int) -> tuple[int, int]:
    """NIST FIPS 203 Algorithm 12: BaseCaseMultiply mod (X^2 - gamma)."""
    t0 = unified_pqc_mul(a0, b0, "mlkem")
    t1 = unified_pqc_mul(unified_pqc_mul(a1, b1, "mlkem"), gamma, "mlkem")
    c0 = t0 + t1
    if c0 >= Q_MLKEM:
        c0 -= Q_MLKEM

    u0 = unified_pqc_mul(a0, b1, "mlkem")
    u1 = unified_pqc_mul(a1, b0, "mlkem")
    c1 = u0 + u1
    if c1 >= Q_MLKEM:
        c1 -= Q_MLKEM

    return c0, c1


def ntt_poly_mul(a: list[int], b: list[int], mode: str) -> list[int]:
    """Polynomial multiplication in Z_q[X]/(X^256 + 1)."""
    a_ntt = forward_ntt(a, mode)
    b_ntt = forward_ntt(b, mode)

    if mode == "mldsa":
        c_ntt = [unified_pqc_mul(x, y, "mldsa") for x, y in zip(a_ntt, b_ntt)]
    elif mode == "mlkem":
        c_ntt = [0] * 256
        for i in range(128):
            gamma = pow(17, 2 * bit_reverse(i, 7) + 1, Q_MLKEM)
            c_ntt[2 * i], c_ntt[2 * i + 1] = mlkem_base_mul(
                a_ntt[2 * i], a_ntt[2 * i + 1], b_ntt[2 * i], b_ntt[2 * i + 1], gamma
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return inverse_ntt(c_ntt, mode)


def naive_negacyclic_mul(a: list[int], b: list[int], modulus: int) -> list[int]:
    n = len(a)
    res = [0] * n
    for i in range(n):
        for j in range(n):
            prod = (a[i] * b[j]) % modulus
            idx = i + j
            if idx >= n:
                res[idx - n] = (res[idx - n] - prod) % modulus
            else:
                res[idx] = (res[idx] + prod) % modulus
    return res


if __name__ == "__main__":
    # 1. ML-DSA (FIPS 204)
    a_d = [random.randrange(Q_MLDSA) for _ in range(256)]
    b_d = [random.randrange(Q_MLDSA) for _ in range(256)]
    assert inverse_ntt(forward_ntt(a_d, "mldsa"), "mldsa") == a_d
    assert ntt_poly_mul(a_d, b_d, "mldsa") == naive_negacyclic_mul(a_d, b_d, Q_MLDSA)
    print("[PASS] ML-DSA (FIPS 204, N=256, q=8380417): NTT round-trip + poly_mul exact match.")

    # 2. ML-KEM (FIPS 203)
    a_k = [random.randrange(Q_MLKEM) for _ in range(256)]
    b_k = [random.randrange(Q_MLKEM) for _ in range(256)]
    assert inverse_ntt(forward_ntt(a_k, "mlkem"), "mlkem") == a_k
    assert ntt_poly_mul(a_k, b_k, "mlkem") == naive_negacyclic_mul(a_k, b_k, Q_MLKEM)
    print("[PASS] ML-KEM (FIPS 203, N=256, q=3329):    NTT round-trip + BaseMul poly_mul exact match.")