import random
import time

Q_MLDSA = 8_380_417  # 2^23 - 2^13 + 1 (NIST FIPS 204 ML-DSA)
Q_MLKEM = 3_329      # 13 * 2^8 + 1   (NIST FIPS 203 ML-KEM)


def unified_pqc_mul(a: int, b: int, mode: str) -> int:
    """
    Bit-accurate hardware-friendly modular multiplier without division:
      - mode == 'mldsa': 8-fold Solinas reduction (2^23 == 2^13 - 1 mod q)
      - mode == 'mlkem': 7-fold shift-add reduction (2^12 == 2^9 + 2^8 - 1 mod q)
    """
    z = a * b

    if mode == "mldsa":
        mask = (1 << 23) - 1
        for _ in range(8):
            z_lo = z & mask
            z_hi = z >> 23
            z = z_lo + (z_hi << 13) - z_hi
        modulus = Q_MLDSA
    elif mode == "mlkem":
        mask = (1 << 12) - 1
        # 7 folds guarantee z in [0, 2*Q_MLKEM) for any 24-bit product z < 2^24
        for _ in range(7):
            z_lo = z & mask
            z_hi = z >> 12
            z = z_lo + (z_hi << 9) + (z_hi << 8) - z_hi
        modulus = Q_MLKEM
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if z >= modulus:
        z -= modulus

    return z


if __name__ == "__main__":
    iterations = 1_000_000
    print(f"Running verification across {iterations:,} random vectors per mode...")

    for a, b in [(0, 0), (1, 1), (Q_MLDSA - 1, Q_MLDSA - 1)]:
        assert unified_pqc_mul(a, b, "mldsa") == (a * b) % Q_MLDSA
    for a, b in [(0, 0), (1, 1), (Q_MLKEM - 1, Q_MLKEM - 1)]:
        assert unified_pqc_mul(a, b, "mlkem") == (a * b) % Q_MLKEM

    start_time = time.perf_counter()
    for _ in range(iterations):
        a_d = random.randrange(Q_MLDSA)
        b_d = random.randrange(Q_MLDSA)
        assert unified_pqc_mul(a_d, b_d, "mldsa") == (a_d * b_d) % Q_MLDSA

        a_k = random.randrange(Q_MLKEM)
        b_k = random.randrange(Q_MLKEM)
        assert unified_pqc_mul(a_k, b_k, "mlkem") == (a_k * b_k) % Q_MLKEM
    elapsed = time.perf_counter() - start_time

    print(f"[PASS] ML-DSA (q = {Q_MLDSA}): {iterations:,} vectors verified.")
    print(f"[PASS] ML-KEM (q = {Q_MLKEM}): {iterations:,} vectors verified.")
    print(f"Total benchmark time (2M reductions): {elapsed:.2f} sec.")
