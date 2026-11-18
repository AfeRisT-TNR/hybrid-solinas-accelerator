# hybrid-solinas-accelerator

Polynomial arithmetic reference model and WIP RTL for NIST FIPS 204 (ML-DSA) and FIPS 203 (ML-KEM).

Both standards work over R_q = Z_q[X]/(X^256 + 1), so they can share the same 16-bank SRAM layout, XOR address generator, and butterfly control logic. Originally started as an ML-DSA + BabyBear experiment, pivoted to ML-DSA + ML-KEM since pairing signature and KEM hardware on N=256 makes more sense for small PQC chips.

## Parameters

| | ML-DSA (FIPS 204) | ML-KEM (FIPS 203) |
|---|---|---|
| q | 8380417 (2^23 - 2^13 + 1, 23-bit) | 3329 (13 * 2^8 + 1, 12-bit) |
| N | 256 | 256 |
| zeta | 1753 (512th root) | 17 (256th root) |
| NTT stages | 8 (complete, half_step 128..1) | 7 (incomplete, half_step 128..2) |
| Pointwise | 256 scalar muls | 128 degree-1 BaseCaseMultiply |
| INTT scale | 256^-1 mod q = 8347681 | 128^-1 mod q = 3303 |

## Repo layout

* `pqc_reducer.py` - shift-add modular multipliers (no division / Barrett)
* `unified_ntt.py` - CT-DIT forward NTT, GS-DIF inverse NTT, ML-KEM BaseCaseMultiply
* `sram_xor_agu.py` - combinational XOR-AGU and 16x16 SRAM bank simulator

Run tests:

```bash
python3 pqc_reducer.py
python3 unified_ntt.py
python3 sram_xor_agu.py
```

## Notes on the math / hardware mapping

### Modular reduction (`pqc_reducer.py`)

* ML-DSA: `2^23 == 2^13 - 1 (mod q)`, folds as `z_lo + (z_hi << 13) - z_hi` (8 folds, then one conditional `-q`).
* ML-KEM: `2^12 == 2^9 + 2^8 - 1 (mod q)`, folds as `z_lo + (z_hi << 9) + (z_hi << 8) - z_hi` (7 folds into `[0, 2q)`, then `-q`). Always stays unsigned.

### NTT and 16-bank XOR-AGU (`sram_xor_agu.py`)

Forward pass uses Cooley-Tukey (DIT) and inverse uses Gentleman-Sande (DIF), so bit-reversal cancels out without a separate permutation pass.

For 8 parallel butterflies (16 reads/cycle), a 256-entry polynomial is split into 16 banks x 16 words:

```text
bank   = addr[3:0] ^ addr[7:4]
offset = addr[7:4]
```

At stage `s` (stride `2^s`), `hw_agu(s, cyc, bfu)` maps the 3 BFU bits onto `{0..3} \ {s mod 4}` and the 4 cycle counter bits onto the remaining address bits. All 16 bank indices per cycle stay distinct across every stage of ML-DSA (`s = 7..0`), ML-KEM (`s = 7..1`), and ML-KEM BaseMul pair reads (`s = 0`).

Because the upper address bits stay fixed within a cycle for `s >= 3`, all 8 BFUs share 1 twiddle factor per cycle on stages 3..7 (2 on stage 2, 4 on stage 1, 8 on stage 0).

## TODO

* Add Verilator / Icarus testbench against `pqc_reducer.py`.
* Build out the 8-BFU array, XOR-AGU FSM, and 16x16 crossbar in SystemVerilog, verify against `sram_xor_agu.py`.
* Run Yosys/Vivado synthesis for real area/LUT/Fmax numbers.
* Eventually hook up a shared Keccak (SHAKE-128/256) core.
