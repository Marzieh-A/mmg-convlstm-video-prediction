"""STAGE 1 - Forensic inspection of the raw Moving MNIST .npy file.

Checks, WITHOUT any forced reshape:
  file size, NumPy magic string, format version, header length,
  header dict, declared dtype, declared shape, expected vs actual
  payload size, header/payload consistency, truncation/corruption.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

RAW_PATH = Path("datasets/raw/moving_mnist/mnist_test_seq.npy")

EXPECTED_SHAPE = (20, 10000, 64, 64)  # [T, N, H, W] official Moving MNIST layout

MAGIC = b"\x93NUMPY"


def main() -> int:
    print("=" * 70)
    print("FORENSIC INSPECTION:", RAW_PATH)
    print("=" * 70)

    if not RAW_PATH.exists():
        print("FILE NOT FOUND - do not fabricate or re-download silently.")
        return 1

    size = RAW_PATH.stat().st_size
    print(f"file_size            : {size} bytes ({size / (1024**3):.3f} GB)")

    with RAW_PATH.open("rb") as f:
        magic = f.read(6)
        print(f"magic                : {magic!r}  (expected {MAGIC!r})")
        if magic != MAGIC:
            print("VERDICT: NOT a valid .npy file (bad magic) -> STOP")
            return 1

        ver = f.read(2)
        major, minor = ver[0], ver[1]
        print(f"format_version       : {major}.{minor}")

        if major == 1:
            hlen_raw = f.read(2)
            hlen = struct.unpack("<H", hlen_raw)[0]
        else:
            hlen_raw = f.read(4)
            hlen = struct.unpack("<I", hlen_raw)[0]
        print(f"header_length        : {hlen} bytes")

        header_bytes = f.read(hlen)
        try:
            header = header_bytes.decode("latin1").strip()
        except Exception as e:
            print(f"header decode failed : {e} -> STOP")
            return 1
        print(f"header               : {header}")

        data_start = 6 + 2 + (2 if major == 1 else 4) + hlen
        payload = size - data_start
        print(f"data_start_offset    : {data_start}")
        print(f"actual_payload_size  : {payload} bytes ({payload / (1024**3):.3f} GB)")

        # parse declared dtype / shape from header text
        import ast
        try:
            hdict = ast.literal_eval(header)
        except Exception as e:
            print(f"header parse failed  : {e} -> header is not a valid dict -> STOP")
            return 1
        dtype = hdict.get("descr")
        shape = hdict.get("shape")
        fortran = hdict.get("fortran_order")
        print(f"declared_dtype       : {dtype}")
        print(f"declared_shape       : {shape}")
        print(f"fortran_order        : {fortran}")

        if shape is None:
            print("VERDICT: header has no shape -> STOP")
            return 1

        import numpy as np
        try:
            itemsize = np.dtype(dtype).itemsize
        except Exception as e:
            print(f"invalid dtype        : {e} -> STOP")
            return 1

        n_elements = 1
        for d in shape:
            n_elements *= d
        expected_payload = n_elements * itemsize
        print(f"expected_payload     : {expected_payload} bytes "
              f"({expected_payload / (1024**3):.3f} GB)")

        if payload == expected_payload:
            print("payload_consistency  : OK (actual == expected)")
        elif payload < expected_payload:
            print(f"payload_consistency  : MISMATCH - file appears TRUNCATED "
                  f"(missing {expected_payload - payload} bytes = "
                  f"{(expected_payload - payload) / itemsize} elements)")
        else:
            print(f"payload_consistency  : MISMATCH - extra {payload - expected_payload} bytes")

        if tuple(shape) == EXPECTED_SHAPE and payload == expected_payload:
            print("shape_vs_expected    : OK  (20, 10000, 64, 64)")
        else:
            print(f"shape_vs_expected    : MISMATCH (expected {EXPECTED_SHAPE})")

        # sanity read of first few elements WITHOUT loading whole array
        f.seek(data_start)
        head = f.read(min(64, payload))
        print(f"first_bytes_hex      : {head[:32].hex()}")

    print("=" * 70)
    print("VERDICT: see payload_consistency and shape_vs_expected above.")
    print("If MISMATCH -> file is invalid; do NOT reshape, do NOT fabricate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())