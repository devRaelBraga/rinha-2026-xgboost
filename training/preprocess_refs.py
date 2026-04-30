#!/usr/bin/env python3
import json
import gzip
import numpy as np
import os
import argparse

def quantize(v):
    """
    Quantize float32 feature value to uint8.
    -1.0 maps to 255 (sentinel for NULL)
    [0.0, 1.0] maps to [0, 250]
    """
    if v == -1.0:
        return 255
    return int(round(v * 250))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="resources/references.json.gz")
    parser.add_argument("--output-dir", default="training/output")
    args = parser.parse_args()

    print(f"Loading references from {args.input}...")
    open_fn = gzip.open if args.input.endswith(".gz") else open
    with open_fn(args.input, "rt", encoding="utf-8") as f:
        refs = json.load(f)

    n = len(refs)
    print(f"Processing {n} vectors...")

    # 14 dimensions as per context.md
    dim = 14
    vectors_bin = np.zeros((n, dim), dtype=np.uint8)
    labels_bitset = bytearray((n + 7) // 8)

    for i, r in enumerate(refs):
        # Quantize vectors
        v = r["vector"]
        for j in range(dim):
            vectors_bin[i, j] = quantize(v[j])
        
        # Pack labels (1=fraud, 0=legit)
        if r["label"] == "fraud":
            byte_idx = i // 8
            bit_idx = i % 8
            labels_bitset[byte_idx] |= (1 << bit_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    
    vec_path = os.path.join(args.output_dir, "refs_vectors.bin")
    label_path = os.path.join(args.output_dir, "refs_labels.bin")

    print(f"Saving vectors to {vec_path} ({len(vectors_bin.tobytes())} bytes)...")
    with open(vec_path, "wb") as f:
        f.write(vectors_bin.tobytes())

    print(f"Saving labels to {label_path} ({len(labels_bitset)} bytes)...")
    with open(label_path, "wb") as f:
        f.write(labels_bitset)

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
