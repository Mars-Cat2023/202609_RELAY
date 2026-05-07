#!/usr/bin/env python3
"""Pair FASTA files under by_length/ with PDBs under esmfold_pdb/<fasta_stem>/.

Computes metrics available **without** re-running ESMFold:
  - Mean pLDDT per structure (CA atom B-factors in PDB)
  - Corpus mean pLDDT (weighted by number of sequences)
  - Sequence diversity (% unique sequences, across all parsed FASTAs)
  - Pooled amino-acid entropy (Shannon, nats) over all residues

Does **not** compute pTM, pAE, or PAPL foldability % (those need model outputs;
extend dplm/analysis/cal_plddt_dir.py to log jsonl).

Layout must match dplm ``cal_plddt_dir.py``:
  esmfold_pdb/<stem>/SEQUENCE_x_plddt_<float>.pdb
where <stem> is basename(fasta) with ``.fasta`` removed.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple


def read_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    desc: str | None = None
    seq_chunks: List[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if desc is not None:
                    yield desc, "".join(seq_chunks)
                desc = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if desc is not None:
        yield desc, "".join(seq_chunks)


def mean_plddt_ca_from_pdb(pdb_path: Path) -> float:
    """Mean B-factor over CA atoms (ESMFold stores pLDDT there)."""
    vals: List[float] = []
    with pdb_path.open() as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 66:
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            vals.append(float(line[60:66]))
    if not vals:
        raise ValueError(f"no CA atoms with B-factors in {pdb_path}")
    return sum(vals) / len(vals)


def find_pdb_for_header(pdb_subdir: Path, header: str) -> Path | None:
    if not pdb_subdir.is_dir():
        return None
    prefix = f"{header}_plddt_"
    for p in pdb_subdir.iterdir():
        if p.suffix.lower() == ".pdb" and p.name.startswith(prefix):
            return p
    return None


def shannon_nats(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return h


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fasta-dir",
        type=Path,
        default=here / "by_length",
        help="Directory containing .fasta files (default: toy_fasta_eval/by_length)",
    )
    p.add_argument(
        "--pdb-root",
        type=Path,
        default=here / "esmfold_pdb",
        help="Parent of per-fasta PDB dirs (default: toy_fasta_eval/esmfold_pdb)",
    )
    args = p.parse_args()

    fasta_paths = sorted(Path(args.fasta_dir).glob("*.fasta"))
    if not fasta_paths:
        print(f"No .fasta files in {args.fasta_dir}")
        return 1

    all_seqs: List[str] = []
    aa_counts: Counter = Counter()
    paired_plddts: List[Tuple[str, str, float]] = []
    missing: List[Tuple[str, str]] = []

    for fasta_path in fasta_paths:
        stem = fasta_path.name[:-6] if fasta_path.name.endswith(".fasta") else fasta_path.stem
        pdb_subdir = Path(args.pdb_root) / stem
        print(f"\n== {fasta_path.name}  ->  {pdb_subdir}/")

        for header, seq in read_fasta(fasta_path):
            all_seqs.append(seq)
            aa_counts.update(seq.upper())
            pdb_p = find_pdb_for_header(pdb_subdir, header)
            if pdb_p is None:
                missing.append((str(fasta_path), header))
                print(f"  MISSING PDB  {header}")
                continue
            m = mean_plddt_ca_from_pdb(pdb_p)
            paired_plddts.append((fasta_path.name, header, m))
            print(f"  {header}  mean_pLDDT_CA={m:.2f}  <- {pdb_p.name}")

    print("\n=== Summary ===")
    if paired_plddts:
        avg = sum(t[2] for t in paired_plddts) / len(paired_plddts)
        print(f"Structures with PDB: {len(paired_plddts)}")
        print(f"Mean pLDDT (unweighted over structures): {avg:.2f}")
    if missing:
        print(f"Missing PDBs: {len(missing)}")

    n = len(all_seqs)
    if n:
        uniq = len(set(all_seqs))
        print(f"Sequences (FASTA): {n}")
        print(f"Unique sequences: {uniq}  ({100.0 * uniq / n:.2f}% diversity)")

    # Restrict entropy to standard amino acids (paper-scale proteins)
    standard = "ACDEFGHIKLMNPQRSTVWY"
    aa20 = Counter({k: aa_counts.get(k, 0) for k in standard})
    h = shannon_nats(aa20)
    print(f"Pooled AA entropy (20 standard, nats): {h:.4f}")

    print(
        "\nNote: PAPL pTM, pAE, and foldability (pLDDT>80 & pTM>0.7 & pAE<10) "
        "need tensors from ESMFold inference — log them in cal_plddt_dir.py or jsonl."
    )
    return 0 if not missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
