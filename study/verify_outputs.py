"""Verify the published study outputs against their SHA-256 provenance manifests.

Each notebook writes outputs/partN/logs/provenance_manifest.json. Every saved file
is listed with its SHA-256 hash, and the hashes form a chain:

    chain_0 = "0" * 64
    chain_k = sha256(chain_{k-1} + file_hash_k)

This script (1) recomputes each chain from the recorded hashes and compares it with
the manifest's final chain hash and with outputs/study_manifest.json, and (2)
re-hashes every file that is present and compares it with the last hash recorded
for that path. Patient-level files (outputs/partN/data) are never distributed and
are reported as such, not as failures.

Usage (from the study folder):

    python verify_outputs.py
    python verify_outputs.py --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(rel: str) -> Path:
    return Path(*rel.replace("\\", "/").split("/"))


def is_patient_level(rel: Path) -> bool:
    parts = [p.lower() for p in rel.parts]
    return "data" in parts or "cache" in parts


def verify_part(manifest_path: Path, base: Path, verbose: bool) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files", [])

    chain = "0" * 64
    chain_ok = True
    for entry in entries:
        chain = hashlib.sha256((chain + entry["sha256"]).encode()).hexdigest()
        if entry.get("chain_sha256") and entry["chain_sha256"] != chain:
            chain_ok = False
    chain_ok = chain_ok and chain == manifest.get("final_chain_sha256")

    last_hash = {}
    for entry in entries:
        last_hash[normalise(entry["file"])] = entry["sha256"]

    counts = {"matched": 0, "mismatched": 0, "missing": 0, "not_distributed": 0}
    problems = []
    for rel, expected in sorted(last_hash.items()):
        path = base / rel
        if is_patient_level(rel):
            counts["not_distributed"] += 1
            continue
        if not path.exists():
            counts["missing"] += 1
            problems.append(f"missing     {rel.as_posix()}")
            continue
        if sha256_file(path) == expected:
            counts["matched"] += 1
            if verbose:
                print(f"  ok          {rel.as_posix()}")
        else:
            counts["mismatched"] += 1
            problems.append(f"MISMATCH    {rel.as_posix()}")

    return {
        "notebook": manifest.get("notebook", manifest_path.parent.parent.name),
        "final_chain": manifest.get("final_chain_sha256"),
        "chain_ok": chain_ok,
        "counts": counts,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, default=HERE, help="folder that contains outputs/ (default: this folder)")
    parser.add_argument("--verbose", action="store_true", help="list every verified file")
    args = parser.parse_args()

    outputs = args.base / "outputs"
    manifests = sorted(outputs.glob("part*/logs/provenance_manifest.json"))
    if not manifests:
        print(f"No provenance manifests found under {outputs}")
        return 1

    study = {}
    study_path = outputs / "study_manifest.json"
    if study_path.exists():
        study = json.loads(study_path.read_text(encoding="utf-8")).get("notebooks", {})
    study_chains = {v.get("final_chain_sha256") for v in study.values()}

    all_ok = True
    for manifest_path in manifests:
        res = verify_part(manifest_path, args.base, args.verbose)
        in_study = (not study) or res["final_chain"] in study_chains
        ok = res["chain_ok"] and in_study and res["counts"]["mismatched"] == 0 and res["counts"]["missing"] == 0
        all_ok &= ok
        c = res["counts"]
        print(f"{manifest_path.parent.parent.name}: chain {'intact' if res['chain_ok'] else 'BROKEN'}"
              f"{'' if in_study else ' (not in study_manifest.json)'}; "
              f"{c['matched']} files verified, {c['mismatched']} changed, {c['missing']} missing, "
              f"{c['not_distributed']} patient-level files not distributed")
        for line in res["problems"]:
            print("   " + line)

    print("\nAll published outputs match their provenance records." if all_ok
          else "\nSome outputs differ from their provenance records (see above).")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
