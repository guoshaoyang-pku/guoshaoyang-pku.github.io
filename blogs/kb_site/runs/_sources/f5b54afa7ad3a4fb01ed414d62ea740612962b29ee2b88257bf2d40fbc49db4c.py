#!/usr/bin/env python3
"""Restore frozen non-text evidence from assets.json without network or training."""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
MAX_BUNDLE = 24 * 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024
MAX_FILE = 2 * 1024 * 1024
ALLOWED_SUFFIXES = {".npz", ".png", ".pdf", ".txt"}


def restore():
    bundle_path = HERE / "assets.json"
    if bundle_path.is_symlink() or bundle_path.stat().st_size > MAX_BUNDLE:
        raise ValueError("unexpected asset bundle")
    bundle = json.loads(bundle_path.read_bytes())
    if bundle.get("schema") != "frozen-audit-assets-v1":
        raise ValueError("unsupported asset schema")
    entries = bundle["files"]
    if not isinstance(entries, list) or not 0 < len(entries) <= 512:
        raise ValueError("unexpected asset count")
    verified = []
    names = set()
    total = 0
    for entry in entries:
        name = entry["path"]
        rel = PurePosixPath(name)
        if (not isinstance(name, str) or rel.is_absolute() or str(rel) != name
                or "\\" in name or ":" in name or "\x00" in name
                or any(part in {"", ".", ".."} or part.startswith(".") for part in rel.parts)
                or rel.suffix not in ALLOWED_SUFFIXES or name in names):
            raise ValueError("unsafe or duplicate asset path")
        names.add(name)
        length = entry["bytes"]
        if not isinstance(length, int) or not 0 <= length <= MAX_FILE:
            raise ValueError("asset exceeds size bound")
        total += length
        if total > MAX_TOTAL:
            raise ValueError("bundle exceeds total size bound")
        encoded = entry["base64"]
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_FILE + 2) // 3):
            raise ValueError("unexpected encoded size")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != length or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("asset hash or length mismatch")
        target = HERE.joinpath(*rel.parts)
        for parent in [target, *target.parents]:
            if parent == HERE:
                break
            if parent.is_symlink():
                raise ValueError("asset path contains a symlink")
        if not target.resolve().is_relative_to(HERE):
            raise ValueError("asset path escapes this repo")
        if target.exists() and (not target.is_file() or target.read_bytes() != raw):
            raise ValueError("refusing to replace changed existing file: " + name)
        verified.append((target, raw))
    restored = 0
    for target, raw in verified:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        restored += 1
    print(f"verified {len(verified)} assets ({total} bytes); restored {restored}")


if __name__ == "__main__":
    restore()
