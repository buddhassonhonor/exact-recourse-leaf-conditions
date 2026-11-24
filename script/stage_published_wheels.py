"""Stage pinned published-baseline wheels outside the project's main environment."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import requests


def main():
    defaults = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=defaults / "docs" / "published_baseline_wheels.json")
    parser.add_argument("--out", type=Path, default=defaults / "tmp" / "external" / "ocean")
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, record in entries.items():
        wheel = args.out.parent / record["filename"]
        payload = wheel.read_bytes() if wheel.exists() else None
        if payload is None or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            response = requests.get(record["url"], timeout=60)
            response.raise_for_status()
            payload = response.content
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"{name} wheel does not match the archived digest")
            wheel.write_bytes(payload)
        with ZipFile(wheel) as archive:
            archive.extractall(args.out)
        print(f"{name}: {record['filename']}")


if __name__ == "__main__":
    main()
