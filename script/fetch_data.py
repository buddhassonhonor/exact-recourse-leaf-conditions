"""Fetch the recorded public inputs into the layout used by the benchmarks."""

import argparse
import csv
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests

DOCS = Path(__file__).resolve().parents[1] / "docs"
UCI = "https://archive.ics.uci.edu/static/public"
BIKE_MIRROR = "https://huggingface.co/datasets/t22000t/bike-sharing-tabular/resolve/main/hour.csv"
BANK_ZIP = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"


def verified(data, expected):
    return hashlib.sha256(data).hexdigest() == expected


def ensure_file(session, target, expected, urls):
    if target.is_file() and verified(target.read_bytes(), expected):
        return
    problems = []
    for url, member in urls:
        try:
            response = session.get(url, timeout=(15, 90))
            response.raise_for_status()
            data = response.content
            if member:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    matches = [n for n in archive.namelist() if Path(n).name == member]
                    if len(matches) != 1:
                        raise ValueError(f"archive does not contain one {member}")
                    data = archive.read(matches[0])
            for variant in (data, data.replace(b"\r\n", b"\n"),
                            data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")):
                if verified(variant, expected):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(variant)
                    return
            problems.append(f"{url}: source checksum differs")
        except (requests.RequestException, zipfile.BadZipFile, ValueError) as error:
            problems.append(f"{url}: {error}")
    raise RuntimeError(f"{target}: " + "; ".join(problems))


def fetch_openml(session, root, ids):
    with (DOCS / "datasets.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = {int(info["dataset"]): info for info in csv.DictReader(handle)}
    for dataset in ids or metadata:
        entry = metadata[dataset]
        target = root / Path(entry["cached_file"].replace("\\", "/"))
        endpoint = f"https://www.openml.org/api/v1/xml/data/{dataset}"
        response = session.get(endpoint, timeout=(15, 60))
        response.raise_for_status()
        description = {
            element.tag.rsplit("}", 1)[-1]: element.text
            for element in ET.fromstring(response.content)
        }
        if description.get("version") != entry["version"]:
            raise RuntimeError(f"OpenML version differs for {dataset}")
        ensure_file(session, target, entry["sha256"], [(description["parquet_url"], None)])
        (target.parent / "description.xml").write_bytes(response.content)
        print(f"OpenML {dataset} verified")


def fetch_uci(session, root):
    seoul_source = (DOCS / "seoul_source.json").read_bytes()
    seoul = json.loads(seoul_source)
    sources = (
        ("bank/bank-full.csv", "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d",
         [(BANK_ZIP, "bank-full.csv"), (f"{UCI}/222/bank+marketing.zip", "bank-full.csv")]),
        ("bike_sharing/hour.csv", "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f",
         [(f"{UCI}/275/bike+sharing+dataset.zip", "hour.csv"), (BIKE_MIRROR, None)]),
        ("seoul_bike_sharing/SeoulBikeData.csv", seoul["sha256"],
         [(f"{UCI}/560/seoul+bike+sharing+demand.zip", "SeoulBikeData.csv"), (seoul["source_url"], None)]),
    )
    for relative, checksum, urls in sources:
        ensure_file(session, root / relative, checksum, urls)
        print(f"{relative} verified")
    (root / "seoul_bike_sharing" / "source.json").write_bytes(seoul_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--group", choices=("all", "openml", "uci"), default="all")
    parser.add_argument("--id", nargs="+", type=int, help="Fetch only these OpenML identifiers.")
    args = parser.parse_args()
    with requests.Session() as session:
        if args.group in ("all", "openml"):
            fetch_openml(session, args.data_root, args.id)
        if args.group in ("all", "uci"):
            fetch_uci(session, args.data_root)


if __name__ == "__main__":
    main()
