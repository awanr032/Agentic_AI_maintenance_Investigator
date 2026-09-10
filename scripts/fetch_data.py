"""Fetch MaintIE's gold/silver datasets and taxonomy from the source repo into data/.

Source: https://github.com/nlp-tlp/maintie (MIT licensed). Files aren't vendored into
this repo (see .gitignore) — run this once after cloning.
"""
import pathlib
import requests

RAW_BASE = "https://raw.githubusercontent.com/nlp-tlp/maintie/main"
FILES = {
    "data/gold_release.json": f"{RAW_BASE}/data/gold_release.json",
    "data/silver_release.json": f"{RAW_BASE}/data/silver_release.json",
    "data/scheme.json": f"{RAW_BASE}/data/scheme.json",
}

def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    for rel_path, url in FILES.items():
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {url} -> {dest}")
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  WARNING: got {resp.status_code} — check the file still exists at "
                  f"this path/branch on github.com/nlp-tlp/maintie and adjust FILES above.")
            continue
        dest.write_bytes(resp.content)
        print(f"  OK ({len(resp.content)} bytes)")

if __name__ == "__main__":
    main()
