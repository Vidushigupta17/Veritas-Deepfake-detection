"""Pull a small balanced subset out of a big DFDC chunk zip WITHOUT unzipping all of it.

Usage:
  python extract_dfdc_subset.py --zip C:\\Downloads\\00.zip --out C:\\data\\DFDC --n-real 70 --n-fake 70
Writes the chosen .mp4 files plus a filtered metadata.json into --out. Then you can delete the big zip.
"""
from __future__ import annotations
import argparse, json, os, posixpath, random, zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-real", type=int, default=70); ap.add_argument("--n-fake", type=int, default=70)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with zipfile.ZipFile(a.zip) as z:
        names = z.namelist()
        metas = [n for n in names if posixpath.basename(n) == "metadata.json"]
        if not metas: raise SystemExit("No metadata.json inside this zip. Is it a DFDC chunk?")
        meta_path = metas[0]; prefix = posixpath.dirname(meta_path)
        meta = json.loads(z.read(meta_path))
        have = {posixpath.basename(n): n for n in names if n.lower().endswith(".mp4")}
        real = [f for f, m in meta.items() if m.get("label") == "REAL" and f in have]
        fake = [f for f, m in meta.items() if m.get("label") == "FAKE" and f in have]
        print(f"Zip has {len(real)} real and {len(fake)} fake videos (with files).")
        random.seed(a.seed); random.shuffle(real); random.shuffle(fake)
        pick = real[:a.n_real] + fake[:a.n_fake]
        if not pick: raise SystemExit("Nothing to extract.")
        sub = {}
        for i, f in enumerate(pick, 1):
            with z.open(have[f]) as src, open(os.path.join(a.out, f), "wb") as dst:
                dst.write(src.read())
            sub[f] = meta[f]
            if i % 20 == 0: print(f"  extracted {i}/{len(pick)}", flush=True)
    json.dump(sub, open(os.path.join(a.out, "metadata.json"), "w"))
    print(f"Done: {len(pick)} videos + metadata.json in {a.out}")
    print("Next: python build_manifest.py --dfdc-json {0}\\metadata.json --dfdc-dir {0} --n-real {1} --n-fake {2}"
          .format(a.out, a.n_real, a.n_fake))

if __name__ == "__main__":
    main()
