"""Build data/manifest.csv from a Celeb-DF v2 folder (Celeb-real, YouTube-real, Celeb-synthesis).

Usage: python build_manifest.py --root D:\\Celeb-DF-v2 --n-real 100 --n-fake 100
Person id = first idX in the filename (id0_0001.mp4 -> id0; id0_id16_0003.mp4 -> id0), so all clips of a
person, real or fake, stay on one side of the split. YouTube-real clips have no identity -> own group.
Also works for any folder pair:  --real-dir X --fake-dir Y
"""
from __future__ import annotations
import argparse, glob, json, os, random, re
import pandas as pd

def person_of(fn):
    m = re.match(r"(id\d+)", os.path.basename(fn))
    if m: return m.group(1)
    # e.g. UADFV: 0047.mp4 (real) and 0047_fake.mp4 (fake) share group 0047
    return re.sub(r"_fake$", "", os.path.splitext(os.path.basename(fn))[0])

def vids(d):
    return sorted(sum([glob.glob(os.path.join(d, e)) for e in ("*.mp4", "*.avi", "*.mov", "*.mkv")], []))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--real-dir", nargs="*"); ap.add_argument("--fake-dir", nargs="*")
    ap.add_argument("--dfdc-json", help="DFDC metadata.json (use with --dfdc-dir)")
    ap.add_argument("--dfdc-dir", help="folder holding the DFDC .mp4 files")
    ap.add_argument("--n-real", type=int, default=100); ap.add_argument("--n-fake", type=int, default=100)
    ap.add_argument("--out", default="data/manifest.csv"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.dfdc_json:
        meta = json.load(open(a.dfdc_json))
        stem = lambda f: os.path.splitext(f)[0]
        r, fk = [], []
        for fn, m in meta.items():
            p = os.path.join(a.dfdc_dir, fn)
            if not os.path.exists(p): continue
            # a fake shares its group with the real video it was made from
            g = stem(m["original"]) if m.get("label") == "FAKE" and m.get("original") else stem(fn)
            (fk if m.get("label") == "FAKE" else r).append((p, g))
        if not r or not fk:
            raise SystemExit(f"No usable DFDC videos found (real={len(r)}, fake={len(fk)}). Check --dfdc-dir.")
        random.seed(a.seed); random.shuffle(r); random.shuffle(fk)
        rows = [(p, "real", g, g) for p, g in r[:a.n_real]] + [(p, "fake", g, g) for p, g in fk[:a.n_fake]]
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        df = pd.DataFrame(rows, columns=["path", "label", "person", "source"]); df.to_csv(a.out, index=False)
        print(f"{len(df)} videos ({(df.label=='real').sum()} real / {(df.label=='fake').sum()} fake) -> {a.out}")
        return
    real_dirs = a.real_dir or [os.path.join(a.root, "Celeb-real"), os.path.join(a.root, "YouTube-real")]
    fake_dirs = a.fake_dir or [os.path.join(a.root, "Celeb-synthesis")]
    real = sum([vids(d) for d in real_dirs], []); fake = sum([vids(d) for d in fake_dirs], [])
    if not real or not fake:
        raise SystemExit(f"No videos found (real={len(real)}, fake={len(fake)}).\n"
                         f"Looked in: {real_dirs + fake_dirs}\n"
                         "Celeb-DF is not downloaded/unzipped there, or the path is wrong. Nothing written.")
    random.seed(a.seed); random.shuffle(real); random.shuffle(fake)
    rows = [(p, "real", person_of(p), person_of(p)) for p in real[:a.n_real]] + \
           [(p, "fake", person_of(p), person_of(p)) for p in fake[:a.n_fake]]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df = pd.DataFrame(rows, columns=["path", "label", "person", "source"]); df.to_csv(a.out, index=False)
    print(f"{len(df)} videos ({(df.label=='real').sum()} real / {(df.label=='fake').sum()} fake), "
          f"{df.person.nunique()} person groups -> {a.out}")
if __name__ == "__main__":
    main()
