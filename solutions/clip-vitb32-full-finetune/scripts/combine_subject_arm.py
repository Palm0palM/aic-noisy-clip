"""Combine the ten production views with the subject-crop view.

Writes two CSVs from the same probability dumps:
  B          mean of the ten views
  B_subject  mean of the ten views plus the subject view (11 views). Rows whose
             localisation fell back carry no subject view and keep the ten-view
             result, so the extra view never fabricates a crop.

    python scripts/combine_subject_arm.py \
        --views-npz artifacts/subject_test_v31/test_view_probs.npz \
        --subject-npz artifacts/subject_test_v31/subject_view_probs.npz \
        --out-dir artifacts/subject_test_v31/arms
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

VIEWS10 = [
    "center:512:1.0", "flip:512:1.0", "center:512:1.14", "flip:512:1.14",
    "center:512:1.28", "center:512:1.4",
    "tl:512:1.14", "tr:512:1.14", "bl:512:1.14", "br:512:1.14",
]


def write_csv(path: Path, files, labels) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for file_name, label in zip(files, labels):
            handle.write(f"{file_name},{int(label):04d}\n")
    with zipfile.ZipFile(path.with_name(f"pred_{path.stem}.zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, arcname="pred_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-npz", required=True)
    parser.add_argument("--subject-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    views = np.load(args.views_npz, allow_pickle=False)
    files = [str(f) for f in views["files"]]
    if not files or len(set(files)) != len(files):
        raise SystemExit("Input files must be nonempty and unique")
    missing = [v for v in VIEWS10 if v not in views.files]
    if missing:
        raise SystemExit(f"missing views: {missing}")
    ten = np.mean(np.stack([views[v].astype(np.float32) for v in VIEWS10]), axis=0)

    subject = np.load(args.subject_npz, allow_pickle=False)
    subject_files = [str(f) for f in subject["files"]]
    if subject_files != files:
        raise SystemExit("subject dump and view dump list different test files")
    subject_probs = subject["probs"].astype(np.float32)
    fallback = subject["fallback"].astype(bool)
    if subject_probs.shape != ten.shape or fallback.shape != (len(files),):
        raise SystemExit("Probability shapes or fallback mask do not match")
    if not np.isfinite(ten).all() or not np.isfinite(subject_probs).all():
        raise SystemExit("Non-finite probabilities")
    if (ten < 0).any() or (subject_probs < 0).any():
        raise SystemExit("Negative probabilities")
    if not np.all(subject_probs[fallback] == 0):
        raise SystemExit("Fallback rows must contain zero subject vectors")
    if not np.allclose(subject_probs[~fallback].sum(1), 1, atol=1e-3):
        raise SystemExit("Used subject rows are not probability distributions")
    reports = []
    for dump_path, report_name in [(args.views_npz, "report.json"), (args.subject_npz, "subject_view_report.json")]:
        report_path = Path(dump_path).parent / report_name
        reports.append(json.loads(report_path.read_text()) if report_path.exists() else {})
    same_checkpoint = all(r.get("checkpoint_sha256") for r in reports)
    if same_checkpoint and (reports[0]["checkpoint_sha256"] != reports[1]["checkpoint_sha256"] or reports[0].get("weights") != reports[1].get("weights")):
        raise SystemExit("Both arms must use exactly the same checkpoint and weight state")

    eleven = ten.copy()
    used = ~fallback
    eleven[used] = (ten[used] * len(VIEWS10) + subject_probs[used]) / (len(VIEWS10) + 1)
    if not np.array_equal(eleven[fallback], ten[fallback]):
        raise RuntimeError("Fallback changed the baseline")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "B.csv").exists() or (out / "B_subject.csv").exists():
        raise SystemExit("Refusing to overwrite existing arms; use a new output directory")
    write_csv(out / "B.csv", files, ten.argmax(1))
    write_csv(out / "B_subject.csv", files, eleven.argmax(1))

    agreement = float((ten.argmax(1) == eleven.argmax(1)).mean())
    changed = int((ten.argmax(1) != eleven.argmax(1)).sum())
    report = {"images": len(files), "subject_used": int(used.sum()), "fallback": int(fallback.sum()),
              "changed": changed, "checkpoint_identity_verified": bool(same_checkpoint),
              "view_dtypes": {v: str(views[v].dtype) for v in VIEWS10},
              "subject_dtype": str(subject["probs"].dtype),
              "sources": [str(args.views_npz), str(args.subject_npz)]}
    (out / "combine_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[combine] {len(files)} images, subject view used on {int(used.sum())} "
          f"({used.mean():.1%}), fallback {int(fallback.sum())}")
    print(f"[combine] predictions changed on {changed} rows ({changed / len(files):.3%}), "
          f"agreement {agreement:.5f}")
    print(f"[combine] wrote {out / 'B.csv'} and {out / 'B_subject.csv'}", flush=True)


if __name__ == "__main__":
    main()
