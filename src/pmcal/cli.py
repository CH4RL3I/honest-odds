"""Command line interface: `pmcal fetch`, `pmcal analyze`."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SAMPLE = Path("data/sample.parquet")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pmcal", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download resolved markets + price history, build the dataset")
    f.add_argument("--out", type=Path, default=Path("data/processed/full.parquet"))
    f.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    f.add_argument("--min-volume", type=float, default=20_000.0, help="minimum lifetime USD volume")
    f.add_argument(
        "--min-lifetime-days", type=float, default=31.0, help="min days from start to resolution"
    )
    f.add_argument("--sample", type=int, default=None, help="random subsample of eligible markets")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--max-pages", type=int, default=None, help="cap Gamma pages (debugging)")

    a = sub.add_parser("analyze", help="run the analysis and write figures + docs/results.md")
    a.add_argument("--data", type=Path, default=DEFAULT_SAMPLE)
    a.add_argument("--out-dir", type=Path, default=Path("docs"))

    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        from .fetch import build_dataset

        df = build_dataset(
            args.raw_dir,
            args.min_volume,
            args.max_pages,
            args.min_lifetime_days,
            args.sample,
            args.seed,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["t_res_dt"], errors="ignore").to_parquet(
            args.out, index=False, compression="zstd"
        )
        print(f"wrote {args.out}: {len(df):,} rows, {df['market_id'].nunique():,} markets")
    else:
        from .analyze import run

        run(args.data, args.out_dir)


if __name__ == "__main__":
    main()
