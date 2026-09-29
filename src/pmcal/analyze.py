"""Run the full analysis: tables, figures and docs/results.md."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import (  # noqa: E402
    brier,
    calibration_table,
    fit_recalibration,
    holm,
    log_loss,
    murphy_decomposition,
)
from .parse import HORIZONS_DAYS  # noqa: E402
from .strategy import Costs, run_strategy, time_split  # noqa: E402

MIN_CELL = 30  # minimum observations for a breakdown row to be reported
STRAT_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#6b7280"


def score_row(d: pd.DataFrame) -> dict:
    fit = fit_recalibration(d["price"], d["outcome"], groups=d["market_id"])
    mu = murphy_decomposition(d["price"], d["outcome"], edges=np.linspace(0, 1, 11))
    if abs(fit.slope) > 15:  # (quasi-)complete separation: MLE does not exist, report nothing
        fit = replace(
            fit,
            intercept=np.nan,
            slope=np.nan,
            se_slope=np.nan,
            p_slope_eq_1=np.nan,
            p_joint=np.nan,
        )
    return {
        "n": len(d),
        "base_rate": d["outcome"].mean(),
        "brier": brier(d["price"], d["outcome"]),
        "brier_climatology": d["outcome"].mean() * (1 - d["outcome"].mean()),
        "log_loss": log_loss(d["price"], d["outcome"]),
        "reliability": mu.reliability,
        "resolution": mu.resolution,
        "uncertainty": mu.uncertainty,
        "intercept": fit.intercept,
        "slope": fit.slope,
        "se_slope": fit.se_slope,
        "p_slope_eq_1": fit.p_slope_eq_1,
        "p_joint": fit.p_joint,
    }


def by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"horizon_days": h} | score_row(df[df.horizon_days == h])
        for h in HORIZONS_DAYS
        if (df.horizon_days == h).sum() >= MIN_CELL
    ]
    return pd.DataFrame(rows)


def breakdown(df: pd.DataFrame, horizon: int, col: str) -> pd.DataFrame:
    d = df[df.horizon_days == horizon]
    rows = []
    for key, g in d.groupby(col, observed=True):
        if len(g) < MIN_CELL or g["outcome"].nunique() < 2:
            continue
        rows.append({col: key} | score_row(g))
    out = pd.DataFrame(rows)
    if len(out):
        ok = out["p_slope_eq_1"].notna()
        out["p_holm"] = np.nan
        out.loc[ok, "p_holm"] = holm(out.loc[ok, "p_slope_eq_1"])
    return out


def add_volume_tercile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    per = df.drop_duplicates("market_id")
    q = per["volume"].quantile([1 / 3, 2 / 3]).to_numpy()
    lab = np.where(
        df["volume"] <= q[0], "T1 low", np.where(df["volume"] <= q[1], "T2 mid", "T3 high")
    )
    df["volume_tercile"] = lab
    return df


def favourite_longshot(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Realised frequency vs price in coarse buckets at the extremes."""
    d = df[df.horizon_days == horizon]
    edges = [0, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 1.0]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (d.price >= lo) & ((d.price < hi) if hi < 1 else (d.price <= hi))
        g = d[m]
        if len(g) < MIN_CELL:
            continue
        t = calibration_table(g["price"], g["outcome"], edges=[lo, hi])
        r = t.iloc[0]
        rows.append(
            {
                "bucket": f"[{lo:.2f}, {hi:.2f}{']' if hi == 1 else ')'}",
                "n": int(r.n),
                "mean_price": r.mean_price,
                "freq": r.freq,
                "ci_lo": r.ci_lo,
                "ci_hi": r.ci_hi,
                "gap": r.freq - r.mean_price,
            }
        )
    return pd.DataFrame(rows)


def fig_calibration(df: pd.DataFrame, path: Path) -> None:
    hs = [h for h in HORIZONS_DAYS if (df.horizon_days == h).sum() >= MIN_CELL]
    fig, axes = plt.subplots(1, len(hs), figsize=(4.6 * len(hs), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs, strict=True):
        d = df[df.horizon_days == h]
        t = calibration_table(d["price"], d["outcome"])
        ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="perfect calibration")
        ax.errorbar(
            t.mean_price,
            t.freq,
            yerr=[t.freq - t.ci_lo, t.ci_hi - t.freq],
            fmt="o-",
            color=BLUE,
            ecolor=BLUE,
            capsize=3,
            lw=1.4,
            ms=5,
            label="realised frequency (95% Wilson)",
        )
        fit = fit_recalibration(d["price"], d["outcome"], groups=d["market_id"])
        ax.set_title(
            f"{h} days before resolution\nN={len(d):,}, Brier={brier(d.price, d.outcome):.3f}, "
            f"slope={fit.slope:.2f} (SE {fit.se_slope:.2f})",
            fontsize=10,
        )
        ax.set_xlabel("market-implied probability (Yes price)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ins = ax.inset_axes([0.58, 0.08, 0.38, 0.16])
        ins.hist(d["price"], bins=20, range=(0, 1), color=GREY, alpha=0.6)
        ins.set_yticks([])
        ins.tick_params(labelsize=6)
        ins.set_title("price distribution", fontsize=6, pad=1)
    axes[0].set_ylabel("share of markets that resolved Yes")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Resolved Polymarket binary markets: calibration by horizon", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_slopes(tables: dict[str, pd.DataFrame], path: Path) -> None:
    labels, slopes, ses = [], [], []
    for name, t in tables.items():
        col = t.columns[0]
        for _, r in t.iterrows():
            labels.append(f"{name}: {r[col]} (n={int(r.n):,})")
            slopes.append(r.slope)
            ses.append(r.se_slope)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(labels) + 1.4))
    ax.axvline(1.0, color=GREY, ls="--", lw=1)
    ax.errorbar(slopes, y, xerr=1.96 * np.array(ses), fmt="o", color=BLUE, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(
        "calibration slope on logit(price), 95% CI (1 = calibrated; <1 = prices too extreme)"
    )
    ax.set_title("Calibration slope by horizon, and by category / volume at 7 days", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_strategy(results: dict, split_dt: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    any_line = False
    for label, r in results.items():
        if r["n_bets"] == 0:
            continue
        rows = r["bet_rows"].assign(pnl=r["pnl"]).sort_values("t_res")
        ax.plot(
            pd.to_datetime(rows["t_res"], unit="s", utc=True),
            rows["pnl"].cumsum(),
            label=label,
            lw=1.3,
        )
        any_line = True
    ax.axhline(0, color=GREY, lw=1)
    ax.set_ylabel("cumulative P&L (USD per 1-contract bet)")
    ax.set_title(
        f"Out-of-sample recalibration strategy, test period from {split_dt}\n"
        "(assumed 1c half-spread, 2% fee on winnings; not trading advice)",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    if any_line:
        ax.legend(fontsize=8, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame, fmt: dict[str, str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(
                format(int(v) if f.endswith("d") else v, f)
                if (f and pd.notna(v))
                else ("" if pd.isna(v) else str(v))
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


FMT = {
    "n": ",d",
    "base_rate": ".3f",
    "brier": ".4f",
    "brier_climatology": ".4f",
    "log_loss": ".4f",
    "reliability": ".4f",
    "resolution": ".4f",
    "uncertainty": ".4f",
    "intercept": ".3f",
    "slope": ".3f",
    "se_slope": ".3f",
    "p_slope_eq_1": ".3g",
    "p_joint": ".3g",
    "p_holm": ".3g",
    "horizon_days": "d",
}


def run(data: Path, out_dir: Path, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data)
    df = add_volume_tercile(df)
    n_markets = df["market_id"].nunique()
    t0 = pd.to_datetime(df["t_res"].min(), unit="s", utc=True).date()
    t1 = pd.to_datetime(df["t_res"].max(), unit="s", utc=True).date()

    hz = by_horizon(df)
    cat7 = breakdown(df, 7, "category")
    vol7 = breakdown(df, 7, "volume_tercile")
    catall = {h: breakdown(df, h, "category") for h in HORIZONS_DAYS}
    fl7 = favourite_longshot(df, 7)
    fl1 = favourite_longshot(df, 1)

    fig_calibration(df, out_dir / "calibration.png")
    slope_tables = {
        "horizon": hz.rename(columns={"horizon_days": "h"}).assign(
            h=lambda t: t["h"].astype(str) + "d"
        ),
        "category": cat7,
        "volume": vol7,
    }
    fig_slopes(slope_tables, out_dir / "slopes.png")

    # strategy: one row per market at a fixed horizon, so bets are independent draws
    costs = Costs()
    strat_rows, strat_fig, split_info = [], {}, {}
    for h in (7, 1):
        d = df[df.horizon_days == h]
        if d["market_id"].nunique() < 100:
            continue
        train, test, split = time_split(d)
        split_dt = str(pd.to_datetime(split, unit="s", utc=True).date())
        split_info[h] = (split_dt, train["market_id"].nunique(), test["market_id"].nunique())
        for thr in STRAT_THRESHOLDS:
            r = run_strategy(train, test, thr, costs)
            strat_rows.append(
                {"horizon_days": h, "costs": "1c spread + 2% fee"}
                | {k: v for k, v in r.items() if k not in ("pnl", "bet_rows")}
            )
            if thr in (0.02, 0.05) and h == 7:
                strat_fig[f"edge > {thr:.2f}"] = r
        r0 = run_strategy(train, test, 0.02, Costs(0.0, 0.0))
        strat_rows.append(
            {"horizon_days": h, "costs": "frictionless"}
            | {k: v for k, v in r0.items() if k not in ("pnl", "bet_rows")}
        )
    strat = pd.DataFrame(strat_rows)
    if 7 in split_info:
        fig_strategy(strat_fig, split_info[7][0], out_dir / "strategy.png")

    # ---- results.md ----
    L = [
        f"# Results\n\nGenerated by `pmcal analyze` from `{data}`. {n_markets:,} markets resolving "
        f"{t0} to {t1}; {len(df):,} market-horizon snapshots.\n"
    ]
    L.append("## 1. Calibration by horizon\n")
    L.append(md_table(hz, FMT) + "\n")
    L.append(
        "`brier_climatology` = base_rate * (1 - base_rate): the Brier score of always quoting the "
        "base rate. Reliability/resolution/uncertainty use 10 equal-width bins; "
        "BS = REL - RES + UNC holds up to a small within-bin term (see README).\n"
    )
    L.append(
        "Slope and intercept: logistic regression of outcome on logit(price) (prices clipped to "
        "[0.01, 0.99]); H0 is slope 1, intercept 0. Standard errors are cluster-robust by market. "
        "`p_slope_eq_1` is a Wald test on the slope, `p_joint` a Wald chi2(2) test of both.\n"
    )
    L.append("### Calibration bins (10 equal-width bins, Wilson 95% CI)\n")
    for h in HORIZONS_DAYS:
        d = df[df.horizon_days == h]
        if len(d) < MIN_CELL:
            continue
        t = calibration_table(d["price"], d["outcome"])
        L.append(f"**{h} days before resolution**\n")
        L.append(
            md_table(
                t,
                {
                    "n": ",d",
                    "bin_lo": ".1f",
                    "bin_hi": ".1f",
                    "mean_price": ".3f",
                    "freq": ".3f",
                    "ci_lo": ".3f",
                    "ci_hi": ".3f",
                },
            )
            + "\n"
        )
    L.append("## 2. Favourite-longshot bias: extreme-price buckets\n")
    for h, t in ((7, fl7), (1, fl1)):
        if len(t):
            L.append(
                f"**{h}-day horizon** (gap = realised frequency minus mean price; negative gap at "
                "low prices = longshots overpriced)\n"
            )
            L.append(
                md_table(
                    t,
                    {
                        "n": ",d",
                        "mean_price": ".3f",
                        "freq": ".3f",
                        "ci_lo": ".3f",
                        "ci_hi": ".3f",
                        "gap": "+.3f",
                    },
                )
                + "\n"
            )
    L.append("## 3. Where is calibration worst?\n")
    L.append(
        f"Rows with fewer than {MIN_CELL} observations are omitted. `p_holm` is the Holm-adjusted p-value "
        "across the rows of each table (slope = 1).\n"
    )
    L.append(
        "Slope columns are blank where the logistic fit has (quasi-)complete separation "
        "(|slope| > 15), e.g. a category whose 1-day prices predict every outcome.\n"
    )
    L.append("**By category, 7-day horizon**\n")
    L.append(md_table(cat7, FMT) + "\n")
    L.append("**By volume tercile (lifetime USD volume), 7-day horizon**\n")
    L.append(md_table(vol7, FMT) + "\n")
    for h in (30, 1):
        if len(catall.get(h, [])):
            L.append(f"**By category, {h}-day horizon**\n")
            L.append(md_table(catall[h], FMT) + "\n")
    L.append("## 4. Naive strategy check (out of sample)\n")
    L.append(
        "**Not trading advice.** Recalibration mapping (logistic on logit(price)) is fitted on "
        "markets resolving before the split date only; bets are placed on markets whose snapshot "
        "and resolution both fall after it. A side is bought when its model-implied expected "
        "profit per 1-USD contract, after costs, exceeds the threshold. Costs: entry at price + "
        "half-spread, fee on winnings only. Fills at the snapshot price plus assumed spread; no "
        "slippage, market impact or capacity limit. Mean P&L CI: cluster bootstrap over markets.\n"
    )
    for h, (sd, ntr, nte) in split_info.items():
        L.append(f"- {h}-day horizon: split {sd}; {ntr:,} training markets, {nte:,} test markets.")
    L.append("")
    if len(strat):
        s = strat.copy()
        s = s[
            [
                "horizon_days",
                "costs",
                "threshold",
                "n_test",
                "n_bets",
                "hit_rate",
                "mean_pnl",
                "ci_lo",
                "ci_hi",
                "roi",
                "total_pnl",
                "fit_slope",
            ]
        ]
        L.append(
            md_table(
                s,
                {
                    "horizon_days": "d",
                    "threshold": ".2f",
                    "n_test": ",d",
                    "n_bets": ",d",
                    "hit_rate": ".3f",
                    "mean_pnl": "+.4f",
                    "ci_lo": "+.4f",
                    "ci_hi": "+.4f",
                    "roi": "+.3f",
                    "total_pnl": "+.2f",
                    "fit_slope": ".3f",
                },
            )
            + "\n"
        )
    (out_dir / "results.md").write_text("\n".join(L))
    log(f"wrote {out_dir / 'results.md'} and figures")
    return {
        "hz": hz,
        "strat": strat,
        "cat7": cat7,
        "vol7": vol7,
        "fl7": fl7,
        "fl1": fl1,
        "n_markets": n_markets,
        "split_info": split_info,
    }
