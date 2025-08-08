from __future__ import annotations
import os, textwrap
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

def _ensure_dir(p: Optional[str]) -> None:
    if p:
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def _wrap(s: str, width: int) -> str:
    s = (s or "").strip()
    return "\n".join(textwrap.wrap(s, width=width)) if s else ""


def plot_post_engagement_trajectories(
    df_comments: pd.DataFrame,
    posts: pd.DataFrame,
    out_path: Optional[str] = None,
    show: bool = False,
    *,
    top_k: int = 5,
    select_by: str = "eng_score",    # or "comments"
    freq: str = "H",                 # "H" hourly or "D" daily
    min_comments_per_post: int = 20, # threshold for Top-K ranking
    min_comments_plot: int = 3,      # threshold to include in background lines
    caption_col: str = "caption",
    post_time_col: str = "first_ts",
    wrap_caption: int = 90,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes], Optional[str], pd.DataFrame]:
    """
    Plot cumulative engagement trajectories for ALL posts (faint background),
    highlight Top-K posts, and show a metrics table + full captions.
    """
    import os, textwrap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # ---------- helpers ----------
    def _ensure_dir(p: Optional[str]) -> None:
        if p:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    def _wrap(s: str, w: int) -> str:
        s = (s or "").strip()
        return "\n".join(textwrap.wrap(s, width=w)) if s else ""

    # ---------- validate inputs ----------
    need = {"media_id", "comments", "tag_rate", "pos_rate", select_by, caption_col, post_time_col}
    miss = need - set(posts.columns)
    if miss:
        raise ValueError(f"'posts' is missing columns: {miss}")

    # Top-K by select_by (ensure enough comments)
    top_df = posts.loc[posts["comments"] >= min_comments_per_post].copy()
    if top_df.empty:
        raise ValueError("No posts meet min_comments_per_post; lower the threshold.")
    top_df = top_df.sort_values(select_by, ascending=False).head(top_k).copy()

    # ---------- FORCE EVERYTHING TO UTC (no tz-localize/convert branches) ----------
    df_comments = df_comments.copy()
    df_comments["ts"] = pd.to_datetime(df_comments["ts"], errors="coerce", utc=True)

    # Build t0 map: prefer explicit post_time; else min comment ts. All in UTC.
    t0_posts = pd.to_datetime(posts[post_time_col], errors="coerce", utc=True)
    # key t0_posts by media_id
    t0_posts.index = posts["media_id"].values
    t0_from_comments = df_comments.groupby("media_id")["ts"].min()

    # Combine (explicit post_time wins). Concat then groupby-first keeps dtype as tz-aware.
    t0_map = pd.concat([t0_from_comments, t0_posts]).groupby(level=0).first()

    # ---------- build trajectories for ALL posts ----------
    unit = "h" if freq.upper() == "H" else "D"
    rows = []
    for mid, sub in df_comments.groupby("media_id"):
        if len(sub) < min_comments_plot:
            continue
        t0 = t0_map.get(mid, pd.NaT)
        if pd.isna(t0):
            continue

        sub = sub.sort_values("ts")
        rs = (
            sub.set_index("ts")
               .resample(freq)
               .size()
               .rename("n")
               .to_frame()
        )
        if rs.empty:
            continue

        # safety: ensure index is UTC-aware (it should be, but guard anyway)
        if rs.index.tz is None:
            rs.index = rs.index.tz_localize("UTC")

        # safety: ensure t0 is a pandas Timestamp with tz
        if not isinstance(t0, pd.Timestamp):
            t0 = pd.Timestamp(t0, tz="UTC")
        elif t0.tz is None:
            t0 = t0.tz_localize("UTC")

        rs["n"] = rs["n"].fillna(0).astype(int)
        rs["cum_n"] = rs["n"].cumsum()
        rs["age_delta"] = (rs.index - t0)                # tz-aware subtraction
        rs["age_val"]   = rs["age_delta"] / np.timedelta64(1, unit)
        rs["media_id"]  = mid
        rows.append(rs.reset_index())

    if not rows:
        raise ValueError("No series to plot after resampling (check inputs).")

    tsdf_all = pd.concat(rows, ignore_index=True)

    # Subset for Top-K lines
    top_ids = set(top_df["media_id"])
    tsdf_top = tsdf_all[tsdf_all["media_id"].isin(top_ids)].copy()

    # ---------- T50 (hours) for Top-K ----------
    t50_rows = []
    for mid, sub in tsdf_top.groupby("media_id"):
        sub = sub.sort_values("age_delta")
        total = float(sub["cum_n"].max())
        if total <= 0:
            t50 = np.nan
        else:
            half = 0.5 * total
            hit = sub[sub["cum_n"] >= half]
            t50 = float(hit.iloc[0]["age_delta"] / np.timedelta64(1, "h")) if not hit.empty else np.nan
        t50_rows.append({"media_id": mid, "T50_hours": t50})
    t50df = pd.DataFrame(t50_rows)

    # ---------- layout (plot + table + captions) ----------
    label_map = {mid: f"P{i+1}" for i, mid in enumerate(top_df["media_id"].tolist())}

    # captions for panel sizing
    captions_full, total_lines = [], 0
    for mid in top_df["media_id"]:
        cap_series = posts.loc[posts["media_id"] == mid, caption_col]
        cap = str(cap_series.iloc[0]) if not cap_series.empty else ""
        text = f"{label_map[mid]}: {cap.strip()}"
        wrapped = textwrap.wrap(text, width=wrap_caption) if text else []
        total_lines += max(1, len(wrapped)) + 1
        captions_full.append((label_map[mid], wrapped))

    base_h = 7.0
    fig_h = base_h + 0.18 * max(0, total_lines - 12)
    fig = plt.figure(figsize=(14, fig_h))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[2.2, 1.4],
        width_ratios=[1.6, 1.0],
        hspace=0.18, wspace=0.18
    )

    ax_line = fig.add_subplot(gs[0, :])
    ax_tbl  = fig.add_subplot(gs[1, 0]); ax_tbl.axis("off")
    ax_caps = fig.add_subplot(gs[1, 1]); ax_caps.axis("off")

    # ALL posts as faint background
    for mid, sub in tsdf_all.groupby("media_id"):
        sub = sub.sort_values("age_val")
        ax_line.plot(sub["age_val"].to_numpy(), sub["cum_n"].to_numpy(),
                     linewidth=0.8, alpha=0.25)

    # Top-K highlighted + labeled
    for mid, sub in tsdf_top.groupby("media_id"):
        sub = sub.sort_values("age_val")
        ax_line.plot(sub["age_val"].to_numpy(), sub["cum_n"].to_numpy(),
                     linewidth=1.8, label=label_map.get(mid, str(mid)))

    ax_line.set_title(f"Post Engagement Trajectories (All Posts, Top {top_k} Highlighted; freq={freq})")
    ax_line.set_ylabel("Cumulative comments")
    ax_line.set_xlabel("Age (hours)" if freq.upper() == "H" else "Age (days)")
    ax_line.legend(title="Top posts", ncols=min(5, top_k), fontsize=9)

    # Bottom-left: Top-K metrics table
    merged = (
        top_df[["media_id", "comments", "tag_rate", "pos_rate", "eng_score"]]
        .merge(t50df, on="media_id", how="left")
    )
    merged["Post"] = merged["media_id"].map(label_map)
    merged = merged[["Post", "media_id", "comments", "tag_rate", "pos_rate", "eng_score", "T50_hours"]].copy()
    for c in ["tag_rate", "pos_rate", "eng_score"]:
        merged[c] = merged[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    merged["T50_hours"] = merged["T50_hours"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")

    table = ax_tbl.table(
        cellText=merged.values,
        colLabels=["Post", "Post ID", "Comments", "Tag Rate", "Pos Rate", "Eng Score", "T50 (h)"],
        loc="center", cellLoc="left", colLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    # Bottom-right: full captions wrapped
    y = 1.0
    line_step = 0.045
    for label, wrapped in captions_full:
        if not wrapped:
            continue
        ax_caps.text(0.0, y, label, fontsize=10, weight="bold", va="top", ha="left"); y -= line_step
        for line in wrapped:
            ax_caps.text(0.06, y, line, fontsize=9, va="top", ha="left"); y -= line_step
        y -= line_step  # spacer

    fig.tight_layout()

    saved = None
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        saved = out_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    out_table = merged.copy()
    out_table.columns = ["Post", "Post ID", "Comments", "Tag Rate", "Pos Rate", "Eng Score", "T50 (h)"]
    return fig, (ax_line, ax_tbl, ax_caps), saved, out_table












def plot_seasonality_heatmaps(
    vol: pd.DataFrame,
    tag: pd.DataFrame,
    out_volume_path: Optional[str] = None,
    out_tag_rate_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes, Optional[str], plt.Figure, plt.Axes, Optional[str]]:
    """
    Heatmaps for DoW x Hour:
      vol: counts pivot (index=dow, columns=hour)
      tag: tag_rate pivot (index=dow, columns=hour)
    """
    # Volume heatmap
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    im1 = ax1.imshow(vol.values, aspect="auto")
    ax1.set_title("Comments heatmap (DoW × Hour)")
    ax1.set_ylabel("DoW (0=Mon)")
    ax1.set_xlabel("Hour")
    fig1.colorbar(im1, ax=ax1)
    fig1.tight_layout()
    saved1 = None
    if out_volume_path:
        _ensure_dir(out_volume_path)
        fig1.savefig(out_volume_path, dpi=150)
        saved1 = out_volume_path

    # Tag-rate heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    im2 = ax2.imshow(tag.values, aspect="auto", vmin=0, vmax=1)
    ax2.set_title("Tag rate heatmap (DoW × Hour)")
    ax2.set_ylabel("DoW (0=Mon)")
    ax2.set_xlabel("Hour")
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    saved2 = None
    if out_tag_rate_path:
        _ensure_dir(out_tag_rate_path)
        fig2.savefig(out_tag_rate_path, dpi=150)
        saved2 = out_tag_rate_path

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
    return fig1, ax1, saved1, fig2, ax2, saved2

def plot_top_posts(
    posts: pd.DataFrame,
    out_csv_path: Optional[str] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Save (and return) the top posts by eng_score.

    Requires columns: ['eng_score','media_id','comments','tag_rate','pos_rate','first_ts','caption'].
    """
    required = {"eng_score","media_id","comments","tag_rate","pos_rate","first_ts","caption"}
    missing = required - set(posts.columns)
    if missing:
        raise ValueError(f"posts is missing columns: {missing}")

    top = posts.sort_values("eng_score", ascending=False).head(top_n).copy()
    if out_csv_path:
        _ensure_dir(out_csv_path)
        top.to_csv(out_csv_path, index=False)
    return top

def plot_term_leaders(
    term_df: pd.DataFrame,
    out_csv_path: Optional[str] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Save (and return) the top caption terms by avg_eng_score.

    Expects columns: ['term','n_posts','avg_comments','avg_tag_rate','avg_pos_rate','avg_eng_score'].
    """
    required = {"term","n_posts","avg_comments","avg_tag_rate","avg_pos_rate","avg_eng_score"}
    missing = required - set(term_df.columns)
    if missing:
        raise ValueError(f"term_df is missing columns: {missing}")

    top = term_df.sort_values("avg_eng_score", ascending=False).head(top_n).copy()
    if out_csv_path:
        _ensure_dir(out_csv_path)
        top.to_csv(out_csv_path, index=False)
    return top
