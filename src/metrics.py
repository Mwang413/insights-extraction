import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer


def per_post_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("media_id").agg(
        comments=("comment_text","size"),
        tag_rate=("has_tag","mean"),
        pos_rate=("sent_label", lambda s: (s=="pos").mean()),
        caption=("media_caption","first"),
        cap_len=("cap_len","first"),
        cap_has_question=("cap_has_question","first"),
        cap_emoji_count=("cap_emoji_count","first"),
        first_ts=("ts","min"),
    ).reset_index()
    # small smoothing to avoid extremes
    agg["eng_score"] = 0.7*agg["tag_rate"] + 0.3*agg["pos_rate"]
    return agg

def seasonality_tables(df: pd.DataFrame):
    # volume by dow/hour
    vol = pd.pivot_table(df, index="dow", columns="hour", values="comment_text", aggfunc="size", fill_value=0)
    # tag rate by dow/hour
    tag = pd.pivot_table(df, index="dow", columns="hour", values="has_tag", aggfunc="mean", fill_value=0.0)
    return vol, tag

def detect_promo_spikes(df: pd.DataFrame):
    # daily counts -> rolling z-score to flag spikes (proxy for promo events)
    daily = df.groupby(pd.Grouper(key="ts", freq="D")).size().rename("n").to_frame()
    daily["roll_mean"] = daily["n"].rolling(7, min_periods=3).mean()
    daily["roll_std"]  = daily["n"].rolling(7, min_periods=3).std()
    daily["z"] = (daily["n"] - daily["roll_mean"]) / (daily["roll_std"] + 1e-6)
    daily["is_spike"] = daily["z"] >= 2.0
    return daily


def caption_feature_correlations(posts: pd.DataFrame) -> pd.DataFrame:
    # compute simple Pearson r against outcomes
    out = []
    metrics = ["comments","tag_rate","pos_rate","eng_score"]
    features = ["cap_len","cap_has_question","cap_emoji_count"]
    for m in metrics:
        for f in features:
            x = posts[f].astype(float)
            if posts[f].dtype == bool:
                x = posts[f].astype(int)
            y = posts[m].astype(float)
            if x.var() == 0 or y.var() == 0:
                r = np.nan
            else:
                r = np.corrcoef(x, y)[0,1]
            out.append({"metric": m, "feature": f, "pearson_r": r})
    return pd.DataFrame(out).sort_values(["metric","pearson_r"], ascending=[True, False])

def caption_keywords(df: pd.DataFrame, min_df=15, top_k=40):
    # extract top unigrams/bigrams from captions to act as product/feature proxies (e.g., scent names)
    caps = df.drop_duplicates("media_id")["media_caption"]
    vec = CountVectorizer(ngram_range=(1,2), min_df=min_df, stop_words="english")
    X = vec.fit_transform(caps)
    vocab = vec.get_feature_names_out()
    counts = np.asarray(X.sum(0)).ravel()
    top = pd.DataFrame({"term": vocab, "count": counts}).sort_values("count", ascending=False).head(top_k)
    return top

def assign_caption_terms_to_posts(posts: pd.DataFrame, top_terms: pd.DataFrame):
    # mark whether each top term appears in the caption; then compare engagement per term
    posts = posts.copy()
    for t in top_terms["term"]:
        safe = re.escape(t)
        posts[f"cap_has::{t}"] = posts["caption"].str.contains(rf"(?i)\b{safe}\b")
    return posts

def term_level_engagement(posts_with_terms: pd.DataFrame, top_terms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in top_terms["term"]:
        col = f"cap_has::{t}"
        if col not in posts_with_terms.columns: 
            continue
        sub = posts_with_terms[posts_with_terms[col]]
        if len(sub) >= 5:  # avoid tiny groups
            rows.append({
                "term": t,
                "n_posts": len(sub),
                "avg_comments": sub["comments"].mean(),
                "avg_tag_rate": sub["tag_rate"].mean(),
                "avg_pos_rate": sub["pos_rate"].mean(),
                "avg_eng_score": sub["eng_score"].mean(),
            })
    return pd.DataFrame(rows).sort_values("avg_eng_score", ascending=False)
