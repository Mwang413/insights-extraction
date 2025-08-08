# --- helpers (keep or remove if you already have them) ---
import re, hashlib
import pandas as pd

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def load_comments(path: str) -> pd.DataFrame:
    """
    Load CSV and normalize required column names:
      timestamp, media_id, media_caption, comment_text
    (case-insensitive)
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    required = ["timestamp", "media_id", "media_caption", "comment_text"]
    missing = [r for r in required if r not in cols]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")
    return df.rename(columns={
        cols["timestamp"]: "timestamp",
        cols["media_id"]: "media_id",
        cols["media_caption"]: "media_caption",
        cols["comment_text"]: "comment_text",
    })

def dedup_near(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact dupes on (media_id, comment_text) and near-dupes by hashing
    a normalized alnum-only version of comment_text.
    """
    df = df.drop_duplicates(subset=["media_id", "comment_text"]).copy()

    def norm_for_hash(x: str) -> str:
        x = normalize_text(str(x))
        return re.sub(r"[^a-z0-9 ]", "", x)

    h = df["comment_text"].fillna("").map(norm_for_hash).map(
        lambda s: hashlib.md5(s.encode()).hexdigest()
    )
    return df.loc[~h.duplicated()].copy()

def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic spam/PR filter:
      - URLs / promo phrases / excessive repeated chars / emoji-only lines
      - brand PR voice (seller-like replies)
    Returns a filtered copy.
    """
    txt = df["comment_text"].fillna("").astype(str)

    is_bot = (
        txt.str.contains(r"(?:http[s]?://|www\.)", regex=True, na=False)
        | txt.str.contains(
            r"(?i)\b(?:dm|telegram|whatsapp|promo|sponsor|collab|partnership|crypto|forex|investment)\b",
            regex=True, na=False
        )
        | txt.str.contains(r"([a-z])\1{4,}", regex=True, na=False)  # e.g., heyyyyy
        | txt.str.contains(r"^\W+$", regex=True, na=False)          # emoji-only / symbols
    )

    is_brand = txt.str.contains(
        r"(?i)(?:thanks for reaching out|please (?:dm|email)|send us a|our team|support@|we(?:'| a)re sorry)",
        regex=True, na=False
    )

    keep = ~(is_bot | is_brand)
    return df.loc[keep].copy()

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features:
      ts (UTC), date, dow/hour/week (nullable ints),
      has_tag, simple sentiment (fallback if VADER missing),
      caption features (length, question mark, emoji count)
    """
    out = df.copy()

    # text normalization
    out["comment_text"]  = out["comment_text"].astype(str).map(normalize_text)
    out["media_caption"] = out["media_caption"].fillna("").astype(str)

    # robust timestamp parse (strip zero-width/BOM), keep UTC
    ts_norm = (out["timestamp"].astype(str)
               .str.strip()
               .str.replace(r"[\u200b\u200e\ufeff]", "", regex=True))
    out["ts"] = pd.to_datetime(ts_norm, errors="coerce", utc=True)

    # time parts as nullable ints → won’t crash on NaT
    out["date"] = out["ts"].dt.date
    out["dow"]  = out["ts"].dt.dayofweek.astype("UInt8")  # 0..6
    out["hour"] = out["ts"].dt.hour.astype("UInt8")       # 0..23
    iso = out["ts"].dt.isocalendar()                      # pandas nullable ints
    out["week"] = iso.week.astype("UInt8")                # 1..53

    # tagging heuristic (positive engagement)
    out["has_tag"] = out["comment_text"].str.contains(r"@[\w.]+", regex=True, na=False)

    # sentiment: try VADER, else fallback lexicon
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _an = SentimentIntensityAnalyzer()
        scores = out["comment_text"].map(_an.polarity_scores)
        out["sent"] = scores.map(lambda d: d["compound"]).astype(float)
    except Exception:
        POS = set("""love loved amazing great awesome perfect fantastic wonderful excellent favorite best yes yay adore obsessed incredible""".split())
        NEG = set("""hate hated awful bad terrible worst broke rash burn itchy allergic disappointed waste never no""".split())
        def _simple_sent(s: str) -> float:
            toks = re.findall(r"[a-z']+", s.lower())
            pos = sum(t in POS for t in toks)
            neg = sum(t in NEG for t in toks)
            return max(-1.0, min(1.0, (pos - neg) / 3.0))
        out["sent"] = out["comment_text"].map(_simple_sent).astype(float)

    out["sent_label"] = pd.cut(out["sent"], [-1, -0.05, 0.05, 1],
                               labels=["neg", "neu", "pos"])

    # caption features (null-safe)
    cap = out["media_caption"]
    out["cap_len"]          = cap.str.len().astype("Int32")
    out["cap_has_question"] = cap.str.contains(r"\?", regex=True, na=False)
    # emoji count via findall (works across pandas versions)
    out["cap_emoji_count"]  = cap.str.findall(r"[\U0001F300-\U0001FAFF]").str.len().fillna(0).astype("Int16")

    return out
