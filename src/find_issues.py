# src/issues.py
from __future__ import annotations
import re, json, math, textwrap
from typing import Callable, Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# ----------------------------
# Minimal cleaning + filters
# ----------------------------
ISSUE_RX = re.compile(
    r"(?i)\b("
    r"broke|broken|leak|leaking|leaky|spill|damaged|defect|"
    r"rash|burn|itch|allergic|irritat|sting|hurt|"
    r"didn.?t work|doesn.?t work|no effect|"
    r"too (?:strong|greasy|oily|sticky)|smell (?:too|very) (?:strong|bad)|"
    r"price|expensive|overpriced|"
    r"shipping|delivery|lost|late|delayed|"
    r"restock|sold out|out of stock|can.?t find|where to buy|"
    r"messy|hard to open|cap|pump|packaging|"
    r"reaction|acne|break.?out"
    r")\b"
)

def _prep_comment(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"@[\w.]+", "@user", s)         # anonymize mentions
    s = re.sub(r"http\S+|www\.\S+", "", s)     # drop urls
    s = re.sub(r"\s+", " ", s)
    return s

def _select_negative_comments(df_comments: pd.DataFrame, media_id: Any) -> List[str]:
    sub = df_comments[df_comments["media_id"] == media_id].copy()
    sub["txt"] = sub["comment_text"].astype(str).map(_prep_comment)
    neg = (sub["sent_label"] == "neg") | sub["txt"].str.contains(ISSUE_RX, regex=True, na=False)
    comments = sub.loc[neg, "txt"].tolist()
    # Dedup near-identical (very light)
    seen = set(); uniq = []
    for t in comments:
        key = re.sub(r"[^a-z0-9 ]", "", t.lower())
        if key not in seen:
            seen.add(key); uniq.append(t)
    return uniq

# ----------------------------
# Prompting + LLM I/O
# ----------------------------
PROMPT_HEADER = """You are analyzing customer comments for ONE Instagram post from a skincare brand.
Extract the TOP recurring **issues/dissatisfactions** (not praise, not generic tags).
Group semantically similar complaints together. Ignore spam and '@user' tags.

Return STRICT JSON with this schema:
{
  "issues": [
    {
      "title": "short label for the issue",
      "description": "1-2 sentences explaining the pain point",
      "frequency_estimate": "low|medium|high",
      "example_comments": ["verbatim comment 1", "verbatim comment 2"]
    }
  ]
}

Rules:
- Max 5 issues.
- Only include TRUE issues/complaints or requests that imply dissatisfaction (e.g., restock).
- Use representative examples from the comments provided.
- Do not add extra fields.
"""

def build_chunk_prompt(comments: List[str]) -> str:
    bullet_comments = "\n".join(f"- {c}" for c in comments)
    return f"{PROMPT_HEADER}\n\nComments:\n{bullet_comments}\n\nJSON only:"

def default_json_loader(text: str) -> Dict[str, Any]:
    # Try exact parse; if it fails, try to extract the first {...} block.
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # last-ditch: bracket sanitize
        cleaned = text.strip().splitlines()
        cleaned = [ln for ln in cleaned if not ln.strip().startswith(("```", "json"))]
        cleaned = "\n".join(cleaned)
        return json.loads(cleaned)  # will raise; let caller handle

# ----------------------------
# Chunking + merge
# ----------------------------
def chunk_list(items: List[str], chunk_size: int = 80, max_chars: int = 12000) -> List[List[str]]:
    chunks, cur, cur_chars = [], [], 0
    for it in items:
        add = len(it)
        if len(cur) >= chunk_size or (cur_chars + add) > max_chars:
            if cur:
                chunks.append(cur); cur, cur_chars = [], 0
        cur.append(it)
        cur_chars += add
    if cur:
        chunks.append(cur)
    return chunks

def _norm_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def merge_issues(issue_lists: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Heuristic merge of issues across chunks by normalized title overlap."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for js in issue_lists:
        issues = js.get("issues", []) if isinstance(js, dict) else []
        for it in issues:
            title = _norm_title(it.get("title", ""))
            if not title:
                continue
            b = buckets.setdefault(title, {"title": it.get("title",""), "description": it.get("description",""),
                                           "count": 0, "examples": []})
            b["count"] += 1
            # collect a couple examples
            for ex in (it.get("example_comments") or [])[:2]:
                if len(b["examples"]) < 6:
                    b["examples"].append(ex)
    # rank by count, then by example volume
    merged = sorted(buckets.values(), key=lambda x: (x["count"], len(x["examples"])), reverse=True)[:top_n]
    # format output
    out = []
    for m in merged:
        out.append({
            "title": m["title"] or "Issue",
            "description": m["description"],
            "frequency_estimate": "high" if m["count"] >= 3 else ("medium" if m["count"] == 2 else "low"),
            "example_comments": m["examples"][:3]
        })
    return out

# ----------------------------
# Public entrypoint
# ----------------------------
def mine_post_issues_with_llm(
    df_comments: pd.DataFrame,
    posts: pd.DataFrame,
    *,
    top_k: int = 5,
    select_by: str = "eng_score",       # or "comments"
    min_comments_per_post: int = 20,
    llm_call_fn: Callable[[str], str],  # YOU provide this function
    chunk_size: int = 80,
    max_chars: int = 12000,
) -> pd.DataFrame:
    """
    For each Top-K post, send negative comments to an LLM in chunks and get the top issues.
    Returns a LONG dataframe: one row per (post, issue).
    """
    need = {"media_id","comments",select_by,"caption","first_ts"}
    miss = need - set(posts.columns)
    if miss:
        raise ValueError(f"'posts' missing columns: {miss}")

    top_df = (posts[posts["comments"] >= min_comments_per_post]
                .sort_values(select_by, ascending=False)
                .head(top_k)
                .copy())
    if top_df.empty:
        raise ValueError("No posts meet min_comments_per_post; lower the threshold.")

    rows = []
    for rank, (_, prow) in enumerate(top_df.iterrows(), start=1):
        mid = prow["media_id"]
        comments = _select_negative_comments(df_comments, mid)
        if not comments:
            continue
        chunks = chunk_list(comments, chunk_size=chunk_size, max_chars=max_chars)

        parsed_chunks = []
        for ch in chunks:
            prompt = build_chunk_prompt(ch)
            raw = llm_call_fn(prompt)  # <- your model call
            try:
                js = default_json_loader(raw)
                parsed_chunks.append(js)
            except Exception as e:
                # skip bad chunk silently; you can log if needed
                continue

        if not parsed_chunks:
            continue

        merged = merge_issues(parsed_chunks, top_n=5)

        for issue_rank, issue in enumerate(merged, start=1):
            rows.append({
                "post_rank": rank,
                "media_id": mid,
                "post_comments": int(prow["comments"]),
                "post_caption": str(prow["caption"])[:140],
                "issue_rank": issue_rank,
                "issue_title": issue.get("title",""),
                "issue_description": issue.get("description",""),
                "issue_frequency": issue.get("frequency_estimate",""),
                "issue_examples": issue.get("example_comments", []),
            })

    out = pd.DataFrame(rows)
    # nice ordering
    if not out.empty:
        out = out.sort_values(["post_rank","issue_rank"]).reset_index(drop=True)
    return out
