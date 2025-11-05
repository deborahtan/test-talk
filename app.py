import os
import io
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from groq import Groq
import requests
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datetime import datetime
from functools import lru_cache

# Optional Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Use your published CSV URL (publish-to-web -> CSV)
CSV_EXPORT_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTH33TC1xTixH8TWGAOUUe3o-UIFX82HMaBv8BlI4KA5UnJxYs50QBitDUwXB_Jkl8M52CdE66s_XDx/pub?output=csv"

# HF model config
HF_EMOTION_MODEL = os.getenv("HF_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
EMOTION_MODEL_MAX = int(os.getenv("EMOTION_MODEL_MAX", "200"))

# ---------- fetching helper (robust, with debug preview) ----------
def fetch_sheet_as_records(csv_url, timeout=30):
    try:
        resp = requests.get(csv_url, timeout=timeout)
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None, None

    st.write("Debug: HTTP status", resp.status_code)
    st.write("Debug: Content-Type", resp.headers.get("content-type"))

    preview = resp.content[:1600]
    try:
        preview_text = preview.decode(resp.encoding or "utf-8", errors="replace")
    except Exception:
        preview_text = str(preview)
    st.code(preview_text, language="plain")

    if resp.status_code != 200:
        st.error("HTTP status not 200. Check that the publish-to-web CSV URL is correct and the sheet is published.")
        return None, None

    ct = resp.headers.get("content-type", "")
    if "html" in ct.lower() or preview_text.lstrip().lower().startswith("<!doctype html"):
        st.error("Response looks like HTML (a webpage). Use the publish-to-web CSV link, not the pubhtml page.")
        return None, None

    # parse CSV robustly
    try:
        buf = io.BytesIO(resp.content)
        df = pd.read_csv(buf, dtype=str)
        return df.to_dict(orient="records"), "google_sheet"
    except Exception as e:
        try:
            txt = resp.content.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(txt), dtype=str)
            return df.to_dict(orient="records"), "google_sheet"
        except Exception as e2:
            st.error(f"Failed to parse CSV: {e}; fallback failed: {e2}")
            return None, None

# ---------- minimal helpers (same as your pipeline) ----------
def safe_int(v):
    try:
        return int(float(v or 0))
    except Exception:
        return 0

def normalize_item(item):
    def g(k):
        if not item:
            return None
        if k in item and pd.notna(item[k]):
            return item.get(k)
        parts = k.split(".")
        cur = item
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur
    return {
        "author": g("authorMeta.name") or g("author") or g("author_name"),
        "avatar": g("authorMeta.avatar") or g("authorAvatar") or g("author_image"),
        "text": g("text") or g("caption") or g("description") or "",
        "likes": safe_int(g("diggCount") or g("likeCount") or g("likes") or 0),
        "shares": safe_int(g("shareCount") or g("shares") or 0),
        "plays": safe_int(g("playCount") or g("plays") or g("views") or 0),
        "comments": safe_int(g("commentCount") or g("comments") or 0),
        "music": g("musicMeta.musicName") or "",
        "music_author": g("musicMeta.musicAuthor") or "",
        "created": g("createTimeISO") or g("created_at") or g("post_date") or ""
    }

# ---------- emotion model (cached small) ----------
try:
    from transformers import pipeline
except Exception:
    pipeline = None

@lru_cache(maxsize=1)
def get_emotion_pipeline(model_name=HF_EMOTION_MODEL, device=-1):
    if pipeline is None:
        return None
    try:
        return pipeline("text-classification", model=model_name, return_all_scores=True, device=device)
    except Exception as e:
        st.warning(f"Failed to load HF model {model_name}: {e}")
        return None

def map_model_label_to_emotion(label):
    label = label.lower()
    return {
        "joy": "happy", "happiness": "happy",
        "love": "generous",
        "sadness": "sad", "sad": "sad",
        "anger": "angry", "angry": "angry",
        "fear": "fear", "surprise": "surprise",
        "neutral": "neutral"
    }.get(label, label)

def analyze_text_with_model(text, model_pipeline=None, min_confidence=0.55):
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.0}
    if model_pipeline is None:
        model_pipeline = get_emotion_pipeline()
    if model_pipeline:
        try:
            preds = model_pipeline(text[:1000])
            if isinstance(preds, list) and len(preds) > 0:
                scores = preds[0]
                top = max(scores, key=lambda x: x["score"])
                label = top["label"]
                conf = float(top["score"])
                emotion = map_model_label_to_emotion(label)
                if conf < min_confidence:
                    return {"emotion": "unclear", "sentiment": "unclear", "confidence": conf}
                sentiment = "positive" if emotion in ("happy","generous","surprise") else ("negative" if emotion in ("sad","angry","fear") else "neutral")
                return {"emotion": emotion, "sentiment": sentiment, "confidence": conf}
        except Exception:
            pass
    # fallback heuristic
    return analyze_text_heuristic(text)

def analyze_text_heuristic(text):
    if not text or not isinstance(text, str):
        return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.0}
    if re.search(r"\b(win|love|happy|joy|great|awesome|yay|yes|delight|best)\b", text, re.I):
        return {"emotion": "happy", "sentiment": "positive", "confidence": 0.65}
    if re.search(r"\b(annoy|panic|stress|forgot|hate|angry|sad|oops|frustrat)\b", text, re.I):
        return {"emotion": "angry", "sentiment": "negative", "confidence": 0.65}
    if re.search(r"\b(memory|remember|nostalgia|nostalgic|back then|old times)\b", text, re.I):
        return {"emotion": "nostalgic", "sentiment": "neutral", "confidence": 0.7}
    if re.search(r"\b(give|donate|gift|generous|share|help)\b", text, re.I):
        return {"emotion": "generous", "sentiment": "positive", "confidence": 0.7}
    return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.5}

# ---------- small utils ----------
def extract_hashtags(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"#\w+", text.lower())

def most_used_emoji(texts):
    emoji_pattern = re.compile("[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000027BF]+", flags=re.UNICODE)
    counter = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        for em in emoji_pattern.findall(t):
            counter[em] += 1
    if counter:
        return counter.most_common(1)[0]
    return (None, 0)

@st.cache_data(show_spinner=False)
def process_records(records, use_model=True, max_items=500):
    hashtag_counter = Counter()
    keyword_counter = Counter()
    sentiment_counts = Counter()
    emotional_barometer = Counter()
    posts = []
    model_pipeline = get_emotion_pipeline() if use_model else None

    for i, raw in enumerate((records or [])[:max_items]):
        item = raw if isinstance(raw, dict) else {}
        r = normalize_item(item)
        if use_model and model_pipeline and i < EMOTION_MODEL_MAX:
            analysis = analyze_text_with_model(r["text"], model_pipeline=model_pipeline, min_confidence=0.55)
        else:
            analysis = analyze_text_heuristic(r["text"])
        emotion = analysis.get("emotion", "unclear")
        sentiment = analysis.get("sentiment", "unclear")
        sentiment_counts[sentiment] += 1
        emotional_barometer[emotion] += 1
        tags = [t.lstrip("#") for t in extract_hashtags(r["text"])]
        hashtag_counter.update(tags)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", (r["text"] or "").lower())
        keywords = [w for w in words if w not in ENGLISH_STOP_WORDS]
        keyword_counter.update(keywords)
        posts.append({
            "author": r["author"],
            "avatar": r["avatar"],
            "text": r["text"],
            "sentiment": sentiment,
            "emotion": emotion,
            "likes": r["likes"],
            "shares": r["shares"],
            "plays": r["plays"],
            "comments": r["comments"],
            "music": r["music"],
            "music_author": r["music_author"],
            "created": r["created"]
        })

    top_keywords = [k for k, _ in keyword_counter.most_common(10)]
    return dict(hashtag_counter), dict(sentiment_counts), dict(emotional_barometer), top_keywords, posts

# ---------- main ----------
records, source_label = fetch_sheet_as_records(CSV_EXPORT_URL)
if not records:
    st.stop()

hashtag_counter, sentiment_counts, emotional_barometer, top_keywords, top_posts_data = process_records(records, use_model=True, max_items=500)

# UI (same layout as you requested; removed video embeds)
st.set_page_config(page_title="NZ Christmas Retail Trendspotter", layout="wide")
st.title("🎄 NZ Christmas Retail Trendspotter")
st.caption(f"Source: {source_label} — emotion model max {EMOTION_MODEL_MAX} rows")

st.sidebar.title("📅 Date Range")
_ = st.sidebar.radio("View trends from:", ["Last 24 hours", "Last 7 days"])

# derive totals and emoji
total_posts = sum(sentiment_counts.values()) or 1
sentiment_pct = {k: round(v / total_posts * 100, 1) for k, v in sentiment_counts.items()}
texts = [p.get("text", "") for p in top_posts_data]
top_emoji, top_emoji_count = most_used_emoji(texts)

# Vibe check, posts, creatives, hashtag cloud etc. (omitted here for brevity — reuse the UI from your existing app)
st.write("Emotion split:", sentiment_pct)
if top_emoji:
    st.write("Top emoji:", top_emoji, top_emoji_count)
else:
    st.write("No emoji detected")
