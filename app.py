import os
import io
import math
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

# ---------- robust fetch (with debug preview) ----------
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

# ---------- text safety helper ----------
def safe_text(s):
    """
    Normalize text input for downstream processing.
    Converts None, NaN, non-str to '' and trims whitespace.
    """
    if s is None:
        return ""
    # pandas NaN is a float; catch it
    try:
        if isinstance(s, float) and math.isnan(s):
            return ""
    except Exception:
        pass
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    return s.strip()

# ---------- normalize and small helpers ----------
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
    if not text:
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
    return analyze_text_heuristic(text)

def analyze_text_heuristic(text):
    if not text:
        return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.0}
    if re.search(r"\b(win|love|happy|joy|great|awesome|yay|yes|delight|best)\b", text, re.I):
        return {"emotion": "happy", "sentiment": "positive", "confidence": 0.65}
    if re.search(r"\b(annoy|panic|stress|forgot|hate|angry|sad|oops|frustrat)\b", text, re.I):
        return {"emotion": "angry", "sentiment": "negative", "confidence": 0.65}
    if re.search(r"\b(memory|remember|nostalgia|nostalgic|back then|old times)\b", text, re.I):
        return {"emotion": "nostalgic", "sentiment": "neutral", "confidence": 0.7}
    if re.search(r"\b(give|donate|gift|generous|share|help)\b", text, re.I):
        return {"emotion": "generous", "sentiment": "positive", "confidence": 0.7}
    if re.search(r"\b(fun|lol|haha|hilarious|silly)\b", text, re.I):
        return {"emotion": "fun", "sentiment": "positive", "confidence": 0.6}
    return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.5}

# ---------- other small utils ----------
def extract_hashtags(text):
    text = safe_text(text)
    return re.findall(r"#\w+", text.lower())

def most_used_emoji(texts):
    emoji_pattern = re.compile("[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000027BF]+", flags=re.UNICODE)
    counter = Counter()
    for t in texts:
        t = safe_text(t)
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

        # normalize text safely
        txt = safe_text(r.get("text"))

        # use model for first EMOTION_MODEL_MAX rows only to limit resource use
        if use_model and model_pipeline and i < EMOTION_MODEL_MAX:
            analysis = analyze_text_with_model(txt, model_pipeline=model_pipeline, min_confidence=0.55)
        else:
            analysis = analyze_text_heuristic(txt)

        emotion = analysis.get("emotion", "unclear")
        sentiment = analysis.get("sentiment", "unclear")
        sentiment_counts[sentiment] += 1
        emotional_barometer[emotion] += 1

        tags = [t.lstrip("#") for t in extract_hashtags(txt)]
        hashtag_counter.update(tags)

        words = re.findall(r"\b[a-zA-Z]{3,}\b", txt.lower())
        keywords = [w for w in words if w not in ENGLISH_STOP_WORDS]
        keyword_counter.update(keywords)

        posts.append({
            "author": r.get("author"),
            "avatar": r.get("avatar"),
            "text": txt,
            "sentiment": sentiment,
            "emotion": emotion,
            "likes": r.get("likes", 0),
            "shares": r.get("shares", 0),
            "plays": r.get("plays", 0),
            "comments": r.get("comments", 0),
            "music": r.get("music", ""),
            "music_author": r.get("music_author", ""),
            "created": r.get("created", "")
        })

    top_keywords = [k for k, _ in keyword_counter.most_common(10)]
    return dict(hashtag_counter), dict(sentiment_counts), dict(emotional_barometer), top_keywords, posts

# ---------- main ----------
records, source_label = fetch_sheet_as_records(CSV_EXPORT_URL)
if not records:
    st.stop()

hashtag_counter, sentiment_counts, emotional_barometer, top_keywords, top_posts_data = process_records(records, use_model=True, max_items=500)

# UI
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

# Vibe Check
st.subheader("🔥 Vibe Check — Emotion Summary")
st.markdown("Synthesis of dominant emotions and top themes from the sample of posts. Low-confidence results are labelled 'unclear'.")

def top_themes_from_posts(posts, top_n=6):
    tokens = []
    for p in posts:
        text = safe_text(p.get("text", ""))
        tokens += re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    c = Counter(tokens)
    return [t for t, _ in c.most_common(top_n)]

top_themes = top_themes_from_posts(top_posts_data, top_n=6)

if top_themes:
    st.markdown("**Top themes:** " + ", ".join(top_themes))
else:
    st.markdown("**Top themes:** none found")

# songs
songs = Counter()
for p in top_posts_data:
    if safe_text(p.get("music")):
        songs[p["music"]] += 1
top_songs = [s for s, _ in songs.most_common(5)]

if top_songs:
    st.markdown("**Frequent songs in posts:** " + ", ".join(top_songs))
else:
    st.markdown("**Frequent songs:** none detected")

# sentiment percent split and emoji
st.markdown("**Emotion split (by post count):**")
split_text = " • ".join([f"{k.capitalize()}: {v}% ({sentiment_counts[k]} posts)" for k, v in sentiment_pct.items()])
st.markdown(split_text)

if top_emoji:
    st.markdown(f"**Most used emoji:** {top_emoji} — {top_emoji_count} times")
else:
    st.markdown("**Most used emoji:** none detected")

# Top posts table (no video embed)
st.subheader("🎄 Top Posts and Emotion Overview")
posts_df = pd.DataFrame(top_posts_data)
if not posts_df.empty:
    display_cols = ["author", "text", "sentiment", "emotion", "likes", "shares", "plays", "comments", "music", "created"]
    available = [c for c in display_cols if c in posts_df.columns]
    st.dataframe(posts_df[available].fillna(""), use_container_width=True)
else:
    st.info("No posts to display")

# Trend Spotter & Generate Creatives
st.subheader("🧠 Trend Spotter")
st.markdown("**New / emergent trends derived from top keywords and hashtags**")
for t in top_keywords[:6]:
    st.markdown(f"- {t}")

st.markdown("**Emotional barometer (counts):**")
for e, c in emotional_barometer.items():
    st.markdown(f"- {e.capitalize()}: {c}")

# Generate Creatives
st.subheader("✨ Creative Ideas Based on Trends")
post_summary = "\n".join([f"- \"{(p['text'] or '')[:120].strip()}...\" ({p['sentiment']}, {p['emotion']})" for p in sorted(top_posts_data, key=lambda x: x.get("likes",0)+x.get("shares",0)+x.get("comments",0), reverse=True)[:20]])
st.markdown("**Using today's top posts for inspiration:**")
st.markdown(post_summary)

def generate_creatives_groq(topics, sentiment_summary, post_summary):
    if not groq_client:
        return "Groq API key not configured. Set GROQ_API_KEY env var to enable."
    prompt = (
        "You're a creative assistant. Generate 3 short social lines for NZ Christmas retail: cheeky, emotionally honest, and shareable.\n\n"
        f"Trending hashtags: {topics}\n"
        f"Emotion summary: {sentiment_summary}\n"
        f"Top posts:\n{post_summary}\n\n"
        "Return 3 short lines (one per line)."
    )
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

if "creative_lines" not in st.session_state:
    st.session_state.creative_lines = ""

if st.button("Generate Creatives"):
    st.session_state.creative_lines = generate_creatives_groq(", ".join(list(hashtag_counter.keys())[:20]) if 'hashtag_counter' in globals() else "", dict(sentiment_counts), post_summary)

if st.session_state.creative_lines:
    st.markdown("#### ✨ Generated Lines")
    for line in st.session_state.creative_lines.split("\n"):
        if line.strip():
            st.markdown(f"✅ {line.strip()}")

# Hashtag wordcloud and top hashtags side-by-side & smaller
st.subheader("🌈 Hashtag Cloud & Top Hashtags")
col1, col2 = st.columns([1, 1])
with col1:
    small_freq = dict(Counter(hashtag_counter).most_common(40) if 'hashtag_counter' in globals() else {"empty":1})
    wc = WordCloud(width=400, height=160, max_font_size=40, background_color="white").generate_from_frequencies(small_freq)
    fig, ax = plt.subplots(figsize=(4,1.6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
with col2:
    st.markdown("**Top hashtags**")
    for tag, cnt in Counter(hashtag_counter).most_common(20) if 'hashtag_counter' in globals() else []:
        st.markdown(f"- #{tag} — {cnt} mentions")

# Explore by hashtag (compact)
st.markdown("### 🔍 Explore Posts by Hashtag")
selected_tag = st.selectbox("Select a hashtag", options=[""] + list(dict(hashtag_counter).keys()) if 'hashtag_counter' in globals() else [""])
if selected_tag:
    filtered = [p for p in top_posts_data if f"#{selected_tag}" in (p["text"] or "").lower()]
    st.markdown(f"Showing {len(filtered)} posts with #{selected_tag}")
    for i, post in enumerate(filtered):
        with st.expander(f"Post {i+1} — {post['sentiment']}, {post['emotion']}"):
            cols = st.columns([1, 5])
            with cols[0]:
                if post["avatar"]:
                    st.image(post["avatar"], width=72)
            with cols[1]:
                st.markdown(f"**Text:** {post['text']}")
                st.markdown(f"**Engagement:** 👍 {post['likes']} | 🔁 {post['shares']} | ▶️ {post['plays']} | 💬 {post['comments']}")
                if post["music"]:
                    st.markdown(f"**Music:** {post['music']} — {post.get('music_author','')}")

st.markdown("---")
st.markdown("Powered by Dentsu")
