import os
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

# Optional: keep Groq integration but not required for emotion analysis
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Google Sheet CSV export (user-provided)
SHEET_ID = "1nOUyqPniKKoje9JFTGXfjMF5POoKeM79tOCi75jYqtk"
GID = "690233754"
CSV_EXPORT_URL = f"https://docs.google.com/spreadsheets/d/e/2PACX-1vTH33TC1xTixH8TWGAOUUe3o-UIFX82HMaBv8BlI4KA5UnJxYs50QBitDUwXB_Jkl8M52CdE66s_XDx/pub?gid=690233754&single=true&output=csv"

# Hugging Face model choice (small, emotion classifier)
HF_EMOTION_MODEL = os.getenv("HF_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
# Limit model runs to top N posts to keep resource use small
EMOTION_MODEL_MAX = int(os.getenv("EMOTION_MODEL_MAX", "200"))

# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_sheet_as_records(csv_url, timeout=20):
    try:
        resp = requests.get(csv_url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(resp.text), dtype=str)
        return df.to_dict(orient="records"), "google_sheet"
    except Exception as e:
        st.warning(f"Failed to fetch sheet CSV: {e}")
        return None, None

def normalize_item(item):
    # Map likely keys to canonical fields
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
        "author": g("authorMeta.name") or g("authorMeta/name") or g("author") or g("author_name"),
        "avatar": g("authorMeta.avatar") or g("authorMeta/avatar") or g("authorAvatar") or g("author_image"),
        "text": g("text") or g("caption") or g("description") or g("post_text") or "",
        "likes": safe_int(g("diggCount") or g("likeCount") or g("likes") or 0),
        "shares": safe_int(g("shareCount") or g("shares") or 0),
        "plays": safe_int(g("playCount") or g("plays") or g("views") or 0),
        "comments": safe_int(g("commentCount") or g("comments") or 0),
        "music": g("musicMeta.musicName") or g("musicMeta/music_name") or "",
        "music_author": g("musicMeta.musicAuthor") or g("musicMeta/music_author") or "",
        "video_url": g("webVideoUrl") or g("video_url") or g("post_url") or "",
        "created": g("createTimeISO") or g("created_at") or g("post_date") or ""
    }

def safe_int(v):
    try:
        return int(float(v or 0))
    except Exception:
        return 0

# ─── Emotion model (HF) with small cache ───────────────────────────────────────
try:
    from transformers import pipeline
except Exception:
    pipeline = None

@lru_cache(maxsize=1)
def get_emotion_pipeline(model_name=HF_EMOTION_MODEL, device=-1):
    """
    Returns a HF pipeline for text-classification (emotion).
    device=-1 uses CPU. Keep cached to avoid reloading.
    """
    if pipeline is None:
        return None
    try:
        nlp = pipeline("text-classification", model=model_name, return_all_scores=True, device=device)
        return nlp
    except Exception as e:
        st.warning(f"Failed to load HF emotion model {model_name}: {e}")
        return None

def map_model_label_to_emotion(label):
    label = label.lower()
    map_emotion = {
        "joy": "happy",
        "happiness": "happy",
        "love": "generous",
        "sadness": "sad",
        "sad": "sad",
        "anger": "angry",
        "angry": "angry",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral",
        # add more mappings if model uses different labels
    }
    return map_emotion.get(label, label)

def analyze_text_with_model(text, model_pipeline=None, min_confidence=0.55):
    """
    Returns {emotion, sentiment, confidence}
    - emotion: one of happy, sad, angry, fear, surprise, generous, neutral, unclear
    - sentiment: positive/neutral/negative/unclear derived from emotion
    - confidence: top score
    """
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
                if emotion in ("happy", "generous", "surprise"):
                    sentiment = "positive"
                elif emotion in ("sad", "angry", "fear"):
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                return {"emotion": emotion, "sentiment": sentiment, "confidence": conf}
        except Exception:
            # model failed, fall through to heuristic
            pass
    # fallback heuristic
    return analyze_text_heuristic(text)

def analyze_text_heuristic(text):
    # simple keyword heuristics for common emotions
    if not text or not isinstance(text, str):
        return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.0}
    pos_kw = re.compile(r"\b(win|love|happy|joy|great|awesome|yay|yes|delight|best|cheer)\b", re.I)
    neg_kw = re.compile(r"\b(annoy|panic|stress|forgot|hate|angry|sad|oops|frustrat|sick of|upset)\b", re.I)
    nostalgic_kw = re.compile(r"\b(memory|remember|nostalgia|nostalgic|back then|old times|used to)\b", re.I)
    generous_kw = re.compile(r"\b(give|donate|gift|generous|share|help)\b", re.I)
    fun_kw = re.compile(r"\b(fun|lol|haha|hilarious|silly)\b", re.I)
    if pos_kw.search(text):
        return {"emotion": "happy", "sentiment": "positive", "confidence": 0.65}
    if neg_kw.search(text):
        return {"emotion": "angry", "sentiment": "negative", "confidence": 0.65}
    if nostalgic_kw.search(text):
        return {"emotion": "nostalgic", "sentiment": "neutral", "confidence": 0.7}
    if generous_kw.search(text):
        return {"emotion": "generous", "sentiment": "positive", "confidence": 0.7}
    if fun_kw.search(text):
        return {"emotion": "fun", "sentiment": "positive", "confidence": 0.6}
    return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.5}

# ─── Other helpers ────────────────────────────────────────────────────────────

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
    """
    use_model True => attempt HF model for top EMOTION_MODEL_MAX items, then fallback
    We still analyze all rows, but only run the HF model on the first EMOTION_MODEL_MAX entries to control cost.
    """
    hashtag_counter = Counter()
    keyword_counter = Counter()
    sentiment_counts = Counter()
    emotional_barometer = Counter()
    posts = []

    # prepare model pipeline (cached)
    model_pipeline = get_emotion_pipeline() if use_model else None

    for i, raw in enumerate((records or [])[:max_items]):
        item = raw if isinstance(raw, dict) else {}
        r = normalize_item(item)
        # Decide whether to use model on this row
        if use_model and model_pipeline and i < EMOTION_MODEL_MAX:
            analysis = analyze_text_with_model(r["text"], model_pipeline=model_pipeline, min_confidence=0.55)
        else:
            # either model disabled, model missing, or beyond EMOTION_MODEL_MAX
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
            "video_url": r["video_url"],
            "created": r["created"]
        })

    top_keywords = [k for k, _ in keyword_counter.most_common(10)]
    return dict(hashtag_counter), dict(sentiment_counts), dict(emotional_barometer), top_keywords, posts

# ─── Load sheet and run processing ─────────────────────────────────────────────
records, source_label = fetch_sheet_as_records(CSV_EXPORT_URL)
if not records:
    st.error("Failed to fetch Google Sheet CSV. Check sheet permissions or the SHEET_ID/GID.")
    records = []
    source_label = "none"

# Process records: use_model=True attempts HF model (limited by EMOTION_MODEL_MAX)
hashtag_counter, sentiment_counts, emotional_barometer, top_keywords, top_posts_data = process_records(records, use_model=True, max_items=500)

# derive totals and percentages
total_posts = sum(sentiment_counts.values()) or 1
sentiment_pct = {k: round(v / total_posts * 100, 1) for k, v in sentiment_counts.items()}

# most-used emoji
texts = [p.get("text", "") for p in top_posts_data]
top_emoji, top_emoji_count = most_used_emoji(texts)

# top themes from top posts (simple token frequency)
def top_themes_from_posts(posts, top_n=6):
    tokens = []
    for p in posts:
        text = p.get("text", "") or ""
        tokens += re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    c = Counter(tokens)
    return [t for t, _ in c.most_common(top_n)]

top_themes = top_themes_from_posts(top_posts_data, top_n=6)

# songs appearing
songs = Counter()
for p in top_posts_data:
    if p.get("music"):
        songs[p["music"]] += 1
top_songs = [s for s, _ in songs.most_common(5)]

# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NZ Christmas Retail Trendspotter", layout="wide")
st.title("🎄 NZ Christmas Retail Trendspotter")
st.caption(f"Source: {source_label} — emotion model max {EMOTION_MODEL_MAX} rows")

st.sidebar.title("📅 Date Range")
_ = st.sidebar.radio("View trends from:", ["Last 24 hours", "Last 7 days"])

# Vibe Check — emotion based
st.subheader("🔥 Vibe Check — Emotion Summary")
st.markdown("Synthesis of dominant emotions and top themes from the sample of posts. Low-confidence results are labelled 'unclear'.")

if top_themes:
    st.markdown("**Top themes:** " + ", ".join(top_themes))
else:
    st.markdown("**Top themes:** none found")

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

# Generate Creatives (Groq optional)
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
    st.session_state.creative_lines = generate_creatives_groq(", ".join(list(hashtag_counter.keys())[:20]), dict(sentiment_counts), post_summary)

if st.session_state.creative_lines:
    st.markdown("#### ✨ Generated Lines")
    for line in st.session_state.creative_lines.split("\n"):
        if line.strip():
            st.markdown(f"✅ {line.strip()}")

# Hashtag wordcloud and top hashtags side-by-side & smaller
st.subheader("🌈 Hashtag Cloud & Top Hashtags")
col1, col2 = st.columns([1, 1])
with col1:
    small_freq = dict(Counter(hashtag_counter).most_common(40) or {"empty":1})
    wc = WordCloud(width=400, height=160, max_font_size=40, background_color="white").generate_from_frequencies(small_freq)
    fig, ax = plt.subplots(figsize=(4,1.6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
with col2:
    st.markdown("**Top hashtags**")
    for tag, cnt in Counter(hashtag_counter).most_common(20):
        st.markdown(f"- #{tag} — {cnt} mentions")

# Explore by hashtag (compact)
st.markdown("### 🔍 Explore Posts by Hashtag")
selected_tag = st.selectbox("Select a hashtag", options=[""] + list(dict(hashtag_counter).keys()))
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
