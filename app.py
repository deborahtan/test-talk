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

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = "llama-3.1-8b-instant"
groq_client = Groq(api_key=GROQ_API_KEY)

# Google Sheet (CSV export) — user-provided sheet and gid
SHEET_ID = "1nOUyqPniKKoje9JFTGXfjMF5POoKeM79tOCi75jYqtk"
GID = "690233754"
CSV_EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?gid={GID}&format=csv"

# Fallback Apify (kept for convenience if sheet not available)
APIFY_DATASET_URL = "https://api.apify.com/v2/datasets/fU0Y0M3aAPofsFXEi/items?format=json&view=overview&clean=true"
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def fetch_sheet_as_records(csv_url, timeout=20):
    try:
        resp = requests.get(csv_url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(resp.text), dtype=str)
        # convert to list of dicts for the pipeline
        return df.to_dict(orient="records"), "google_sheet"
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def fetch_apify_data():
    headers = {"Authorization": f"Bearer {APIFY_TOKEN}"} if APIFY_TOKEN else {}
    resp = requests.get(APIFY_DATASET_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()

def normalize_item(item):
    # Map common variants to canonical fields used in the app
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
        "sentiment": g("sentiment") or "",
        "emotion": g("emotion") or "",
        "likes": int(float(g("diggCount") or g("likeCount") or g("likes") or 0)),
        "shares": int(float(g("shareCount") or g("shares") or 0)),
        "plays": int(float(g("playCount") or g("plays") or g("views") or 0)),
        "comments": int(float(g("commentCount") or g("comments") or 0)),
        "music": g("musicMeta.musicName") or g("musicMeta/musicName") or g("music_name") or "",
        "music_author": g("musicMeta.musicAuthor") or g("musicMeta/musicAuthor") or "",
        "video_url": g("webVideoUrl") or g("video_url") or g("post_url") or "",
        "created": g("createTimeISO") or g("created_at") or g("post_date") or ""
    }

def analyze_text_locally(text):
    if not isinstance(text, str) or text.strip() == "":
        return {"sentiment": "neutral", "emotion": "unclear", "confidence": 0.0}
    pos_kw = re.compile(r"\b(win|love|happy|joy|great|awesome|yay|yes)\b", re.I)
    neg_kw = re.compile(r"\b(annoy|panic|stress|forgot|hate|angry|sad|oops)\b", re.I)
    if pos_kw.search(text):
        return {"sentiment": "positive", "emotion": "joy", "confidence": 0.8}
    if neg_kw.search(text):
        return {"sentiment": "negative", "emotion": "stress", "confidence": 0.8}
    return {"sentiment": "neutral", "emotion": "unclear", "confidence": 0.6}

@st.cache_data(show_spinner=False)
def process_records(records, use_groq=False, max_items=500):
    hashtag_counter = Counter()
    keyword_counter = Counter()
    sentiment_counts = Counter()
    emotional_barometer = Counter()
    posts = []

    for i, raw in enumerate(records[:max_items]):
        item = raw if isinstance(raw, dict) else {}
        r = normalize_item(item)

        if r["sentiment"] and r["emotion"]:
            analysis = {"sentiment": r["sentiment"], "emotion": r["emotion"], "confidence": 0.9}
        else:
            if use_groq:
                try:
                    prompt = (
                        f"Analyze the following social media post and return JSON: sentiment (positive/neutral/negative), "
                        f"emotion (joy, stress, nostalgia, overwhelm, generosity, excitement, frustration, relief), confidence (0-1).\n\nPost: \"{r['text']}\""
                    )
                    resp = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    analysis = json.loads(resp.choices[0].message.content)
                except Exception:
                    analysis = analyze_text_locally(r["text"])
            else:
                analysis = analyze_text_locally(r["text"])

        sentiment_counts[analysis["sentiment"]] += 1
        emotional_barometer[analysis["emotion"]] += 1

        tags = re.findall(r"#\w+", (r["text"] or "").lower())
        hashtag_counter.update([t.lstrip("#") for t in tags])

        words = re.findall(r"\b[a-zA-Z]{3,}\b", (r["text"] or "").lower())
        keywords = [w for w in words if w not in ENGLISH_STOP_WORDS]
        keyword_counter.update(keywords)

        posts.append({
            "author": r["author"],
            "avatar": r["avatar"],
            "text": r["text"],
            "sentiment": analysis["sentiment"],
            "emotion": analysis["emotion"],
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

# ─── Load dataset: try Google Sheet CSV first, then Apify ────────────────────────
records, source_label = fetch_sheet_as_records(CSV_EXPORT_URL)
if not records:
    try:
        records = fetch_apify_data()
        source_label = "apify"
    except Exception as e:
        st.error(f"Failed to load dataset from Google Sheet or Apify: {e}")
        records = []
        source_label = "none"

# ─── Process records (fast local analysis by default; set use_groq=True to call model) ─
hashtag_counter, sentiment_counts, emotional_barometer, top_keywords, top_posts_data = process_records(records, use_groq=False, max_items=500)

# ─── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NZ Christmas Retail Trendspotter", layout="wide")
st.title("🎄 NZ Christmas Retail Trendspotter")
st.caption(f"Source: {source_label} — lightweight analysis (toggle use_groq in code for model calls)")

st.sidebar.title("📅 Date Range")
date_range = st.sidebar.radio("View trends from:", ["Last 24 hours", "Last 7 days"])

# Vibe Check
st.subheader("🔥 Vibe Check")
if emotional_barometer:
    top_emotion = max(emotional_barometer, key=emotional_barometer.get)
else:
    top_emotion = "unclear"
st.markdown(f"**Dominant Emotion:** {top_emotion.capitalize()}")
st.markdown(f"**Trending Keywords:** {', '.join(top_keywords[:10])}")

top_songs = list({p["music"] for p in top_posts_data if p["music"]})[:3]
if top_songs:
    st.markdown(f"**Top Songs:** {', '.join(top_songs)}")

# Top posts viewer
st.subheader("🎥 Top Posts")
if "post_offset" not in st.session_state:
    st.session_state.post_offset = 0

visible_posts = top_posts_data[st.session_state.post_offset:st.session_state.post_offset + 10]
for i, post in enumerate(visible_posts):
    with st.expander(f"Post {i+1 + st.session_state.post_offset} — {post['sentiment'].capitalize()}, {post['emotion'].capitalize()}"):
        cols = st.columns([1, 5])
        with cols[0]:
            if post["avatar"]:
                st.image(post["avatar"], width=80)
        with cols[1]:
            st.markdown(f"**Posted:** {post['created']}")
            st.markdown(f"**Text:** {post['text']}")
            st.markdown(f"**🎵 Music:** {post['music']} by {post['music_author']}")
            st.markdown(f"**📊 Engagement:** 👍 {post['likes']} | 🔁 {post['shares']} | ▶️ {post['plays']} | 💬 {post['comments']}")
            if post["video_url"]:
                st.markdown(f"[🔗 View]({post['video_url']})")
                try:
                    st.video(post["video_url"], format="video/mp4", start_time=0)
                except Exception:
                    pass

if st.button("Load 10 More"):
    st.session_state.post_offset += 10

# Creative generator (strategic summary)
st.subheader("✨ Creative Ideas Based on Trends")
top_20_posts = sorted(top_posts_data, key=lambda x: x["likes"] + x["shares"] + x["comments"], reverse=True)[:20]
post_summary = "\n".join([f"- \"{(p['text'] or '')[:100].strip()}...\" ({p['sentiment']}, {p['emotion']})" for p in top_20_posts])
hashtag_summary = ", ".join(list(hashtag_counter.keys())[:20])
keyword_summary = ", ".join(top_keywords[:10])

def copy_to_clipboard(text):
    components.html(f"""
        <script>
        function copyText() {{
            navigator.clipboard.writeText(`{text}`);
        }}
        </script>
        <button onclick="copyText()">📋 Copy to Clipboard</button>
    """, height=40)

def generate_creative_lines_groq(topics, sentiment_summary, post_summary):
    prompt = (
        "You're a creative assistant helping New Zealand retailers connect with shoppers during the Christmas season.\n\n"
        f"Trending hashtags: {topics}\n"
        f"Sentiment summary: {sentiment_summary}\n"
        f"Top posts:\n{post_summary}\n\n"
        "Generate 3 short social lines, emotionally resonant, cheeky and Kiwi-flavoured."
    )
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

if "creative_summary" not in st.session_state:
    st.session_state.creative_summary = ""

st.markdown("**Using today's top posts for inspiration:**")
st.markdown(post_summary)

if st.button("Generate Strategic Summary (Groq)"):
    st.session_state.creative_summary = generate_creative_lines_groq(hashtag_summary, dict(sentiment_counts), post_summary)

if st.session_state.creative_summary:
    st.markdown("#### Strategic Summary")
    st.markdown(st.session_state.creative_summary)
    copy_to_clipboard(st.session_state.creative_summary)

# Hashtag cloud
st.subheader("🌈 Hashtag Word Cloud")
top_30_hashtags = dict(Counter(hashtag_counter).most_common(30))
with st.expander("Click to expand hashtag cloud"):
    wc = WordCloud(width=600, height=200, max_font_size=50, background_color="white").generate_from_frequencies(top_30_hashtags or {"empty":1})
    fig, ax = plt.subplots(figsize=(6,2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

# Explore by hashtag
st.markdown("### Explore Posts by Hashtag")
selected_tag = st.selectbox("Select a hashtag to view relevant posts", options=[""] + list(top_30_hashtags.keys()))
if selected_tag:
    filtered = [p for p in top_posts_data if f"#{selected_tag}" in (p["text"] or "").lower()]
    st.markdown(f"Showing {len(filtered)} posts with #{selected_tag}")
    for i, post in enumerate(filtered):
        with st.expander(f"Post {i+1} — {post['sentiment']}, {post['emotion']}"):
            cols = st.columns([1,5])
            with cols[0]:
                if post["avatar"]:
                    st.image(post["avatar"], width=80)
            with cols[1]:
                st.markdown(f"**Text:** {post['text']}")
                st.markdown(f"**Engagement:** 👍 {post['likes']} | 🔁 {post['shares']} | ▶️ {post['plays']} | 💬 {post['comments']}")
                if post["video_url"]:
                    st.markdown(f"[View on TikTok]({post['video_url']})")

st.markdown("---")
st.markdown("Powered by Dentsu")
