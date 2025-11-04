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

# ─── API Keys and Clients ──────────────────────────────────────────────────────

GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3-1-8b-instant"
APIFY_DATASET_URL = "https://api.apify.com/v2/datasets/your_dataset_id/items?format=json&clean=true"
APIFY_TOKEN = "your_apify_token"

groq_client = Groq(api_key=GROQ_API_KEY)

# ─── Data Fetching and Analysis ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_apify_data():
    headers = {"Authorization": f"Bearer {APIFY_TOKEN}"}
    response = requests.get(APIFY_DATASET_URL, headers=headers)
    response.raise_for_status()
    return response.json()

def analyze_text(text):
    prompt = (
        f"Analyze the following social media post:\n\n"
        f"\"{text}\"\n\n"
        "Return a JSON with three fields: 'sentiment' (positive, neutral, negative), 'emotion' "
        "(e.g., joy, stress, nostalgia, overwhelm, generosity, excitement, frustration, relief), and 'confidence' (0 to 1)."
    )
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("confidence", 0) < 0.5:
            result["sentiment"] = "neutral"
            result["emotion"] = "unclear"
        return result
    except Exception:
        return {"sentiment": "neutral", "emotion": "unclear", "confidence": 0}

@st.cache_data(show_spinner=False)
def process_data(raw_data):
    top_posts_data = []
    sentiment_counts = {}
    emotional_barometer = {}
    hashtag_counter = Counter()
    keyword_counter = Counter()

    for item in raw_data:
        text = item.get("text", "")
        analysis = analyze_text(text)
        sentiment = analysis["sentiment"]
        emotion = analysis["emotion"]

        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        emotional_barometer[emotion] = emotional_barometer.get(emotion, 0) + 1

        hashtags = [tag.strip("#") for tag in text.split() if tag.startswith("#")]
        hashtag_counter.update(hashtags)

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in ENGLISH_STOP_WORDS]
        keyword_counter.update(keywords)

        top_posts_data.append({
            "author": item.get("authorMeta.name"),
            "avatar": item.get("authorMeta.avatar"),
            "text": text,
            "sentiment": sentiment,
            "emotion": emotion,
            "likes": item.get("diggCount", 0),
            "shares": item.get("shareCount", 0),
            "plays": item.get("playCount", 0),
            "comments": item.get("commentCount", 0),
            "music": item.get("musicMeta.musicName", ""),
            "music_author": item.get("musicMeta.musicAuthor", ""),
            "video_url": item.get("webVideoUrl", ""),
            "created": item.get("createTimeISO", "")
        })

    top_keywords = [kw for kw, _ in keyword_counter.most_common(5)]
    return hashtag_counter, sentiment_counts, emotional_barometer, top_keywords, top_posts_data

# ─── Streamlit Page Setup ──────────────────────────────────────────────────────

st.set_page_config(page_title="NZ Christmas Retail Trendspotter", layout="wide")
st.title("🎄 NZ Christmas Retail Trendspotter")
st.caption("Real-time TikTok insights for Kiwi campaigns")

# ─── Sidebar Date Range Selector ───────────────────────────────────────────────

st.sidebar.title("📅 Date Range")
date_range = st.sidebar.radio("View trends from:", ["Last 24 hours", "Last 7 days"])

# ─── Load and Process Data ─────────────────────────────────────────────────────

raw_data = fetch_apify_data()
hashtag_counter, sentiment_counts, emotional_barometer, top_keywords, top_posts_data = process_data(raw_data)

# ─── Trend Summary ─────────────────────────────────────────────────────────────

st.subheader("🔥 Vibe Check")
top_emotion = max(emotional_barometer, key=emotional_barometer.get)
st.markdown(f"**Dominant Emotion:** {top_emotion.capitalize()}")
st.markdown(f"**Trending Keywords:** {', '.join(top_keywords)}")

top_songs = list({post["music"] for post in top_posts_data if post["music"]})[:3]
if top_songs:
    st.markdown(f"**Top Songs:** {', '.join(top_songs)}")

# ─── Top Posts Viewer ──────────────────────────────────────────────────────────

st.subheader("🎥 Top TikTok Posts")
if "post_offset" not in st.session_state:
    st.session_state.post_offset = 0

visible_posts = top_posts_data[st.session_state.post_offset:st.session_state.post_offset + 10]
for i, post in enumerate(visible_posts):
    with st.expander(f"📹 Post {i+1 + st.session_state.post_offset} by {post['author']} — {post['sentiment'].capitalize()}, {post['emotion'].capitalize()}"):
        cols = st.columns([1, 5])
        with cols[0]:
            st.image(post["avatar"], width=80)
        with cols[1]:
            st.markdown(f"**Posted:** {post['created']}")
            st.markdown(f"**Text:** {post['text']}")
            st.markdown(f"**🎵 Music:** {post['music']} by {post['music_author']}")
            st.markdown(f"**📊 Engagement:** 👍 {post['likes']} | 🔁 {post['shares']} | ▶️ {post['plays']} | 💬 {post['comments']}")
            st.markdown(f"[🔗 View on TikTok]({post['video_url']})")
            st.video(post["video_url"], format="video/mp4", start_time=0)

if st.button("🔁 Load 10 More"):
    st.session_state.post_offset += 10

# ─── Creative Line Generator ───────────────────────────────────────────────────

st.subheader("✨ Generate Creative Lines")

top_20_posts = sorted(top_posts_data, key=lambda x: x["likes"] + x["shares"] + x["comments"], reverse=True)[:20]
post_summary = "\n".join([f"- \"{item['text'][:100].strip()}...\" ({item['sentiment']}, {item['emotion']})" for item in top_20_posts])
hashtag_summary = ", ".join(hashtag_counter.keys())
keyword_summary = ", ".join(top_keywords)

def copy_to_clipboard(text):
    components.html(f"""
        <script>
        function copyText() {{
            navigator.clipboard.writeText(`{text}`);
        }}
        </script>
        <button onclick="copyText()">📋 Copy to Clipboard</button>
    """, height=40)

def generate_creative_lines(topics, sentiment_summary, post_summary):
    prompt = (
        "Come up with some witty one-liners that are Christmas-themed but also culturally relevant in 2025's Xmas season.\n\n"
        f"Trending hashtags: {topics}\n"
        f"Trending keywords: {keyword_summary}\n"
        f"Sentiment summary: {sentiment_summary}\n"
        f"Top 20 post excerpts:\n{post_summary}\n\n"
        "Return 3 witty, emotionally honest lines that Kiwi retailers could use in Christmas campaigns."
    )
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating lines: {e}"

if "creative_lines" not in st.session_state:
    st.session_state.creative_lines = ""

if st.button("🧠 Generate Creative Lines"):
    st.session_state.creative_lines = generate_creative_lines(hashtag_summary, sentiment_counts, post_summary)

if st.session_state.creative_lines:
    st.markdown("#### 💡 Creative Lines")
    st.markdown(st.session_state.creative_lines)
    copy_to_clipboard(st.session_state.creative_lines)

# ─── Expandable Word Cloud ─────────────────────────────────────────────────────

st.subheader("Word Cloud")

top_30_hashtags = dict(hashtag_counter.most_common(30))

with st.expander("📈 Click to expand hashtag cloud"):
    wc = WordCloud(
        width=400,
        height=120,
        max_font_size=30,
        background_color="white",
        prefer_horizontal=1.0
    ).generate_from_frequencies(top_30_hashtags)

    fig, ax = plt.subplots(figsize=(4, 1.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

# ─── Hashtag Filter ────────────────────────────────────────────────────────────

st.markdown("### 🔍 Explore Posts by Hashtag")
tag_options = [f"{tag} ({count} posts)" for tag, count in top_30_hashtags.items()]
selected_tag_raw = st.selectbox("Select a hashtag to view relevant posts", options=[""] + tag_options)

if selected_tag_raw and selected_tag_raw != "":
    selected_tag = selected_tag_raw.split(" ")[0]
    filtered_posts = [post for post in top_posts_data if f"#{selected_tag}" in post["text"]]
    st.markdown(f"#### 🎯 Showing {len(filtered_posts)} posts with **#{selected_tag}**")

    for i, post in enumerate(filtered_posts):
        with st.expander(f"📹 Post {i+1} by {post['author']} — {post['sentiment'].capitalize()}, {post['emotion'].capitalize()}"):
            cols = st.columns([1, 5])
            with cols[0]:
                st.image(post["avatar"], width=80)
            with cols[1]:
                st.markdown(f"**Posted:** {post['created']}")
                st.markdown(f"**Text:** {post['text']}")
                st.markdown(f"**🎵 Music:** {post['music']} by {post['music_author']}")
                st.markdown(f"**📊 Engagement:** 👍 {post['likes']} | 🔁 {post['shares']} | ▶️ {post['plays']} | 💬 {post['comments']}")
                st.markdown(f"[🔗 View on TikTok]({post['video_url']})")
                st.video(post["video_url"], format="video/mp4", start_time=0)

# ─── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("Powered by Dentsu")
