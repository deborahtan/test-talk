import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from groq import Groq
import requests
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ─── Setup ─────────────────────────────────────────────────────────────────────

groq_client = Groq(api_key=st.secrets["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"])
GROQ_MODEL = "llama-3.1-8b-instant"
APIFY_DATASET_URL = st.secrets["https://api.apify.com/v2/datasets/fU0Y0M3aAPofsFXEi/items?format=json&view=overview&clean=true"]
APIFY_TOKEN = "apify_api_356ndncSWmZqeg1kyAylb8djs1YnZB161LLe"

# ─── Data Fetching & Analysis ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_apify_data():
    response = requests.get(APIFY_DATASET_URL)
    response.raise_for_status()
    return response.json()

def analyze_text(text):
    prompt = (
        f"Analyze the following social media post:\n\n"
        f"\"{text}\"\n\n"
        "Return a JSON with two fields: 'sentiment' (positive, neutral, negative) and 'emotion' "
        "(e.g., joy, stress, nostalgia, overwhelm, generosity)."
    )
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"sentiment": "neutral", "emotion": "neutral"}

@st.cache_data(show_spinner=False)
def process_data(raw_data):
    top_posts_data = []
    sentiment_counts = {}
    emotional_barometer = {}
    top_hashtags = set()
    keyword_counter = Counter()

    for item in raw_data:
        text = item.get("text", "")
        analysis = analyze_text(text)
        sentiment = analysis["sentiment"]
        emotion = analysis["emotion"]

        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        emotional_barometer[emotion] = emotional_barometer.get(emotion, 0) + 1

        hashtags = [tag.strip("#") for tag in text.split() if tag.startswith("#")]
        top_hashtags.update(hashtags)

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

    return list(top_hashtags), sentiment_counts, emotional_barometer, top_keywords, top_posts_data

# ─── Load and Process ──────────────────────────────────────────────────────────

raw_data = fetch_apify_data()
top_hashtags, sentiment_counts, emotional_barometer, top_keywords, top_posts_data = process_data(raw_data)

# ─── App Layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NZ Christmas Retail Trend Generator", layout="wide")
st.title("🎄 NZ Christmas Retail Trendspotter & Creative Generator V3")

# ─── Sentiment Summary ─────────────────────────────────────────────────────────

with st.container():
    st.subheader("💬 Sentiment Summary")
    for k, v in sentiment_counts.items():
        st.markdown(f"- {k.capitalize()}: {v}")

# ─── Top Posts Table ───────────────────────────────────────────────────────────

with st.container():
    st.subheader("🎄 Top Posts and Sentiment Overview")

    for post in top_posts_data:
        st.markdown("---")
        cols = st.columns([1, 5])

        with cols[0]:
            st.image(post["avatar"], width=80)

        with cols[1]:
            st.markdown(f"**Author:** {post['author']}")
            st.markdown(f"**Posted:** {post['created']}")
            st.markdown(f"**Text:** {post['text']}")
            st.markdown(f"**Sentiment:** {post['sentiment'].capitalize()} | **Emotion:** {post['emotion'].capitalize()}")
            st.markdown(f"**🎵 Music:** {post['music']} by {post['music_author']}")
            st.markdown(f"**📊 Engagement:** 👍 {post['likes']} | 🔁 {post['shares']} | ▶️ {post['plays']} | 💬 {post['comments']}")
            st.markdown(f"[🔗 View on TikTok]({post['video_url']})")
            st.video(post["video_url"])

# ─── Trend Spotter ─────────────────────────────────────────────────────────────

with st.container():
    st.subheader("🧠 Trend Spotter")
    st.markdown("**📈 Top Keywords from Posts:**")
    if top_keywords:
        for kw in top_keywords:
            st.markdown(f"- {kw}")
    else:
        st.markdown("No trends detected.")

    st.markdown("**📊 Emotional Barometer (Post Volume):**")
    for emotion, count in emotional_barometer.items():
        st.markdown(f"- {emotion.capitalize()}: {count}")

    top_emotion = max(emotional_barometer, key=emotional_barometer.get)
    if top_emotion == "stress":
        st.warning("⚠️ Stress is trending. Campaigns should acknowledge pressure and offer relief or simplicity.")
    else:
        st.info(f"💡 Dominant emotion: **{top_emotion.capitalize()}** — lean into it for creative tone.")

# ─── Creative Ideas ────────────────────────────────────────────────────────────

with st.container():
    st.subheader("✨ Creative Ideas Based on Trends")

    post_summary = "\n".join([f"- \"{item['text']}\" ({item['sentiment']})" for item in top_posts_data])
    hashtag_summary = ", ".join(top_hashtags)
    keyword_summary = ", ".join(top_keywords)

    def generate_creative_lines(topics, sentiment_summary, post_summary):
        prompt = (
            "You're a creative assistant helping New Zealand retailers connect with shoppers during the Christmas season.\n\n"
            f"Trending hashtags: {topics}\n"
            f"Trending keywords: {keyword_summary}\n"
            f"Sentiment summary: {sentiment_summary}\n"
            f"Top posts today:\n{post_summary}\n\n"
            "Generate 3 short social lines that reflect current retail vibes.\n"
            "They should be emotionally resonant, cheeky, and Kiwi-flavoured — designed for campaign use.\n\n"
            "Tone: festive but dry, emotionally honest, and culturally grounded. Avoid clichés.\n"
            "Speak to the real stress and joy of a Kiwi Christmas: BBQ prep, tamariki meltdowns, last-minute gifting, and whānau dynamics.\n"
            "Prioritise emotional truth, campaign utility, and shareability."
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

    st.markdown("**📌 Using today's top posts for inspiration:**")
    st.markdown(post_summary)

    if st.button("🔁 Generate or Regenerate Ideas"):
        st.session_state.creative_lines = generate_creative_lines(hashtag_summary, sentiment_counts, post_summary)

    if st.session_state.creative_lines:
        st.markdown("#### ✨ Generated Lines")
        for line in st.session_state.creative_lines.split("\n"):
            if line.strip():
                st.markdown(f"✅ {line.strip()}")

# ─── Static Word Cloud ─────────────────────────────────────────────────────────

with st.container():
    st.markdown("---")
    st.subheader("🌟 Hashtag Word Cloud")

    hashtag_freq = {tag: 1 for tag in top_hashtags}
    wc = WordCloud(
        width=400,
        height=150,
        max_font_size=40,
        background_color="white",
        prefer_horizontal=1.0
    ).generate_from_frequencies(hashtag_freq)

    fig, ax = plt.subplots(figsize=(4, 1.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
