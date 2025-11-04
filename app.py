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

GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.1-8b-instant"
APIFY_DATASET_URL = "https://api.apify.com/v2/datasets/fU0Y0M3aAPofsFXEi/items?format=json&view=overview&clean=true"
APIFY_TOKEN = "apify_api_356ndncSWmZqeg1kyAylb8djs1YnZB161LLe"

groq_client = Groq(api_key=GROQ_API_KEY)

# ─── Data Fetching & Analysis ──────────────────────────────────────────────────

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
        "Return a JSON with two fields: 'sentiment' (positive, neutral, negative) and 'emotion' "
        "(e.g., joy, stress, nostalgia, overwhelm, generosity, excitement, frustration, relief)."
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

# ─── Load and Process ──────────────────────────────────────────────────────────

hashtag_counter, sentiment_counts, emotional_barometer, top_keywords, top_posts_data = process_data(fetch_apify_data())

# ─── App Layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NZ Christmas Retail Trend Generator", layout="wide")
st.title("🎄 NZ Christmas Retail Trendspotter & Creative Generator V3")

# ─── Trend Spotter ─────────────────────────────────────────────────────────────

with st.container():
    st.subheader("🧠 Trend Spotter")

    total_emotions = sum(emotional_barometer.values())
    sorted_emotions = sorted(emotional_barometer.items(), key=lambda x: x[1], reverse=True)
    top_emotion = sorted_emotions[0][0]
    top_emotion_percent = round((sorted_emotions[0][1] / total_emotions) * 100, 1)

    keyword_summary = ", ".join(top_keywords) if top_keywords else "no clear keywords"
    music_mentions = [post["music"] for post in top_posts_data if post["music"]]
    top_music = Counter(music_mentions).most_common(1)
    music_summary = top_music[0][0] if top_music else "no dominant track"

    st.markdown(f"""
    ### 🔥 Vibe Check: {top_emotion.capitalize()} is in the air ({top_emotion_percent}% of posts)

    The streets of TikTok are humming with **{top_emotion}** — whether it's festive chaos, gift-induced panic, or full-blown BBQ euphoria. 
    This isn't your grandma's Christmas. It's messy, emotional, and gloriously Kiwi.

    **Trending keywords?** Think: {keyword_summary}.  
    Translation: we're talking tamariki tantrums, last-minute gifting, and the sacred art of wrapping with leftover newspaper.

    **Soundtrack of the moment?** It's all about **{music_summary}** — setting the tone for backyard dance-offs and driveway rants.

    So if you're crafting campaigns, lean into the madness. Celebrate the stress. Romanticize the rush.  
    Because this Christmas, realness is the new sparkle ✨.
    """)

    st.markdown("### 📊 Emotional Barometer")
    for emotion, count in sorted_emotions:
        percent = round((count / total_emotions) * 100, 1)
        st.markdown(f"- **{emotion.capitalize()}**: {count} posts ({percent}%)")

# ─── Top Posts Table ───────────────────────────────────────────────────────────

with st.container():
    st.subheader("🎄 Top Posts and Sentiment Overview")

    for i, post in enumerate(top_posts_data):
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

# ─── Creative Ideas ────────────────────────────────────────────────────────────

with st.container():
    st.subheader("✨ Creative Ideas Based on Trends")

    top_5_posts = sorted(top_posts_data, key=lambda x: x["likes"] + x["shares"] + x["comments"], reverse=True)[:5]
    post_summary = "\n".join([f"- \"{item['text'][:120].strip()}...\" ({item['sentiment']})" for item in top_5_posts])
    hashtag_summary = ", ".join(hashtag_counter.keys())
    keyword_summary = ", ".join(top_keywords)

    def generate_creative_lines(topics, sentiment_summary, post_summary):
        prompt = (
            "You're a creative assistant helping New Zealand retailers connect with shoppers during the Christmas season.\n\n"
            f"Trending hashtags: {topics}\n"
            f"Trending keywords: {keyword_summary}\n"
            f"Sentiment summary: {sentiment_summary}\n"
            f"Top 5 post excerpts:\n{post_summary}\n\n"
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

    st.markdown("**📌 Using today's top 5 posts for inspiration:**")
    st.markdown(post_summary)

    if st.button("🔁 Generate or Regenerate Ideas"):
        st.session_state.creative_lines = generate_creative_lines(hashtag_summary, sentiment_counts, post_summary)

    if st.session_state.creative_lines:
        st.markdown("#### ✨ Generated Lines")
        for line in st.session_state.creative_lines.split("\n"):
            if line.strip():
                st.markdown(f"✅ {line.strip()}")
