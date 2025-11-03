import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from groq import Groq

# ─── Setup ─────────────────────────────────────────────────────────────────────

groq_client = Groq(api_key="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")  # Replace with your actual key
GROQ_MODEL = "llama-3.1-8b-instant"

# ─── Mock Data ─────────────────────────────────────────────────────────────────

top_hashtags = [
    "christmas", "kiwichristmas", "bbqseason", "christmasgift", "stockingstuffers"
]

sentiment_counts = {
    "positive": 34,
    "neutral": 12,
    "negative": 8
}

emotional_barometer = {
    "joy": 28,
    "stress": 18,
    "nostalgia": 12,
    "overwhelm": 10,
    "generosity": 16
}

new_trends = [
    "BBQ kits with free delivery",
    "Last-minute spa vouchers",
    "DIY stocking filler hacks"
]

top_posts_data = [
    {"post": "Just wrapped the last gift and realised I forgot Mum. Again. #kiwichristmas", "sentiment": "negative"},
    {"post": "BBQ smoke, pōhutukawa shade, and a gift that actually lands — now that’s a win. #bbqseason", "sentiment": "positive"},
    {"post": "Stocking stuffers under $20 that won’t make you look like you forgot — even if you did. #stockingstuffers", "sentiment": "positive"}
]

# ─── App Layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NZ Christmas Retail Trend Generator", layout="wide")
st.title("🎄 NZ Christmas Retail Trendspotter & Creative Generator V1 Draft")

# ─── Sentiment Summary ─────────────────────────────────────────────────────────

with st.container():
    st.subheader("💬 Sentiment Summary")
    for k, v in sentiment_counts.items():
        st.markdown(f"- {k.capitalize()}: {v}")

# ─── Top Posts Table ───────────────────────────────────────────────────────────

with st.container():
    st.subheader("🎄 Top Posts and Sentiment Overview")
    posts_df = pd.DataFrame(top_posts_data)
    st.dataframe(posts_df, use_container_width=True)

# ─── Trend Spotter ─────────────────────────────────────────────────────────────

with st.container():
    st.subheader("🧠 Trend Spotter")
    st.markdown("**📈 New Trends Identified:**")
    for trend in new_trends:
        st.markdown(f"- {trend}")

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
    st.markdown("""
    These lines reflect current sentiment — a mix of excitement, stress, and Kiwi practicality:

    - ✅ *“Christmas magic? Nah, it’s just you panic-buying candles and hoping NZ Post delivers on time.”*  
    - ✅ *“BBQ smoke, pōhutukawa shade, and a gift that actually lands — now that’s a win.”*  
    - ✅ *“Stocking stuffers under $20 that won’t make you look like you forgot — even if you did.”*  
    - ✅ *“Grill kits, gift cards, and a dash of emotional damage — your Christmas sorted.”*
    """)

# ─── Live Creative Generation ──────────────────────────────────────────────────

with st.container():
    st.subheader("📝 Generate More Creative Lines")

    post_summary = "\n".join([f"- \"{item['post']}\" ({item['sentiment']})" for item in top_posts_data])

    def generate_creative_lines(topics, sentiment_summary, post_summary):
        prompt = (
            "You're a creative assistant helping New Zealand retailers connect with shoppers during the Christmas season.\n\n"
            f"Trending hashtags: {topics}\n"
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
        st.session_state.creative_lines = generate_creative_lines(top_hashtags, sentiment_counts, post_summary)

    if st.session_state.creative_lines:
        st.markdown("#### ✨ Generated Lines")
        for line in st.session_state.creative_lines.split("\n"):
            if line.strip():
                st.markdown(f"✅ {line.strip()}")

# ─── Static Word Cloud (Final Section) ─────────────────────────────────────────

with st.container():
    st.markdown("---")
    st.subheader("🌟 Hashtag Word Cloud - I am under construction")

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
