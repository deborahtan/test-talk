import streamlit as st
import pandas as pd
from groq import Groq

# ─── Setup ─────────────────────────────────────────────────────────────────────

groq_client = Groq(api_key="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")  # Replace with your actual key
GROQ_MODEL = "llama-3.1-8b-instant"

# ─── Mock Data ─────────────────────────────────────────────────────────────────

top_hashtags = ["giftguide2025", "kiwichristmas", "bbqseason", "nzpost", "stockingstuffers"]

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
    "Rural shipping anxiety memes",
    "DIY stocking filler hacks"
]

top_posts_data = [
    {"post": "Just wrapped the last gift and realised I forgot Mum. Again. #kiwichristmas", "sentiment": "negative"},
    {"post": "BBQ smoke, pōhutukawa shade, and a gift that actually lands — now that’s a win. #bbqseason", "sentiment": "positive"},
    {"post": "She said ‘no fuss this year’ — so you bought her a spa voucher and cried in the carpark. #giftguide2025", "sentiment": "neutral"},
    {"post": "Stocking stuffers under $20 that won’t make you look like you forgot — even if you did. #stockingstuffers", "sentiment": "positive"},
    {"post": "Rural delivery panic is real. NZ Post, we believe in you. #nzpost", "sentiment": "stress"}
]

# ─── App Layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NZ Christmas Retail Trend Generator", layout="wide")
st.title("🎄 NZ Christmas Retail Trend Listener + Creative Generator")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🎯 Top Hashtags")
    st.text_area("Top Hashtags", "\n".join([f"#{tag}" for tag in top_hashtags]), height=100)

with col2:
    st.subheader("💬 Sentiment Summary")
    sentiment_text = "\n".join([f"{k.capitalize()}: {v}" for k, v in sentiment_counts.items()])
    st.text_area("Sentiment", sentiment_text, height=100)

# ─── Top Posts Table ───────────────────────────────────────────────────────────

st.subheader("🎄 Top Posts and Sentiment Overview")
posts_df = pd.DataFrame(top_posts_data)
st.dataframe(posts_df, use_container_width=True)

# ─── Trend Spotter ─────────────────────────────────────────────────────────────

st.markdown("---")
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

st.markdown("---")
st.subheader("✨ Creative Ideas Based on Trends")

st.markdown("""
These lines reflect current sentiment — a mix of excitement, stress, and Kiwi practicality:

- ✅ *“Christmas magic? Nah, it’s just you panic-buying candles and hoping NZ Post delivers on time.”*  
- ✅ *“BBQ smoke, pōhutukawa shade, and a gift that actually lands — now that’s a win.”*  
- ✅ *“She said ‘no fuss this year’ — so you bought her a spa voucher and cried in the carpark.”*  
- ✅ *“Stocking stuffers under $20 that won’t make you look like you forgot — even if you did.”*  
- ✅ *“Grill kits, gift cards, and a dash of emotional damage — your Christmas sorted.”*
""")

# ─── Live Creative Generation ──────────────────────────────────────────────────

st.markdown("---")
st.subheader("📝 Generate More Creative Lines")

# Combine all top posts and their sentiment into a readable summary
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
        "Speak to the real stress and joy of a Kiwi Christmas: rural delivery panic, BBQ prep, tamariki meltdowns, last-minute gifting, and whānau dynamics.\n"
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

# Initialize session state
if "creative_lines" not in st.session_state:
    st.session_state.creative_lines = ""

# Show the post summary being used
st.markdown("**📌 Using today's top posts for inspiration:**")
st.markdown(post_summary)

# Button to generate or regenerate
if st.button("🔁 Generate or Regenerate Ideas"):
    st.session_state.creative_lines = generate_creative_lines(top_hashtags, sentiment_counts, post_summary)

# Display generated lines
if st.session_state.creative_lines:
    st.markdown("#### ✨ Generated Lines")
    for line in st.session_state.creative_lines.split("\n"):
        if line.strip():
            st.markdown(f"✅ {line.strip()}")
