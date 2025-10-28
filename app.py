import streamlit as st
from groq import Groq

# ─── Setup ─────────────────────────────────────────────────────────────────────

groq_client = Groq(api_key="your-groq-api-key")  # Replace with your actual key
GROQ_MODEL = "mixtral-8x7b-32768"  # Use a valid Groq model

# ─── Mock Data ─────────────────────────────────────────────────────────────────

top_hashtags = ["giftguide2025", "kiwichristmas", "bbqseason", "nzpost", "stockingstuffers"]
top_post = "Just wrapped the last gift and realised I forgot Mum. Again. #kiwichristmas"

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

st.subheader("🎄 Sample Post")
st.text_area("Top Post", top_post, height=80)

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

def generate_creative_lines(topics, sentiment_summary, trending_post):
    prompt = (
        "You're a creative assistant helping New Zealand retailers connect with shoppers during the Christmas season.\n\n"
        f"Trending hashtags: {topics}\n"
        f"Sentiment summary: {sentiment_summary}\n"
        f"Sample post: \"{trending_post}\"\n\n"
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

if st.button("Generate"):
    lines = generate_creative_lines(top_hashtags, sentiment_counts, top_post)
    st.markdown("#### ✨ Generated Lines")
    for line in lines.split("\n"):
        if line.strip():
            st.markdown(f"✅ {line.strip()}")
