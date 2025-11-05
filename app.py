# app.py
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
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datetime import datetime
from functools import lru_cache

# ------------------------------
# Styling (Dentsu)
# ------------------------------
st.set_page_config(
    page_title="Dentsu Conversational Analytics: Trendspotter",
    page_icon="https://img.icons8.com/ios11/16/000000/dashboard-gauge.png",
    layout="wide"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    .stSidebar { min-width: 336px; }
    .stSidebar .stHeading { color: #FAFAFA; }
    .stSidebar .stElementContainer { width: auto; }
    .stAppHeader { display: none; }

    .stMainBlockContainer div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] > div[data-testid="stButton"] { text-align: center; }
    .stMainBlockContainer div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] > div[data-testid="stButton"] button {
        color: #FAFAFA;
        border: 1px solid #FAFAFA33;
        transition: all 0.3s ease;
        background-color: #0E1117;
        width: fit-content;
    }
    .stMainBlockContainer div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] > div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar header logo and controls
# ------------------------------
with st.sidebar:
    st.image("https://www.dentsu.com/assets/images/main-logo-alt.png", width=160)
    if st.button("🧹 Start New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.question_history = []
        st.rerun()
    st.header("Dentsu Conversational Analytics: Trendspotter")
    st.markdown("""
    **How to use**
    - The Trendspotter synthesises top posts across TikTok, Meta, and Instagram and analyses sentiment
    - Click on the Generate creative lines button to see automatically generated ideas based on key trends
    - The assistant responds with creative copy ideas 
    - Conversation context is remembered
    """)
    st.divider()
    st.subheader("📋 Recent Questions")
    if "question_history" not in st.session_state:
        st.session_state.question_history = []
    today = datetime.now().date()
    yesterday = today - pd.Timedelta(days=1)
    today_qs = [q for q in st.session_state.question_history if q["date"] == today]
    yesterday_qs = [q for q in st.session_state.question_history if q["date"] == yesterday]
    if today_qs:
        st.markdown("**Today**")
        for q in reversed(today_qs[-5:]):
            if st.button(q["text"][:50] + ("..." if len(q["text"]) > 50 else ""), key=f"today_{q['timestamp']}", use_container_width=True):
                st.session_state.rerun_question = q["text"]
                st.rerun()
    if yesterday_qs:
        st.markdown("**Yesterday**")
        for q in reversed(yesterday_qs[-5:]):
            if st.button(q["text"][:50] + ("..." if len(q["text"]) > 50 else ""), key=f"yesterday_{q['timestamp']}", use_container_width=True):
                st.session_state.rerun_question = q["text"]
                st.rerun()

# ------------------------------
# Config / Clients
# ------------------------------
CSV_EXPORT_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTH33TC1xTixH8TWGAOUUe3o-UIFX82HMaBv8BlI4KA5UnJxYs50QBitDUwXB_Jkl8M52CdE66s_XDx/pub?output=csv"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = "llama-3.1-8b-instant"
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        client = None

HF_EMOTION_MODEL = os.getenv("HF_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
EMOTION_MODEL_MAX = int(os.getenv("EMOTION_MODEL_MAX", "200"))

# ------------------------------
# Helpers
# ------------------------------
def safe_text(s):
    if s is None:
        return ""
    try:
        if isinstance(s, float) and math.isnan(s):  # pandas NaN
            return ""
    except Exception:
        pass
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    return s.strip()

def safe_int(v):
    try:
        return int(float(v or 0))
    except Exception:
        return 0

def fmt_k(n):
    try:
        n = int(n)
    except Exception:
        return "0"
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}M".rstrip("0").rstrip(".")
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}k".rstrip("0").rstrip(".")
    return str(n)

def nice_date(iso):
    s = safe_text(iso)
    if not s:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return s.split("T")[0] if "T" in s else s

def extract_hashtags(text):
    text = safe_text(text)
    return re.findall(r"#\w+", text.lower())

def top_n_emojis(texts, n=3):
    emoji_pattern = re.compile("[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000027BF]+", flags=re.UNICODE)
    counter = Counter()
    for t in texts:
        t = safe_text(t)
        for em in emoji_pattern.findall(t):
            counter[em] += 1
    return counter.most_common(n)

# ------------------------------
# Fetch CSV
# ------------------------------
def fetch_sheet_as_records(csv_url, timeout=30):
    try:
        resp = requests.get(csv_url, timeout=timeout)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
        df = pd.read_csv(buf, dtype=str)
        return df.to_dict(orient="records"), "google_sheet"
    except Exception as e:
        st.error(f"Failed to fetch/parse CSV: {e}")
        return None, None

# ------------------------------
# Normalize row
# ------------------------------
def normalize_item(item):
    if not isinstance(item, dict):
        return {}
    def g(k):
        if k in item and pd.notna(item[k]):
            return item[k]
        return None
    return {
        "id": g("id") or "",
        "user": g("user") or "",
        "text": g("text") or "",
        "likes": safe_int(g("diggCount") or g("likeCount") or g("likes") or 0),
        "shares": safe_int(g("shareCount") or g("shares") or 0),
        "plays": safe_int(g("playCount") or g("plays") or g("views") or 0),
        "comments": safe_int(g("commentCount") or g("comments") or 0),
        "music": g("musicMeta.musicName") or "",
        "music_author": g("musicMeta.musicAuthor") or "",
        "created": g("createTimeISO") or g("created_at") or g("post_date") or ""
    }

# ------------------------------
# Emotion model (cached small)
# ------------------------------
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
    except Exception:
        return None

def analyze_text_heuristic(text):
    text = safe_text(text)
    if not text:
        return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.0}
    if re.search(r"\b(win|love|happy|joy|great|awesome|yay|yes|delight|best|cheer)\b", text, re.I):
        return {"emotion": "happy", "sentiment": "positive", "confidence": 0.65}
    if re.search(r"\b(annoy|panic|stress|forgot|hate|angry|sad|oops|frustrat|upset)\b", text, re.I):
        return {"emotion": "angry", "sentiment": "negative", "confidence": 0.65}
    if re.search(r"\b(memory|remember|nostalgia|nostalgic|back then|old times)\b", text, re.I):
        return {"emotion": "nostalgic", "sentiment": "neutral", "confidence": 0.7}
    if re.search(r"\b(give|donate|gift|generous|share|help)\b", text, re.I):
        return {"emotion": "generous", "sentiment": "positive", "confidence": 0.7}
    if re.search(r"\b(fun|lol|haha|hilarious|silly)\b", text, re.I):
        return {"emotion": "fun", "sentiment": "positive", "confidence": 0.6}
    return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.5}

def analyze_text_with_model(text, model_pipeline=None, min_confidence=0.55):
    text = safe_text(text)
    if not text:
        return {"emotion": "unclear", "sentiment": "unclear", "confidence": 0.0}
    if model_pipeline is None:
        model_pipeline = get_emotion_pipeline()
    if model_pipeline:
        try:
            preds = model_pipeline(text[:1000])
            if isinstance(preds, list) and preds:
                scores = preds[0]
                top = max(scores, key=lambda x: x["score"])
                label = top["label"].lower()
                conf = float(top["score"])
                map_em = {
                    "joy": "happy",
                    "happiness": "happy",
                    "love": "generous",
                    "sadness": "sad",
                    "anger": "angry",
                    "fear": "fear",
                    "surprise": "surprise",
                    "neutral": "neutral"
                }
                emotion = map_em.get(label, label)
                if conf < min_confidence:
                    return {"emotion": "unclear", "sentiment": "unclear", "confidence": conf}
                sentiment = "positive" if emotion in ("happy", "generous", "surprise") else ("negative" if emotion in ("sad", "angry", "fear") else "neutral")
                return {"emotion": emotion, "sentiment": sentiment, "confidence": conf}
        except Exception:
            pass
    return analyze_text_heuristic(text)

# ------------------------------
# Process records
# ------------------------------
@st.cache_data(show_spinner=False)
def process_records(records, use_model=True, max_items=500):
    hashtag_counter = Counter()
    keyword_counter = Counter()
    sentiment_counts = Counter()
    emotional_barometer = Counter()
    posts = []
    model_pipeline = get_emotion_pipeline() if use_model else None

    for i, raw in enumerate((records or [])[:max_items]):
        r = normalize_item(raw if isinstance(raw, dict) else {})
        txt = safe_text(r.get("text"))
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
            "id": r.get("id", ""),
            "user": r.get("user", ""),
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

# ------------------------------
# Load data and process
# ------------------------------
records, source_label = fetch_sheet_as_records(CSV_EXPORT_URL)
if not records:
    st.stop()

hashtag_counter, sentiment_counts, emotional_barometer, top_keywords, top_posts_data = process_records(records, use_model=True, max_items=500)

# derive summaries
total_posts = sum(sentiment_counts.values()) or 1
sentiment_pct = {k: round(v / total_posts * 100, 1) for k, v in sentiment_counts.items()}
texts = [p.get("text", "") for p in top_posts_data]
top_emojis = top_n_emojis(texts, n=3)

for p in top_posts_data:
    p["eng"] = int(p.get("likes", 0)) + int(p.get("shares", 0)) + int(p.get("comments", 0))
top_posts_sorted = sorted(top_posts_data, key=lambda x: x.get("eng", 0), reverse=True)
top_3_posts = top_posts_sorted[:3]

# ------------------------------
# MAIN CHAT LAYOUT (integrated)
# ------------------------------

# Share button top-right
col_title, col_share = st.columns([6, 1])
with col_title:
    st.title("")
with col_share:
    current_url = "https://dentsusolutions.com/"
    if st.button("🔗 Share", use_container_width=True):
        st.code(current_url, language=None)
        st.success("Link ready to share!")

# Initialize chat history/system prompt if needed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are Dentsu Conversational Analytics assistant. Answer concisely with NZ spelling and reference data when available."}]

# Display conversation
for msg in st.session_state.chat_history:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    if role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)
    else:
        with st.chat_message("user"):
            st.markdown(content)

# restore preset if requested
preset_input = None
if "rerun_question" in st.session_state:
    preset_input = st.session_state.rerun_question
    del st.session_state.rerun_question

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# Quick Questions UI
if not st.session_state.chat_started:
    st.markdown("### 💡 Quick Questions")
    preset_questions = [
        "📊 Generate an emotionally resonant creative line related to Christmas.",
        "🎯 Give me a detailed summary of what people are experiencing this Christmas.",
        "📉 Recap what the pain points are for everyone this Christmas."
    ]
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    for question in preset_questions:
        if st.button(question, use_container_width=True, key=f"preset_{question}"):
            preset_input = question
            st.session_state.chat_started = True
    st.markdown('</div>', unsafe_allow_html=True)
else:
    with st.sidebar:
        st.divider()
        st.subheader("💡 Quick Questions")
        for question in [
        "📊 Generate an emotionally resonant creative line related to Christmas.",
        "🎯 Give me a detailed summary of what people are experiencing this Christmas.",
        "📉 Recap what the pain points are for everyone this Christmas."
        ]:
            if st.button(question, use_container_width=True, key=f"sidebar_preset_{question}"):
                preset_input = question

# Chat input
user_input = st.chat_input("Select a prompt above or type your custom prompt here")
if preset_input:
    user_input = preset_input

if user_input:
    # record in history
    if "question_history" not in st.session_state:
        st.session_state.question_history = []
    st.session_state.question_history.append({
        "text": user_input,
        "date": datetime.now().date(),
        "timestamp": datetime.now().isoformat()
    })
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analysing social data..."):
            try:
                if not client:
                    st.error("Missing Groq client. Set GROQ_API_KEY in environment.")
                else:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=st.session_state.chat_history
                    )
                    output = response.choices[0].message.content
                    # simple cleaning
                    def clean_output(text):
                        text = re.sub(r'

\[Insert Chart \d+:.*?\]

', '', text, flags=re.DOTALL)
                        text = re.sub(r'<Chart:.*?>', '', text, flags=re.DOTALL)
                        text = re.sub(r'(\d)([a-z])', r'\1 \2', text)
                        return "\n".join([ln for ln in text.split("\n") if not ln.strip().startswith("<Chart") and not ln.strip().startswith("[Insert Chart")]).strip()
                    cleaned_output = clean_output(output)
                    st.markdown(cleaned_output)
                    # chart
                    try:
                        if 'generate_dynamic_chart' in globals() and 'df' in globals():
                            chart = generate_dynamic_chart(user_input, df)
                            if chart is not None:
                                st.altair_chart(chart, use_container_width=True)
                    except Exception:
                        st.info("Chart generation failed (non-fatal).")
                    st.session_state.chat_history.append({"role": "assistant", "content": cleaned_output})
            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    st.warning("⚠️ Too many requests. Please wait a moment and try again.")
                else:
                    st.error("An error occurred while contacting the model. Check logs for details.")

# ------------------------------
# Emotion Summary & Trend Spotter UI
# ------------------------------
st.title("🎄 NZ Christmas Retail Trendspotter")
st.caption(f"Source: {source_label} — emotion model max {EMOTION_MODEL_MAX} rows")
st.sidebar.title("📅 Date Range")
_ = st.sidebar.radio("View trends from:", ["Last 24 hours", "Last 7 days"])

# Emotion summary and trend spotter side-by-side
colL, colR = st.columns([1.2, 1])
with colL:
    st.subheader("🔥 Emotion Summary")
    split_text = " • ".join([f"{k.capitalize()}: {v}% ({sentiment_counts[k]} posts)" for k, v in sentiment_pct.items()])
    st.markdown(f"**Emotion split (by post count):**  {split_text}")
    if top_emojis:
        emo_lines = "  ".join([f"{e} — {c}×" for e, c in top_emojis])
        st.markdown(f"**Top emojis:** {emo_lines}")
    else:
        st.markdown("**Top emojis:** none detected")
    st.markdown("**Emotional barometer (counts):**")
    for e, c in emotional_barometer.items():
        st.markdown(f"- {e.capitalize()}: {c}")

with colR:
    st.subheader("🧭 Trend Spotter — Top Posts")
    st.markdown("Highlights: actual top posts by engagement and key themes.")
    if top_3_posts:
        for i, p in enumerate(top_3_posts, start=1):
            txt_snip = (p.get("text") or "")[:140].replace("\n", " ").strip()
            st.markdown(f"**Top {i}** — _{fmt_k(p.get('eng',0))} engagements_")
            st.markdown(f"> {txt_snip}...")
            music = safe_text(p.get("music"))
            date = nice_date(p.get("created",""))
            meta = []
            if music:
                meta.append(f"Song: {music}")
            if date:
                meta.append(f"Date: {date}")
            if meta:
                st.markdown(" • ".join(meta))
    else:
        st.markdown("No top posts found")

    st.markdown("**Top themes (from post text):**")
    def top_themes_from_posts(posts, top_n=6):
        tokens = []
        for p in posts:
            text = safe_text(p.get("text",""))
            tokens += re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        c = Counter(tokens)
        return [t for t, _ in c.most_common(top_n)]
    top_themes = top_themes_from_posts(top_posts_data, top_n=6)
    if top_themes:
        st.markdown(", ".join(top_themes))
    else:
        st.markdown("None detected")

# Top posts table (no author column)
st.subheader("🎄 Top Posts Table")
if top_posts_sorted:
    def format_row_display(r):
        return {
            "text": safe_text(r.get("text","")),
            "likes": fmt_k(r.get("likes",0)),
            "shares": fmt_k(r.get("shares",0)),
            "plays": fmt_k(r.get("plays",0)),
            "comments": fmt_k(r.get("comments",0)),
            "engagements": fmt_k(r.get("eng",0)),
            "music": safe_text(r.get("music","")),
            "created": nice_date(r.get("created",""))
        }
    display_rows = [format_row_display(r) for r in top_posts_sorted]
    display_df = pd.DataFrame(display_rows)
    st.dataframe(display_df.fillna(""), use_container_width=True)
else:
    st.info("No posts to display")

# Generate Creatives (safe)
st.subheader("✨ Generate Creatives")
st.markdown("Generate short social lines based on the Trend Spotter themes and emotion summary above.")

def improved_prompt(top_hashtags, emotion_summary, top_songs, top_themes):
    return (
        "You are a creative assistant writing witty, sharp, relatable, short social lines for New Zealanders for the Christmas season.\n\n"
        f"Overall emotion split: {emotion_summary}\n"
        f"Dominant themes: {', '.join(top_themes[:6])}\n"
        f"Frequent songs: {', '.join(top_songs[:5]) or 'none'}\n"
        f"Top hashtags: {', '.join(top_hashtags[:10])}\n\n"
        "Produce 3 short, cheeky, emotionally honest, Kiwi-flavoured social lines suitable for organic posts or paid social. Keep each line under 100 characters."
    )

def generate_creatives_groq(top_hashtags, emotion_summary, top_songs, top_themes):
    if not client:
        return "Groq API key not configured. Set GROQ_API_KEY env var to enable external generation."
    prompt = improved_prompt(top_hashtags, emotion_summary, top_songs, top_themes)
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

_safe_hashtags = list(hashtag_counter.keys()) if isinstance(hashtag_counter, dict) else []
_safe_top_songs = []  # songs extraction not stored separately here
_safe_top_themes = top_themes if isinstance(top_themes, list) else []
_safe_emotion_summary = split_text if 'split_text' in locals() else "No emotion summary available"

if "creative_lines" not in st.session_state:
    st.session_state.creative_lines = ""

if st.button("Generate Creatives"):
    if not client:
        st.session_state.creative_lines = "Groq API key not configured. Set GROQ_API_KEY env var to enable external generation."
    else:
        st.session_state.creative_lines = generate_creatives_groq(
            top_hashtags=_safe_hashtags,
            emotion_summary=_safe_emotion_summary,
            top_songs=_safe_top_songs,
            top_themes=_safe_top_themes
        )

if st.session_state.creative_lines:
    st.markdown("#### ✨ Generated Lines")
    for line in str(st.session_state.creative_lines).split("\n"):
        if line.strip():
            st.markdown(f"✅ {line.strip()}")

# Hashtag wordcloud & top hashtags ordered
st.subheader("🌈 Hashtag Cloud & Top Hashtags")
col1, col2 = st.columns([1, 0.8])
with col1:
    small_freq = dict(Counter(hashtag_counter).most_common(40) if isinstance(hashtag_counter, dict) else {"empty":1})
    wc = WordCloud(width=400, height=160, max_font_size=40, background_color="white").generate_from_frequencies(small_freq)
    fig, ax = plt.subplots(figsize=(4,1.6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
with col2:
    st.markdown("**Top hashtags (ordered)**")
    for tag, cnt in Counter(hashtag_counter).most_common(30) if isinstance(hashtag_counter, dict) else []:
        st.markdown(f"- #{tag} — {cnt} mentions")

# Explore by hashtag
st.markdown("### 🔍 Explore Posts by Hashtag")
selected_tag = st.selectbox("Select a hashtag", options=[""] + list(dict(hashtag_counter).keys()) if isinstance(hashtag_counter, dict) else [""])
if selected_tag:
    filtered = [p for p in top_posts_data if f"#{selected_tag}" in (p.get("text","") or "").lower()]
    st.markdown(f"Showing {len(filtered)} posts with #{selected_tag}")
    for i, post in enumerate(filtered):
        with st.expander(f"Post {i+1} — {post.get('sentiment')}, {post.get('emotion')}"):
            cols = st.columns([1, 5])
            with cols[0]:
                if post.get("avatar"):
                    try:
                        st.image(post.get("avatar"), width=72)
                    except Exception:
                        pass
            with cols[1]:
                st.markdown(f"**Text:** {post.get('text')}")
                st.markdown(f"**Engagement:** 👍 {fmt_k(post.get('likes',0))} | 🔁 {fmt_k(post.get('shares',0))} | 💬 {fmt_k(post.get('comments',0))}")
                if post.get("music"):
                    st.markdown(f"**Music:** {post.get('music')} — {post.get('music_author','')}")

# Footer
st.markdown("---")
st.markdown("Powered by Dentsu")
