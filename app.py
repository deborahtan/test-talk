# app.py
import os
import io
import math
import streamlit as st
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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
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

    .big-num { font-size: 40px; font-weight:700; margin-bottom:2px; text-align:center; }
    .small-label { font-size:12px; color:#666; text-align:center; margin-top:0; }
    .scorecard-block { padding:8px 4px; }
    .scorecard-emoji { font-size:14px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    st.markdown(
        """
    **How to use**
    - Synthesises top posts across TikTok, Instagram and Meta and analyses sentiment
    - Conversation context is remembered
    """
    )
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
        return df.to_dict(orient="records"), "TikTok, Instagram, Meta"
    except Exception as e:
        st.error(f"Failed to fetch/parse CSV: {e}")
        return None, None


# ------------------------------
# Normalize row (include URL)
# ------------------------------
def normalize_item(item):
    if not isinstance(item, dict):
        return {}

    def g(k):
        if k in item and pd.notna(item[k]):
            return item[k]
        return None

    url = g("webVideoUrl") or g("videoUrl") or g("url") or g("link") or ""

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
        "created": g("createTimeISO") or g("created_at") or g("post_date") or "",
        "url": safe_text(url),
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
                    "neutral": "neutral",
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
        posts.append(
            {
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
                "created": r.get("created", ""),
                "url": r.get("url", ""),
            }
        )
    top_keywords = [k for k, _ in keyword_counter.most_common(10)]
    return dict(hashtag_counter), dict(sentiment_counts), dict(emotional_barometer), top_keywords, posts


# ------------------------------
# Utility: clean_output
# ------------------------------
def clean_output(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[Insert Chart \d+:.*?\]", "", text, flags=re.DOTALL)
    text = re.sub(r"<Chart:.*?>", "", text, flags=re.DOTALL)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("<Chart") and not ln.strip().startswith("[Insert Chart")]
    return "\n".join(lines).strip()


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
# Main chat / assistant system prompt (kept in session)
# ------------------------------
SYSTEM_PROMPT = (
    "You are the Dentsu Conversational Analytics assistant. "
    "Primary task: craft short, witty, culturally relevant one-liners and captions for the 2025 New Zealand Christmas season aimed at being relatable. "
    "Tone: warm, Kiwi, reassuring, lightly cheeky; avoid clichés and hard sell. Always use NZ English spelling. "
    "Base creative lines on the provided social dataset; do NOT invent metrics. "
    "When asked for analysis, keep answers concise, give one clear executive takeaway, and provide a single recommended creative line as an example. "
    "When producing multiple lines, vary voice (friendly, playful, reassuring) and keep each under 100 characters."
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ------------------------------
# Scorecard (total posts, christmas mentions, top emojis)
# ------------------------------
mentions_total = sum(len(re.findall(r"\bchristmas\b", safe_text(t), flags=re.I)) for t in texts)
top3_emoji_list = top_emojis

# Title + source
st.markdown("## 🎄 NZ Christmas Retail Trendspotter")
st.markdown("**Source:** TikTok, Instagram, Meta")

# Scorecards layout
c1, c2, c3, _ = st.columns([1, 1, 2, 6])
with c1:
    st.markdown(
        "<div class='scorecard-block'><div class='big-num'>{}</div><div class='small-label'>Total posts</div></div>".format(total_posts),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        "<div class='scorecard-block'><div class='big-num'>{}</div><div class='small-label'>Christmas mentions</div></div>".format(mentions_total),
        unsafe_allow_html=True,
    )
with c3:
    if top3_emoji_list:
        emoji_line = " ".join([f"{e} {c}×" for e, c in top3_emoji_list])
        st.markdown(f"<div class='scorecard-block'><div class='scorecard-emoji'><strong>Top emojis:</strong> {emoji_line}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='scorecard-block'><div class='scorecard-emoji'><strong>Top emojis:</strong> none</div></div>", unsafe_allow_html=True)

# Top themes calculation
def top_themes_from_posts(posts, top_n=6):
    tokens = []
    for p in posts:
        text = safe_text(p.get("text",""))
        tokens += re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    c = Counter(tokens)
    return [t for t, _ in c.most_common(top_n)]

top_themes = top_themes_from_posts(top_posts_data, top_n=6)

# ------------------------------
# Emotion Summary (expandable)
# ------------------------------
ordered = sorted(sentiment_pct.items(), key=lambda x: (-x[1], x[0]))
split_parts = []
for k, v in ordered:
    split_parts.append(f"{k.capitalize()}: {v}% ({sentiment_counts[k]} posts)")
emotion_paragraph = " • ".join(split_parts)

with st.expander("🔥 Emotion Summary (expand)"):
    st.markdown(f"**Emotion split (by post count):** {emotion_paragraph}")
    if top_emojis:
        emoji_lines = " ".join([f"{e} — {c}×" for e, c in top_emojis])
        st.markdown(f"**Top emojis:** {emoji_lines}")
    else:
        st.markdown("**Top emojis:** none detected")
    st.markdown("**Emotional barometer (counts):**")
    for e, c in emotional_barometer.items():
        st.markdown(f"- {e.capitalize()}: {c}")

# ------------------------------
# Smart Christmas Spirit Summary
# ------------------------------
st.subheader("🎄 Christmas Spirit Summary")

def generate_spirit_summary(posts, sentiment_counts, emotional_barometer, top_themes):
    total = sum(sentiment_counts.values()) or 1
    pos_pct = sentiment_counts.get("positive", 0) / total * 100
    neg_pct = sentiment_counts.get("negative", 0) / total * 100
    dominant_emotion = max(emotional_barometer.items(), key=lambda x: x[1])[0] if emotional_barometer else "unclear"
    theme_words = ", ".join(top_themes[:4]) if top_themes else "holiday cheer"
    sentiment_line = f"The overall mood is {pos_pct:.0f}% positive, {neg_pct:.0f}% negative, with {dominant_emotion} being the dominant emotion."
    theme_line = f"Key themes include: {theme_words}. People are expressing a mix of excitement, nostalgia, and the typical holiday hustle."
    return f"{sentiment_line}\n\n{theme_line}"

spirit_summary = generate_spirit_summary(top_posts_data, sentiment_counts, emotional_barometer, top_themes)
st.markdown(spirit_summary)

# ------------------------------
# Top Posts Summary (expandable with links)
# ------------------------------
with st.expander("🎄 Top Posts Summary (expand)"):
    if top_posts_sorted:
        for r in top_posts_sorted[:50]:
            eng = fmt_k(r.get("eng",0))
            txt = (safe_text(r.get("text",""))[:180] + "...") if len(safe_text(r.get("text",""))) > 180 else safe_text(r.get("text",""))
            date = nice_date(r.get("created",""))
            post_url = safe_text(r.get("url",""))
            st.markdown(f"**{eng}** — {txt}")
            meta = []
            if r.get("music"):
                meta.append(f"Song: {safe_text(r.get('music'))}")
            if date:
                meta.append(f"Date: {date}")
            if post_url:
                meta.append(f"[Link to post]({post_url})")
            if meta:
                st.markdown(" • ".join(meta))
            st.markdown("---")
    else:
        st.info("No posts to display")

# ------------------------------
# Wordcloud & Hashtags (expandable)
# ------------------------------
with st.expander("🌈 Hashtag Cloud & Top Hashtags (expand)"):
    sel_tag = st.selectbox("Filter cloud by hashtag (optional)", options=[""] + list(dict(hashtag_counter).keys()) if isinstance(hashtag_counter, dict) else [""])
    if sel_tag:
        source_posts = [p for p in top_posts_data if f"#{sel_tag}" in (p.get("text","") or "").lower()]
    else:
        source_posts = top_posts_data

    small_freq = Counter()
    for p in source_posts:
        for tag in extract_hashtags(p.get("text","")):
            small_freq[tag.lstrip("#")] += 1
    if not small_freq:
        small_freq = dict(Counter(hashtag_counter).most_common(40) if isinstance(hashtag_counter, dict) else {"empty":1})
    else:
        small_freq = dict(small_freq.most_common(40))

    wc = WordCloud(width=600, height=240, max_font_size=60, background_color="#E6E6E6").generate_from_frequencies(small_freq)
    fig, ax = plt.subplots(figsize=(6,2.4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_visible(False)
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)

    st.markdown("**Top hashtags (ordered)**")
    for tag, cnt in list(small_freq.items())[:50]:
        st.markdown(f"- #{tag} — {cnt} mentions")

    st.markdown("**Sample posts for this selection**")
    sample_posts = source_posts[:20]
    for p in sample_posts:
        url = safe_text(p.get("url",""))
        if url:
            st.markdown(f"- [{fmt_k(p.get('eng',0))} engagements]({url}) — {safe_text(p.get('text',''))[:80]}...")
        else:
            st.markdown(f"- {fmt_k(p.get('eng',0))} engagements — {safe_text(p.get('text',''))[:80]}...")

# ------------------------------
# Explore Posts by Hashtag
# ------------------------------
with st.expander("🔍 Explore Posts by Hashtag (expand)"):
    selected_tag = st.selectbox("Select a hashtag to explore", options=[""] + list(dict(hashtag_counter).keys()) if isinstance(hashtag_counter, dict) else [""])
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
                    if post.get("url"):
                        st.markdown(f"[Link to post]({safe_text(post.get('url'))})")

# ------------------------------
# Bottom section: Quick Questions + Chat
# ------------------------------

st.divider()

st.markdown("### 💡 Quick Questions")
quick_qs = [
    "📊 Generate an emotionally resonant creative line related to Christmas.",
    "🎯 Give me a detailed summary of what people are experiencing this Christmas.",
    "📉 Recap what the pain points are for everyone this Christmas.",
]
for q in quick_qs:
    if st.button(q, use_container_width=True, key=f"quick_q_{q}"):
        st.session_state.rerun_question = q
        st.rerun()

st.divider()

st.markdown("### 💬 Chat")

# Display conversation (exclude system prompt)
for msg in st.session_state.chat_history:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    if role == "system":
        continue  # Skip system prompt
    if role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)
    else:
        with st.chat_message("user"):
            st.markdown(content)

# preserve preset rerun question logic
preset_input = None
if "rerun_question" in st.session_state:
    preset_input = st.session_state.rerun_question
    del st.session_state.rerun_question

# Chat input
user_input = st.chat_input("Type your question here")
if preset_input:
    user_input = preset_input

if user_input:
    # record in history
    if "question_history" not in st.session_state:
        st.session_state.question_history = []
    st.session_state.question_history.append(
        {"text": user_input, "date": datetime.now().date(), "timestamp": datetime.now().isoformat()}
    )
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analysing social data..."):
            try:
                if not client:
                    st.error("Missing Groq client. Set GROQ_API_KEY in environment.")
                else:
                    response = client.chat.completions.create(model=GROQ_MODEL, messages=st.session_state.chat_history)
                    output = response.choices[0].message.content
                    cleaned_output = clean_output(output)
                    st.markdown(cleaned_output)
                    try:
                        if "generate_dynamic_chart" in globals() and "df" in globals():
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

# Footer
st.markdown("---")
st.markdown("Powered by Dentsu")
