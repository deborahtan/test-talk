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
import plotly.express as px

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
    @font-face {
        font-family: 'StabilGrotesk';
        src: url('app/static/StabilGrotesk-Regular.otf') format('opentype');
        font-weight: 400;
        font-style: normal;
    }
    @font-face {
        font-family: 'StabilGrotesk';
        src: url('app/static/StabilGrotesk-Bold.otf') format('opentype');
        font-weight: 700;
        font-style: normal;
    }
    
    html, body, [class*="css"] {
        font-family: 'StabilGrotesk', sans-serif !important;
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

    /* Multi-line button support for sidebar */
    .stSidebar button {
        white-space: normal !important;
        word-wrap: break-word !important;
        height: auto !important;
        min-height: 2.5rem !important;
        padding: 0.5rem 1rem !important;
        text-align: left !important;
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
    **About the Tool**
    
    🎯 Get instant NZ Christmas insights from real social posts
    
    📊 Data-driven creative lines based on actual sentiment & trends  
    
    🎄 Understand what Kiwis are really feeling this festive season
    
    💡 Make confident campaign decisions backed by social data
    """
    )
    
    st.divider()
    
    st.markdown(
        """
    **How to Use**
    
    1️⃣ **Key Mentions** - View total posts and Christmas engagement at a glance
    
    2️⃣ **Christmas Spirit Summary** - AI-generated narrative of the festive mood
    
    3️⃣ **Emotion Summary** - Expand to see sentiment breakdown and top emojis
    
    4️⃣ **Top Posts** - Expand to browse highest-engagement content with links
    
    5️⃣ **Hashtag Cloud** - Expand to filter and explore trending hashtags
    
    6️⃣ **Explore by Hashtag** - Expand to dive deep into specific hashtag posts
    
    7️⃣ **Quick Questions** - Click preset prompts for instant insights
    
    8️⃣ **Chat** - Ask anything naturally - context is remembered across questions
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
            if st.button(q["text"], key=f"today_{q['timestamp']}", use_container_width=True):
                st.session_state.rerun_question = q["text"]
                st.rerun()
    if yesterday_qs:
        st.markdown("**Yesterday**")
        for q in reversed(yesterday_qs[-5:]):
            if st.button(q["text"], key=f"yesterday_{q['timestamp']}", use_container_width=True):
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

# Calculate Christmas mentions
mentions_total = sum(len(re.findall(r"\bchristmas\b", safe_text(t), flags=re.I)) for t in texts)

# Get top songs from posts
music_counter = Counter()
for p in top_posts_data:
    if p.get("music"):
        music_counter[p.get("music")] += 1
top_songs = [song for song, _ in music_counter.most_common(5)]

# ------------------------------
# UPDATED System Prompt with Context
# ------------------------------
context_summary = f"""
CURRENT DATA CONTEXT (refer to this when creating content):
- Total posts analyzed: {total_posts}
- Christmas mentions: {mentions_total}
- Sentiment breakdown: {sentiment_pct.get('positive', 0):.0f}% positive, {sentiment_pct.get('negative', 0):.0f}% negative, {sentiment_pct.get('neutral', 0):.0f}% neutral
- Top emojis: {', '.join([e for e, _ in top_emojis[:3]])}
- Top songs trending: {', '.join(top_songs[:3]) if top_songs else 'None'}
- Key themes: {', '.join(top_themes[:5])}
- Dominant emotion: {max(emotional_barometer.items(), key=lambda x: x[1])[0] if emotional_barometer else 'unclear'}
"""

SYSTEM_PROMPT = (
    "You are a NZ Christmas 2025 social media copywriter. You have a funny, relatable tone for every kiwi. You MUST ONLY discuss Christmas 2025 in New Zealand using the data below.\n\n"
    f"{context_summary}"
    "\n\n"
    "🚨 ABSOLUTE RULES - NO EXCEPTIONS:\n\n"
    "1. If asked for creative lines, generate ONLY Christmas-themed lines based on the social data trends above:\n"
    "   - Shopping queues/crowds ('Queue goals at The Warehouse')\n"
    "   - Mariah Carey/Christmas music (if in data)\n"
    "   - Baking/pavlova stress\n"
    "   - Decorating the tree\n"
    "   - Secret Santa/gift giving\n"
    "   - Events like christmas parties\n"
    "   - Christmas traffic/parking\n\n"
    "2. NEVER generate lines about:\n"
    "   - 'Elevate your vibe'\n"
    "   - 'Unlock your potential'\n"
    "   - 'Technology meets humanity'\n"
    "   - Wellness/self-care (unless Christmas shopping stress)\n"
    "   - Generic business/tech content\n"
    "   - Mental health (unless Christmas-specific stress)\n\n"
    "3. FORMAT for creative lines (CRITICAL):\n"
    "   Present each line as:\n"
    "   1. 'Clean creative line here 🎄'\n"
    f"   Inspired by: [specific data point - e.g., 'Mariah Carey trending in posts', '{sentiment_pct.get('negative', 0):.0f}% posts showing queue stress', 'Christmas tree emoji appearing in {mentions_total} posts']\n\n"
    "   DO NOT put data references IN the creative line itself. Keep the line clean and usable.\n\n"
    "4. Each creative line must:\n"
    "   - Be Christmas-specific and relatable to Kiwi Christmas experiences\n"
    "   - Under 80 characters (the line only, not including inspiration)\n"
    "   - Use emojis where appropriate\n"
    "   - Be ready to use in actual marketing/social content\n"
    "   - Vary tone: cheeky, relatable, nostalgic, sassy, reassuring\n\n"
    "5. EXAMPLE CORRECT FORMAT:\n"
    "   1. 'Sleigh all day in the Sylvia Park queues 🎄'\n"
    f"   Inspired by: Shopping queue stress appearing in {sentiment_pct.get('negative', 0):.0f}% of posts\n\n"
    "   2. 'Mariah's on repeat and we're not mad about it 🎅'\n"
    "   Inspired by: All I Want for Christmas Is You trending across posts\n\n"
    "   3. 'Pavlova: nailed it or failed it? 😅'\n"
    f"   Inspired by: Baking themes and {sentiment_pct.get('negative', 0):.0f}% posts showing festive pressure\n\n"
    "   4. 'Deck the tree like a pro or stress out trying 🎄'\n"
    "   Inspired by: Christmas tree emojis and decorating posts trending\n\n"
    "   5. 'Secret Santa sorted. Sanity? TBD. 😂'\n"
    "   Inspired by: Gift-giving themes across posts\n\n"
    "6. EXAMPLE WRONG FORMAT (NEVER DO THIS):\n"
    "   ❌ 'Sleigh all day (inspired by 30% negative sentiment)' - NO data in the line!\n"
    "   ❌ 'Queue goals - data shows stress' - NO data in the line!\n"
    "   ❌ 'Elevate your vibe' - NOT CHRISTMAS\n"
    "   Keep the creative line CLEAN. Put inspiration explanation on separate line.\n\n"
    "7. If asked about pain points, discuss ONLY:\n"
    "   - Christmas shopping stress (cite data)\n"
    "   - Gift budget pressure\n"
    "   - Queue fatigue\n"
    "   - Decorating stress\n"
    "   - Family gathering logistics\n"
    "   - Boxing Day sales planning\n\n"
    "8. If asked about anything non-Christmas, respond:\n"
    "   'I only analyse NZ Christmas 2025 trends. Ask me about Christmas shopping, decorating, or festive vibes!'\n\n"
    "Use NZ English. Be cheeky, sassy, but relatable. Creative lines should be campaign-ready with inspiration shown separately."
)

# Initialize or update chat history with current system prompt
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
else:
    # Always update the system prompt to ensure latest version is used
    if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[0].get("role") == "system":
        st.session_state.chat_history[0]["content"] = SYSTEM_PROMPT
    else:
        st.session_state.chat_history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

# Title + source
st.markdown("## 🎄 NZ Christmas Retail Trendspotter")
st.markdown("**Source:** TikTok, Instagram, Meta")

# ------------------------------
# Key Christmas Mentions Heading + Scorecards
# ------------------------------
st.markdown("### 🎅 Key Christmas Mentions")

top3_emoji_list = top_emojis

# Scorecards layout
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div style='text-align:center; padding:20px; border:1px solid #333; border-radius:8px;'>
            <div style='font-size:14px; color:#888; margin-bottom:8px;'>Total posts</div>
            <div style='font-size:36px; font-weight:700;'>{}</div>
        </div>
        """.format(total_posts),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div style='text-align:center; padding:20px; border:1px solid #333; border-radius:8px;'>
            <div style='font-size:14px; color:#888; margin-bottom:8px;'>Christmas mentions</div>
            <div style='font-size:36px; font-weight:700;'>{}</div>
        </div>
        """.format(mentions_total),
        unsafe_allow_html=True,
    )
with c3:
    emoji_count = sum([c for _, c in top3_emoji_list]) if top3_emoji_list else 0
    st.markdown(
        """
        <div style='text-align:center; padding:20px; border:1px solid #333; border-radius:8px;'>
            <div style='font-size:14px; color:#888; margin-bottom:8px;'>Christmas emoji mentions</div>
            <div style='font-size:36px; font-weight:700;'>{}</div>
        </div>
        """.format(emoji_count),
        unsafe_allow_html=True,
    )

# ------------------------------
# Emotion Summary (expandable) - UPDATED with top emojis
# ------------------------------
with st.expander("🔥 Emotion Summary (expand)"):
    # Create sentiment bar chart data
    sentiment_df = pd.DataFrame([
        {"Sentiment": k.capitalize(), "Count": v, "Percentage": round(v / total_posts * 100, 1)}
        for k, v in sentiment_counts.items()
    ])
    
    st.markdown("**Sentiment Breakdown:**")
    # Display sentiment bar chart
    fig_sentiment = px.bar(sentiment_df, x="Sentiment", y="Count", 
                 hover_data=["Percentage"],
                 labels={"Count": "Number of Posts", "Percentage": "Percentage (%)"},
                 color="Sentiment",
                 color_discrete_map={"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E", "Unclear": "#607D8B"})
    fig_sentiment.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Create emotional barometer bar chart
    emotion_df = pd.DataFrame([
        {"Emotion": k.capitalize(), "Count": v}
        for k, v in emotional_barometer.items()
    ])
    
    st.markdown("**Emotional Barometer:**")
    # Display emotional barometer bar chart
    fig_emotion = px.bar(emotion_df, x="Emotion", y="Count",
                 labels={"Count": "Number of Posts"},
                 color="Emotion",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig_emotion.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_emotion, use_container_width=True)
    
    # Top emojis - sleek display
    if top_emojis:
        st.markdown("**Top Emojis:**")
        emoji_cols = st.columns(len(top_emojis))
        for idx, (emoji, count) in enumerate(top_emojis):
            with emoji_cols[idx]:
                st.markdown(
                    f"""
                    <div style='text-align:center; padding:15px; border:1px solid #333; border-radius:8px;'>
                        <div style='font-size:32px; margin-bottom:8px;'>{emoji}</div>
                        <div style='font-size:18px; font-weight:700;'>{count}</div>
                        <div style='font-size:12px; color:#888;'>mentions</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ------------------------------
# Smart Christmas Spirit Summary - UPDATED to use LLM
# ------------------------------
st.subheader("🎄 Christmas Spirit Summary")

@st.cache_data(show_spinner=False)
def generate_spirit_summary_with_llm(posts_json, sentiment_dict, emotional_dict):
    # Convert JSON string back to list of dicts
    import json
    posts = json.loads(posts_json)
    
    # Filter Christmas-related posts
    christmas_posts = [p for p in posts if "christmas" in safe_text(p.get("text", "")).lower()]
    
    # Get top 30 posts by engagement for LLM analysis
    top_christmas_posts = sorted(christmas_posts, key=lambda x: x.get("eng", 0), reverse=True)[:30]
    
    # Prepare post texts for LLM
    post_samples = []
    for i, p in enumerate(top_christmas_posts[:30], 1):
        text = safe_text(p.get("text", ""))
        music = safe_text(p.get("music", ""))
        eng = p.get("eng", 0)
        post_samples.append(f"Post {i} ({fmt_k(eng)} engagements): {text[:200]}... [Music: {music}]")
    
    posts_text = "\n\n".join(post_samples)
    
    # Calculate sentiment stats - sentiment_dict and emotional_dict are plain dicts now
    total = sum(sentiment_dict.values()) or 1
    pos_pct = sentiment_dict.get("positive", 0) / total * 100
    neg_pct = sentiment_dict.get("negative", 0) / total * 100
    dominant_emotion = max(emotional_dict.items(), key=lambda x: x[1])[0] if emotional_dict else "unclear"
    
    # Create LLM prompt
    llm_prompt = f"""Write a 3-4 paragraph summary of what's trending in NZ Christmas 2025 social media.

Data shows: {pos_pct:.0f}% positive vibes, {neg_pct:.0f}% showing stress/chaos, dominant emotion: {dominant_emotion}

TOP POSTS:
{posts_text}

Structure:
- Paragraph 1: What's the overall Christmas vibe in NZ? Cite 1-2 specific posts with engagement numbers
- Paragraph 2: What are people excited about? (songs, traditions, activities) - cite specific examples
- Paragraph 3: What's stressing them out? (queues, budgets, logistics) - cite specific examples  
- Paragraph 4: Key themes appearing across posts (baking, decorating, music, shopping)

Tone: Cheeky Kiwi mate giving the lowdown - "here's what's happening", "Kiwis are vibing on"
DO NOT use the word "sentiment" - just talk about what people are posting about
Cite actual post quotes and engagement numbers
Use **bold** for emphasis"""

    try:
        if client:
            # Create a temporary chat for this summary
            summary_messages = [
                {"role": "system", "content": "You are a social media analyst specializing in sentiment and trend analysis for New Zealand audiences."},
                {"role": "user", "content": llm_prompt}
            ]
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=summary_messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            return f"**Overall vibe:** {pos_pct:.0f}% positive, {neg_pct:.0f}% negative, with **{dominant_emotion}** leading the charge.\n\nUnable to generate detailed summary - LLM client not available."
    except Exception as e:
        return f"**Overall vibe:** {pos_pct:.0f}% positive, {neg_pct:.0f}% negative, with **{dominant_emotion}** leading the charge.\n\nDetailed analysis unavailable. Error: {str(e)}"

# Convert data to proper formats for caching
import json
posts_json = json.dumps(top_posts_data)
# Explicitly convert Counter objects to regular dicts
sentiment_dict = {k: v for k, v in sentiment_counts.items()}
emotional_dict = {k: v for k, v in emotional_barometer.items()}

with st.spinner("Generating Christmas spirit summary..."):
    spirit_summary = generate_spirit_summary_with_llm(posts_json, sentiment_dict, emotional_dict)
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
# Wordcloud & Hashtags (expandable) - UPDATED with transparent background
# ------------------------------
with st.expander("🌈 Hashtag Cloud (expand)"):
    # Sort hashtags by count and create options with counts
    sorted_hashtags = sorted(dict(hashtag_counter).items(), key=lambda x: x[1], reverse=True) if isinstance(hashtag_counter, dict) else []
    hashtag_options = ["All hashtags"] + [f"#{tag} ({count} posts)" for tag, count in sorted_hashtags]
    
    sel_tag_display = st.selectbox("Filter cloud by hashtag (optional)", options=hashtag_options)
    
    # Extract just the tag name
    sel_tag = ""
    if sel_tag_display and sel_tag_display != "All hashtags":
        sel_tag = sel_tag_display.split(" (")[0].lstrip("#")
    
    if sel_tag:
        source_posts = [p for p in top_posts_data if f"#{sel_tag}" in (p.get("text","") or "").lower()]
        total_with_tag = len(source_posts)
        st.markdown(f"**Showing {total_with_tag} posts with #{sel_tag}**")
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

    # TRANSPARENT BACKGROUND
    wc = WordCloud(width=600, height=240, max_font_size=60, background_color=None, mode="RGBA").generate_from_frequencies(small_freq)
    fig, ax = plt.subplots(figsize=(6,2.4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_visible(False)
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)

# ------------------------------
# Explore Posts by Hashtag
# ------------------------------
with st.expander("🔍 Explore Posts by Hashtag (expand)"):
    # Sort hashtags by count for this dropdown too
    sorted_hashtags_explore = sorted(dict(hashtag_counter).items(), key=lambda x: x[1], reverse=True) if isinstance(hashtag_counter, dict) else []
    explore_options = [""] + [f"#{tag} ({count} posts)" for tag, count in sorted_hashtags_explore]
    
    selected_tag_display = st.selectbox("Select a hashtag to explore", options=explore_options, key="explore_hashtag")
    
    # Extract just the tag name
    selected_tag = ""
    if selected_tag_display:
        selected_tag = selected_tag_display.split(" (")[0].lstrip("#")
    
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
# Bottom section: Quick Questions + Chat Display & Input
# ------------------------------

st.divider()

st.markdown("### 💡 Quick Questions")
quick_qs = [
    "📊 Generate 5 creative line options based on today's trends.",
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
            # Format numbered lists as styled elements
            def format_response(text):
                lines = text.split('\n')
                formatted_lines = []
                in_numbered_list = False
                list_items = []
                
                for line in lines:
                    # Check if line starts with a number followed by period or parenthesis
                    if re.match(r'^\d+[\.)]\s+', line.strip()):
                        in_numbered_list = True
                        # Extract the number and content
                        match = re.match(r'^(\d+)[\.)]\s+(.+)$', line.strip())
                        if match:
                            num, content_text = match.groups()
                            list_items.append((num, content_text))
                    elif in_numbered_list and line.strip() == '':
                        # Empty line might indicate end of list
                        continue
                    elif in_numbered_list and not re.match(r'^\d+[\.)]\s+', line.strip()):
                        # List ended, render collected items
                        if list_items:
                            for num, content_text in list_items:
                                formatted_lines.append(
                                    f"""<div style='padding: 12px 16px; margin: 8px 0; border-left: 3px solid #4CAF50; background-color: #1E1E1E; border-radius: 4px;'>
                                    <span style='font-weight: 700; color: #4CAF50; font-size: 18px; margin-right: 12px;'>{num}.</span>
                                    <span style='font-size: 14px;'>{content_text}</span>
                                    </div>"""
                                )
                            list_items = []
                            in_numbered_list = False
                        formatted_lines.append(line)
                    else:
                        formatted_lines.append(line)
                
                # Handle case where list is at the end
                if list_items:
                    for num, content_text in list_items:
                        formatted_lines.append(
                            f"""<div style='padding: 12px 16px; margin: 8px 0; border-left: 3px solid #4CAF50; background-color: #1E1E1E; border-radius: 4px;'>
                            <span style='font-weight: 700; color: #4CAF50; font-size: 18px; margin-right: 12px;'>{num}.</span>
                            <span style='font-size: 14px;'>{content_text}</span>
                            </div>"""
                        )
                
                return '\n'.join(formatted_lines)
            
            # Check if response has numbered list
            if re.search(r'^\d+[\.)]\s+', content, re.MULTILINE):
                formatted_output = format_response(content)
                st.markdown(formatted_output, unsafe_allow_html=True)
            else:
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
                    
                    # Format numbered lists as individual styled elements
                    def format_response(text):
                        # Check if response contains numbered list (1. 2. 3. etc)
                        lines = text.split('\n')
                        formatted_parts = []
                        current_text = []
                        in_numbered_list = False
                        
                        i = 0
                        while i < len(lines):
                            line = lines[i]
                            stripped = line.strip()
                            
                            # Check if line starts with a number followed by period or parenthesis
                            if re.match(r'^\d+[\.)]\s+', stripped):
                                # If we have accumulated text before the list, add it
                                if current_text:
                                    formatted_parts.append('\n'.join(current_text))
                                    current_text = []
                                
                                in_numbered_list = True
                                # Extract the number and content
                                match = re.match(r'^(\d+)[\.)]\s+(.+)$', stripped)
                                if match:
                                    num, content = match.groups()
                                    
                                    # Check if next lines are continuation (don't start with number)
                                    full_content = content
                                    j = i + 1
                                    while j < len(lines):
                                        next_line = lines[j].strip()
                                        if next_line and not re.match(r'^\d+[\.)]\s+', next_line):
                                            full_content += ' ' + next_line
                                            j += 1
                                        else:
                                            break
                                    
                                    # Create styled block for this numbered item
                                    formatted_parts.append(
                                        f"""<div style='padding: 12px 16px; margin: 8px 0; border-left: 3px solid #4CAF50; background-color: #1E1E1E; border-radius: 4px;'>
                                        <span style='font-weight: 700; color: #4CAF50; font-size: 18px; margin-right: 12px;'>{num}.</span>
                                        <span style='font-size: 14px;'>{full_content}</span>
                                        </div>"""
                                    )
                                    i = j
                                    continue
                            elif in_numbered_list and not stripped:
                                # Empty line after list - end of list
                                in_numbered_list = False
                            elif not in_numbered_list:
                                # Regular text, accumulate it
                                current_text.append(line)
                            
                            i += 1
                        
                        # Add any remaining text
                        if current_text:
                            formatted_parts.append('\n'.join(current_text))
                        
                        return '\n'.join(formatted_parts)
                    
                    # Check if response has numbered list
                    if re.search(r'^\d+[\.)]\s+', cleaned_output, re.MULTILINE):
                        formatted_output = format_response(cleaned_output)
                        st.markdown(formatted_output, unsafe_allow_html=True)
                    else:
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
                    st.error(f"An error occurred while contacting the model: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by Dentsu")
