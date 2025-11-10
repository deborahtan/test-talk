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
        border: 1px solid #FAFAFA
