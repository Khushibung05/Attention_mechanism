import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI News Intelligence System",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CYBERPUNK STYLING
# =====================================================

st.markdown("""
<style>

.stApp{
background:
linear-gradient(
135deg,
#050816 0%,
#0b1026 25%,
#160020 60%,
#050505 100%);
color:white;
}

.main-title{
font-size:58px;
font-weight:900;
text-align:center;
background:linear-gradient(
90deg,
#00ffff,
#ff00ff,
#00ff99
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#cccccc;
margin-bottom:25px;
}

.glass{
background:rgba(255,255,255,0.08);
padding:25px;
border-radius:25px;
backdrop-filter:blur(14px);
border:1px solid rgba(255,255,255,0.15);
box-shadow:
0 0 20px rgba(0,255,255,.2),
0 0 40px rgba(255,0,255,.15);
}

.stat-card{
background:rgba(255,255,255,0.08);
padding:15px;
border-radius:20px;
text-align:center;
}

.big-result{
font-size:34px;
font-weight:800;
color:#00ffff;
text-align:center;
}

.confidence{
font-size:22px;
text-align:center;
color:#00ff99;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD FILES
# =====================================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model("model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_resources()

MAX_LEN = 500

# =====================================================
# TEXT PREPROCESSING
# =====================================================

def clean_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    text = re.sub(
        r'[^a-zA-Z\\s]',
        '',
        text
    )

    text = re.sub(
        r'\\s+',
        ' ',
        text
    ).strip()

    return text


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">AI NEWS INTELLIGENCE SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Self-Attention Powered News Classification Platform</div>',
    unsafe_allow_html=True
)

# =====================================================
# INPUT
# =====================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

st.subheader("📰 Enter News Article")

article = st.text_area(
    "",
    height=250,
    placeholder="Paste news article here..."
)

st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# =====================================================
# ANALYZE BUTTON
# =====================================================

if st.button("🚀 Analyze Article", use_container_width=True):

    if len(article.strip()) == 0:
        st.warning("Please enter an article.")
        st.stop()

    clean_article = clean_text(article)

    sequence = tokenizer.texts_to_sequences(
        [clean_article]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0]

    pred_index = np.argmax(prediction)

    category = label_encoder.inverse_transform(
        [pred_index]
    )[0]

    confidence = prediction[pred_index] * 100

    # =================================================
    # PREDICTED CATEGORY
    # =================================================

    st.write("")
    st.subheader("🎯 Predicted Category")

    st.markdown(
        f"""
        <div class="glass">
        <div class="big-result">{category.upper()}</div>
        <div class="confidence">
        Confidence: {confidence:.2f}%
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # =================================================
    # RADAR CHART
    # =================================================

    st.subheader("📊 Category Confidence Radar")

    classes = list(label_encoder.classes_)

    radar = go.Figure()

    radar.add_trace(
        go.Scatterpolar(
            r=list(prediction) + [prediction[0]],
            theta=classes + [classes[0]],
            fill="toself"
        )
    )

    radar.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        radar,
        use_container_width=True
    )

    # =================================================
    # IMPORTANT WORDS
    # =================================================

    st.subheader("🔥 Important Words")

    words = clean_article.split()

    stop_words = {
        "the","a","an","is","are","was","were",
        "to","of","and","for","on","in","with",
        "at","by","from","that","this","it"
    }

    filtered = [
        w for w in words
        if len(w) > 3 and w not in stop_words
    ]

    counts = Counter(filtered)

    top_words = counts.most_common(10)

    if len(top_words) > 0:

        max_freq = top_words[0][1]

        for word, freq in top_words:

            score = (freq/max_freq)*100

            st.write(f"**{word}**")

            st.progress(score/100)

    # =================================================
    # ATTENTION HEATMAP
    # =================================================

    st.subheader("🧠 Word Relationship Heatmap")

    heat_words = list(dict(top_words).keys())[:10]

    if len(heat_words) >= 2:

        size = len(heat_words)

        matrix = np.zeros((size, size))

        for i in range(size):
            for j in range(size):

                if i == j:
                    matrix[i][j] = 1

                else:
                    matrix[i][j] = (
                        counts[heat_words[i]]
                        * counts[heat_words[j]]
                    )

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        sns.heatmap(
            matrix,
            annot=True,
            cmap="coolwarm",
            xticklabels=heat_words,
            yticklabels=heat_words,
            ax=ax
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

    # =================================================
    # ARTICLE STATS
    # =================================================

    st.subheader("📈 Article Statistics")

    word_count = len(words)

    char_count = len(article)

    reading_time = round(word_count / 200, 2)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Word Count",
            word_count
        )

    with c2:
        st.metric(
            "Character Count",
            char_count
        )

    with c3:
        st.metric(
            "Reading Time (mins)",
            reading_time
        )

    # =================================================
    # PROBABILITY TABLE
    # =================================================

    st.subheader("📋 Detailed Probabilities")

    probability_df = pd.DataFrame({
        "Category": classes,
        "Probability (%)":
        np.round(prediction * 100, 2)
    })

    probability_df = probability_df.sort_values(
        "Probability (%)",
        ascending=False
    )

    st.dataframe(
        probability_df,
        use_container_width=True
    )