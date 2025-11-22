import streamlit as st
from transformers import pipeline
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "beauty_sentiment_model_final",
    "beauty_sentiment_model_final"
)

@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        return_all_scores=True
    )

st.set_page_config(page_title="Customer Sentiment Analyzer", layout="centered")
st.title("Beauty Product Reviews Sentiment Analyzer")

classifier = load_classifier()

review = st.text_area(
    "Enter a customer review:",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):
    if review.strip():
        with st.spinner("Analyzing..."):
            preds = classifier(review)[0]
            preds = sorted(preds, key=lambda x: x["score"], reverse=True)
            
            st.success(f"**Predicted Sentiment: {preds[0]['label'].upper()}**")
            
            cols = st.columns(3)
            for i, p in enumerate(preds):
                with cols[i]:
                    score = p["score"] * 100
                    st.metric(p["label"].capitalize(), f"{score:.1f}%")
            
            st.bar_chart({p["label"]: p["score"] for p in preds})
    else:
        st.warning("Please enter a review!")
        