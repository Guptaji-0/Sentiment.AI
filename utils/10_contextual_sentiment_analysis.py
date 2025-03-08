'''import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Initialize Hugging Face sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_sentiment_model()

st.title("📊 Context-Aware Sentiment Analysis")

# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Context-Aware Sentiment Analysis?**", expanded=True):
    st.markdown("""
    Traditional sentiment analysis often **fails** to understand **context, sarcasm, and complex meanings**.  
    This method uses **AI-powered transformers (BERT-based models)** for a **more accurate** analysis.
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Performing Context-Aware Sentiment Analysis on Column:** `{selected_column}`")

    # **Contextual Sentiment Analysis Function**
    def get_contextual_sentiment(text):
        if isinstance(text, str):  # Ensure text is a string
            result = sentiment_pipeline(text)
            return result[0]["label"]
        return "Neutral"

    # Compute Context-Aware Sentiment if not already stored
    if "contextual_sentiment_results" not in st.session_state or "Contextual Sentiment" not in st.session_state.contextual_sentiment_results.columns:
        st.session_state.contextual_sentiment_results = df.copy()
        st.session_state.contextual_sentiment_results["Contextual Sentiment"] = st.session_state.contextual_sentiment_results[selected_column].apply(get_contextual_sentiment)

    df_result = st.session_state.contextual_sentiment_results  # Use stored results

    # 📊 **Contextual Sentiment Distribution**
    st.subheader("📊 Sentiment Distribution (AI-Powered)")

    if "Contextual Sentiment" in df_result.columns:  # Ensure column exists before plotting
        sentiment_counts = df_result["Contextual Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Contextual Sentiment Distribution", hole=0.3, color="Sentiment")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("🚨 Error: Sentiment analysis was not computed correctly.")

    # 📋 **Sample Data with Contextual Sentiments**
    st.subheader("📋 Sample Data with Contextual Sentiments")
    if "Contextual Sentiment" in df_result.columns:
        st.write(df_result[[selected_column, "Contextual Sentiment"]].head())
    else:
        st.error("🚨 Error: Sentiment analysis was not applied correctly.")

    # 📥 **Download Option**
    st.subheader("📥 Download Contextual Sentiment Data")
    csv = df_result[[selected_column, "Contextual Sentiment"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "contextual_sentiment_analysis.csv", "text/csv", key="download-csv")'''
