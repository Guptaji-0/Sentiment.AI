import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Sentiment Intensity Analyzer
analyzer = SentimentIntensityAnalyzer()

st.markdown("<h1 style='text-align: center;'> 📊 Sentiment Intensity Analysis </h1>", unsafe_allow_html=True)


# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Sentiment Intensity Analysis?**", expanded=True):
    st.markdown("""
    Sentiment Intensity Analysis provides a **detailed sentiment score** for text.

    **What does it do?**  
    - Assigns a **compound sentiment score** between **-1 (Very Negative) and +1 (Very Positive)**.  
    - Identifies **how strong** a sentiment is.  

    **Expected Output:**  
    - A **sentiment intensity chart**.  
    - A table showing **detailed sentiment scores** for each text entry.  

    **Example:**  
    - *"I absolutely love this product! It's amazing!"* → **Compound Score: 0.8 (Very Positive)**  
    - *"The service was terrible and slow."* → **Compound Score: -0.6 (Negative)**  
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Analyzing Sentiment Intensity for Column:** `{selected_column}`")

    # **Sentiment Intensity Analysis Function**
    def get_sentiment_intensity(text):
        if isinstance(text, str):  # Ensure text is a string
            scores = analyzer.polarity_scores(text)
            return scores["compound"]  # Compound score represents overall sentiment intensity
        return 0  # Default neutral score for missing values

    # Compute Sentiment Intensity if not already stored
    if "sentiment_intensity_results" not in st.session_state:
        st.session_state.sentiment_intensity_results = df.copy()
        st.session_state.sentiment_intensity_results["Sentiment Intensity"] = st.session_state.sentiment_intensity_results[selected_column].apply(get_sentiment_intensity)

    df_result = st.session_state.sentiment_intensity_results  # Use stored results

    # 📊 **Sentiment Intensity Distribution**
    st.subheader("📊 Sentiment Intensity Distribution")

    if "Sentiment Intensity" in df_result.columns:  # Ensure Sentiment column exists before plotting
        fig = px.histogram(df_result, x="Sentiment Intensity", nbins=30, title="Sentiment Intensity Distribution", 
                           color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("🚨 Error: Sentiment intensity analysis was not computed correctly.")

    # 📈 **Graph Interpretation**
    with st.expander("📈 **How to Interpret This Graph?**"):
        st.markdown("""
        - **More positive scores (closer to +1)** → **Highly positive feedback**  
        - **More negative scores (closer to -1)** → **Highly negative feedback**  
        - **Scores near 0** → Neutral feedback  
        """)

    # 🏆 **Business Insights**
    with st.expander("🏆 **Business Insights from This Graph?**"):
        st.markdown("""
        - **High number of very positive scores** → Customers are extremely satisfied.  
        - **High number of very negative scores** → Strong dissatisfaction; take action.  
        - **Balanced distribution** → Mixed customer opinions, needs further analysis.  
        """)

    # 📋 **Sample Data with Sentiment Intensity Scores**
    st.subheader("📋 Sample Data with Sentiment Intensity Scores")
    st.write(df_result[[selected_column, "Sentiment Intensity"]].head())

    # 📥 **Download Option**
    st.subheader("📥 Download Sentiment Intensity Data")
    csv = df_result[[selected_column, "Sentiment Intensity"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "sentiment_intensity_analysis.csv", "text/csv", key="download-csv")
