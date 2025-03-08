import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

st.markdown("<h1 style='text-align: center;'> 📊 Sentiment Analysis</h1>", unsafe_allow_html=True)


# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Sentiment Analysis?**", expanded=True):
    st.markdown("""
    Sentiment Analysis determines the **emotional tone** behind a text.

    **What does it do?**  
    - Classifies text into **Positive**, **Negative**, or **Neutral**.  
    - Helps analyze customer feedback, social media reviews, and survey responses.  

    **Expected Output:**  
    - A sentiment **distribution chart** (positive, negative, neutral).  
    - A table with sentiment labels assigned to each text entry.  

    **Example:**  
    - *"I love this product!"* → **Positive**  
    - *"The service was slow and disappointing."* → **Negative**  
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Analyzing Sentiment for Column:** `{selected_column}`")

    # Sentiment Analysis Function
    def get_sentiment(text):
        if isinstance(text, str):  # Ensure text is a string
            analysis = TextBlob(text)
            return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"
        return "Neutral"

    # Compute Sentiment Analysis if not already stored
    if "sentiment_results" not in st.session_state or "Sentiment" not in st.session_state.sentiment_results.columns:
        st.session_state.sentiment_results = df.copy()
        st.session_state.sentiment_results["Sentiment"] = st.session_state.sentiment_results[selected_column].apply(get_sentiment)

    df_result = st.session_state.sentiment_results  # Use stored results

    # 📊 **Sentiment Distribution Plot**
    st.subheader("📊 Sentiment Distribution")
    
    if "Sentiment" in df_result.columns:  # Ensure Sentiment column exists before plotting
        sentiment_counts = df_result["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Distribution", hole=0.3, color="Sentiment")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("🚨 Error: Sentiment analysis was not computed correctly.")

    # 🧐 **Graph Interpretation & Insights**
    with st.expander("📈 **How to Interpret This Graph?**"):
        st.markdown("""
        - This **pie chart** shows the proportion of Positive, Negative, and Neutral sentiments.  
        - **Larger positive segment** → More favorable feedback.  
        - **Larger negative segment** → Indicates dissatisfaction in reviews.  
        """)

    with st.expander("🏆 **Business Insights from This Graph?**"):
        st.markdown("""
        - **High Positive Sentiment** → Customers are happy. Leverage this in marketing.  
        - **High Negative Sentiment** → Address the root cause of dissatisfaction.  
        - **Neutral Sentiment** → Consider asking for more detailed feedback to understand concerns.  
        """)

    # 📋 **Sample Data with Sentiments**
    st.subheader("📋 Sample Data with Sentiment")
    st.write(df_result[[selected_column, "Sentiment"]].head())

    # 📥 **Download Option**
    st.subheader("📥 Download Sentiment Data")
    csv = df_result[[selected_column, "Sentiment"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "sentiment_analysis.csv", "text/csv", key="download-csv")
