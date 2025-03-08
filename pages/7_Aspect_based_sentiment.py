import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

nltk.download("punkt")
nltk.download("stopwords")


st.markdown("<h1 style='text-align: center;'> Aspect-Based Sentiment Analysis </h1>", unsafe_allow_html=True)

# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Aspect-Based Sentiment Analysis?**", expanded=True):
    st.markdown("""
    Aspect-Based Sentiment Analysis (ABSA) identifies **specific aspects** within a text and determines their sentiment.

    **How does it work?**  
    - Extracts key aspects (e.g., **Product Quality, Customer Service, Pricing**).  
    - Analyzes sentiment for **each aspect separately**.  

    **Example:**  
    - *"The battery life is great, but the camera is terrible!"*  
      - **Battery Life → Positive**  
      - **Camera → Negative**
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Analyzing Aspects in Column:** `{selected_column}`")

    # **Aspect Extraction & Sentiment Analysis**
    def aspect_sentiment_analysis(text):
        if not isinstance(text, str):
            return {}

        sentences = sent_tokenize(text)
        aspect_sentiments = defaultdict(list)

        for sentence in sentences:
            analysis = TextBlob(sentence)
            polarity = analysis.sentiment.polarity

            # Simple aspect keyword mapping (customize this)
            aspects = {
                "battery": "Battery Life",
                "camera": "Camera",
                "service": "Customer Service",
                "price": "Pricing",
                "delivery": "Delivery Experience",
                "design": "Design & Build",
            }

            for word in sentence.lower().split():
                if word in aspects:
                    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
                    aspect_sentiments[aspects[word]].append(sentiment)

        # Get the most common sentiment for each aspect
        return {aspect: max(set(sentiments), key=sentiments.count) for aspect, sentiments in aspect_sentiments.items()}

    # Compute Aspect-Based Sentiment Analysis if not already stored
    if "aspect_sentiment_results" not in st.session_state:
        st.session_state.aspect_sentiment_results = df.copy()
        st.session_state.aspect_sentiment_results["Aspect Sentiment"] = st.session_state.aspect_sentiment_results[selected_column].apply(aspect_sentiment_analysis)

    df_result = st.session_state.aspect_sentiment_results  # Use stored results

    # 📊 **Aspect Sentiment Distribution**
    st.subheader("📊 Aspect Sentiment Distribution")

    aspect_data = []
    for _, row in df_result.iterrows():
        for aspect, sentiment in row["Aspect Sentiment"].items():
            aspect_data.append({"Aspect": aspect, "Sentiment": sentiment})

    aspect_df = pd.DataFrame(aspect_data)
    
    if not aspect_df.empty:
        fig = px.bar(aspect_df, x="Aspect", color="Sentiment", title="Aspect Sentiment Analysis", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("🚨 No aspects were identified in the text.")

    # 📈 **Graph Interpretation**
    with st.expander("📈 **How to Interpret This Graph?**"):
        st.markdown("""
        - This **bar chart** shows sentiment distribution for each aspect.  
        - **More positive bars** → Good customer experience for that aspect.  
        - **More negative bars** → A common complaint in reviews.  
        """)

    # 🏆 **Business Insights**
    with st.expander("🏆 **Business Insights from This Graph?**"):
        st.markdown("""
        - **Negative Pricing Sentiment?** → Consider adjusting pricing or offering discounts.  
        - **Negative Delivery Sentiment?** → Improve logistics and shipping times.  
        - **Positive Product Quality?** → Highlight in marketing campaigns.  
        """)

    # 📋 **Sample Data with Aspect Sentiments**
    st.subheader("📋 Sample Data with Aspect Sentiment")
    st.write(df_result[[selected_column, "Aspect Sentiment"]].head())

    # 📥 **Download Option**
    st.subheader("📥 Download Aspect Sentiment Data")
    df_download = df_result[[selected_column, "Aspect Sentiment"]].copy()
    df_download["Aspect Sentiment"] = df_download["Aspect Sentiment"].apply(lambda x: str(x))  # Convert dict to string
    csv = df_download.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "aspect_sentiment_analysis.csv", "text/csv", key="download-csv")
