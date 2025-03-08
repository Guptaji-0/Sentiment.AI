import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

st.markdown("<h1 style='text-align: center;'> 🛒 Product/Feature Sentiment Breakdown </h1>", unsafe_allow_html=True)


# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Product/Feature Sentiment Analysis?**", expanded=True):
    st.markdown("""
    **Feature sentiment analysis** extracts customer opinions on specific  
    aspects (e.g., price, quality, service) from reviews.

    **Expected Output:**  
    - Identifies **positive & negative sentiment** for each feature.  
    - Helps businesses understand which aspects customers like or dislike.  

    **How to Use This Data for Business Insights?**  
    - Improve **weak features** that receive negative sentiment.  
    - Highlight **strong features** in marketing campaigns.  
    - Optimize **product/service offerings** based on customer feedback.  

    **Example:**  
    - "The **price** is great, but the **delivery** was slow."  
      → **Price Sentiment: Positive (0.8)**  
      → **Delivery Sentiment: Negative (-0.4)**  
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Analyzing Sentiment for Different Features in:** `{selected_column}`")

    # Check if sentiment analysis was already computed
    if "feature_sentiments" not in st.session_state:
        # Define features to analyze
        features = ["price", "quality", "service", "delivery", "experience"]
        feature_sentiment = {feature: [] for feature in features}

        # Compute sentiment for each feature in reviews
        for review in df[selected_column].dropna():
            for feature in features:
                if feature in review.lower():
                    sentiment = TextBlob(review).sentiment.polarity
                    feature_sentiment[feature].append(sentiment)

        # Convert results to DataFrame
        feature_sentiment_df = pd.DataFrame([
            {"Feature": k, "Sentiment": sum(v) / len(v) if v else 0} for k, v in feature_sentiment.items()
        ])

        # Store results in session state
        st.session_state.feature_sentiments = feature_sentiment_df

    # Load results from session state
    feature_sentiment_df = st.session_state.feature_sentiments

    # 📊 **Feature Sentiment Breakdown**
    st.subheader("📊 Feature Sentiment Breakdown")
    fig = px.bar(feature_sentiment_df, x="Feature", y="Sentiment", title="Feature Sentiment Breakdown", color="Feature")
    st.plotly_chart(fig, use_container_width=True)

    # 🧐 **Graph Interpretation & Insights**
    with st.expander("📈 **How to Interpret This Graph?**"):
        st.markdown("""
        - This bar chart shows the **average sentiment score** for each feature.  
        - **Positive scores** indicate a **favorable opinion**, while **negative scores** indicate dissatisfaction.  
        """)

    with st.expander("🏆 **Business Insights from This Graph?**"):
        st.markdown("""
        - **Features with high sentiment** (e.g., Price: 0.8) should be highlighted in marketing efforts.  
        - **Features with negative sentiment** (e.g., Delivery: -0.4) indicate areas that need improvement.  
        - **A near-zero sentiment score** means mixed opinions; investigate customer feedback further.  
        """)

    # 📋 **Sample Sentiment Scores**
    st.subheader("📋 Feature Sentiment Scores")
    st.write(feature_sentiment_df)

    # 📥 **Download Option**
    st.subheader("📥 Download Sentiment Data")
    csv = feature_sentiment_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "feature_sentiment.csv", "text/csv", key="download-sentiments")
