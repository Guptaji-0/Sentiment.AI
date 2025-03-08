import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from textblob import TextBlob

st.markdown("<h1 style='text-align: center;'> 🎭 Sentiment-Based Customer Segmentations </h1>", unsafe_allow_html=True)


# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Sentiment-Based Customer Segmentation?**", expanded=True):
    st.markdown("""
    **Customer Segmentation** groups customers based on **sentiment analysis**  
    of their reviews, feedback, or comments.

    **Expected Output:**  
    - Each customer is assigned to a **cluster** (e.g., Positive, Neutral, Negative).  
    - A **sentiment score** is calculated for each comment.  

    **How to Use This for Business Insights?**  
    - Identify **satisfied customers** to enhance loyalty programs.  
    - Detect **negative feedback** to improve customer experience.  
    - Personalize **marketing strategies** for different segments.  

    **Example:**  
    - "I love this product!" → **Sentiment Score: 0.8 → Cluster: Positive**  
    - "It’s okay, but could be better." → **Sentiment Score: 0.1 → Cluster: Neutral**  
    - "Terrible experience, won’t buy again!" → **Sentiment Score: -0.7 → Cluster: Negative**  
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Segmenting Customers Based on Sentiment in:** `{selected_column}`")

    # Check if clustering was already computed
    if "sentiment_clusters" not in st.session_state:
        df_clean = df.dropna(subset=[selected_column])  # Remove missing values

        # Get sentiment scores
        df_clean["Sentiment Score"] = df_clean[selected_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

        # Apply clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_clean["Cluster"] = kmeans.fit_predict(df_clean[["Sentiment Score"]])

        # Store results in session state
        st.session_state.sentiment_clusters = df_clean

    # Load results from session state
    df_clean = st.session_state.sentiment_clusters

    # 📊 **Customer Segment Distribution**
    st.subheader("📊 Customer Segment Distribution")
    fig = px.histogram(df_clean, x="Cluster", title="Customer Segments Based on Sentiment", nbins=3, color="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # 📖 **Graph Interpretation (Dropdown)**
    with st.expander("📈 **How to Interpret This Graph?**"):
        st.markdown("""
        - This graph shows **how many customers** fall into each sentiment segment.  
        - Each bar represents a **customer segment**, where:  
          - **Cluster 0:** Likely **Negative Sentiment Customers** (low sentiment scores)  
          - **Cluster 1:** Likely **Neutral Sentiment Customers** (mid-range scores)  
          - **Cluster 2:** Likely **Positive Sentiment Customers** (high scores)  
        """)

    # 🏆 **Business Insights (Dropdown)**
    with st.expander("🏆 **Business Insights from This Visualization**"):
        st.markdown("""
        - If **negative sentiment customers (Cluster 0) are high**, you need to address **common complaints**.  
        - If **positive sentiment customers (Cluster 2) dominate**, your product/service is well-received!  
        - If **neutral sentiment customers (Cluster 1) are high**, focus on **enhancing their experience** to turn them into positive customers.  
        """)

    # 📋 **Sample Data with Sentiments**
    st.subheader("📋 Sample Data with Sentiments & Clusters")
    st.write(df_clean[[selected_column, "Sentiment Score", "Cluster"]].head())

    # 📥 **Download Option**
    st.subheader("📥 Download Segmented Data")
    csv = df_clean[[selected_column, "Sentiment Score", "Cluster"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "customer_segments.csv", "text/csv", key="download-segments")
