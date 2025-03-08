import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.markdown("<h1 style='text-align: center;'> 🧠 Topic Modeling </h1>", unsafe_allow_html=True)


# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Topic Modeling?**", expanded=True):
    st.markdown("""
    **Topic Modeling** identifies **hidden themes** in text.  
    It’s useful for **news articles, customer reviews, research papers**, etc.

    **Expected Output:**  
    - Each text entry will be assigned a **topic number**.  
    - A list of **keywords** defining each topic.  

    **How to Use This for Business Insights?**  
    - Discover **trending discussions** from customer feedback.  
    - Identify **key areas of interest** from research papers.  
    - Categorize **large text datasets** into meaningful groups.  
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Finding Topics in:** `{selected_column}`")

    # User input: Number of topics
    n_topics = st.slider("Select Number of Topics", min_value=2, max_value=10, value=3, step=1)

    # Ensure session state has topic_results
    if "topic_results" not in st.session_state:
        st.session_state.topic_results = {"n_topics": None}  # Initialize storage

    # Only recompute if n_topics changes
    if st.session_state.topic_results["n_topics"] != n_topics:
        df_clean = df.dropna(subset=[selected_column])  # Remove missing values

        # Convert text into numerical format
        vectorizer = CountVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(df_clean[selected_column])

        # Apply LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        topic_distribution = lda.fit_transform(X)

        # Extract topic keywords
        words = vectorizer.get_feature_names_out()
        topic_keywords = {
            i: [words[idx] for idx in topic.argsort()[-10:]] for i, topic in enumerate(lda.components_)
        }

        # Assign topics to texts
        df_clean["Topic"] = topic_distribution.argmax(axis=1)

        # Store results in session state
        st.session_state.topic_results = {
            "df": df_clean,
            "topic_keywords": topic_keywords,
            "n_topics": n_topics  # Store n_topics properly
        }

    # Load results from session state
    df_clean = st.session_state.topic_results["df"]
    topic_keywords = st.session_state.topic_results["topic_keywords"]

    # 📊 **Display Topics**
    st.subheader("🔍 Identified Topics")
    for topic_id, keywords in topic_keywords.items():
        st.write(f"**Topic {topic_id + 1}:** {', '.join(keywords)}")

    # 📊 **Topic Distribution Plot**
    st.subheader("📊 Topic Distribution")
    topic_counts = df_clean["Topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    fig = px.bar(topic_counts, x="Topic", y="Count", title="Topic Distribution", color="Topic", 
                 color_continuous_scale="viridis")
    st.plotly_chart(fig, use_container_width=True)

    # 📌 **How to Interpret This Graph?**
    with st.expander("📈 **How to Interpret This Graph?**", expanded=True):
        st.markdown("""
        - This bar chart shows the **number of text entries** assigned to each topic.
        - **Taller bars** indicate topics that appear more frequently.
        - If a topic has a **very low count**, it may not be significant.
        - The **topics with the highest bars** represent dominant themes in your data.
        - You can adjust the **number of topics** using the slider to find the best grouping.
        """)

    # 🔍 **Interactive Word Cloud for Each Topic**
    st.subheader("☁️ Word Cloud for Topics")
    topic_selected = st.selectbox("Select a Topic to View Word Cloud", list(topic_keywords.keys()))

    if topic_selected in topic_keywords:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(topic_keywords[topic_selected]))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # 📌 **How to Interpret the Word Cloud?**
    with st.expander("☁️ **How to Interpret the Word Cloud?**", expanded=True):
        st.markdown("""
        - This Word Cloud visualizes the **most important words** in the selected topic.
        - **Larger words** indicate higher relevance to the topic.
        - If certain words dominate the cloud, they define the **main theme** of that topic.
        - You can select different topics from the dropdown to see how keywords vary.
        """)

    # 📋 **Sample Data with Topics**
    st.subheader("📋 Sample Data with Topics")
    st.write(df_clean[[selected_column, "Topic"]].head())

    # 📥 **Download Option**
    st.subheader("📥 Download Topic Data")
    csv = df_clean[[selected_column, "Topic"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "topics.csv", "text/csv", key="download-topics")

    # 🏆 **Business Insights from This Visualization**
    with st.expander("🏆 **Business Insights from This Visualization**", expanded=True):
        st.markdown("""
        - **Customer Feedback Analysis**: Identify **top concerns** in reviews (e.g., "delivery", "battery life", "customer support").
        - **Market Research**: Understand **emerging trends** in industry discussions.
        - **Content Categorization**: Automatically classify **news articles, research papers, or legal documents** into themes.
        - **Sentiment Analysis Prep**: Use topics to **group feedback** before running sentiment analysis.
        - **Competitive Analysis**: Track **brand mentions & public opinion** across different topics.
        """)

