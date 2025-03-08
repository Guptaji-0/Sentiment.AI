import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.markdown("<h1 style='text-align: center;'> ☁️ Word Cloud Analysis </h1>", unsafe_allow_html=True)

# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Word Cloud Analysis?**", expanded=True):
    st.markdown("""
    **Word Cloud Analysis** visualizes the most frequently used words in text data.  
    It helps identify key themes, sentiments, and customer concerns.  

    **Expected Output:**  
    - A cloud of words where **frequent words appear larger**.  
    - Insights into common topics in customer feedback or reviews.  

    **How to Use This Data for Business Insights?**  
    - Identify **trending topics** in customer feedback.  
    - Detect **positive or negative themes** in reviews.  
    - Improve **marketing campaigns** by targeting popular keywords.  

    **Example:**  
    - Reviews contain words like **"fast", "quality", "bad service"**  
    - These words will appear larger, showing key themes in the data.  
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Generating Word Cloud for:** `{selected_column}`")

    # Check if Word Cloud was already computed
    if "word_cloud" not in st.session_state:
        # Combine all text data into a single string
        text_data = " ".join(df[selected_column].dropna().astype(str))

        # Define stopwords to remove common words
        stopwords = set(STOPWORDS)

        # Generate Word Cloud
        wordcloud = WordCloud(
            width=800, height=400, background_color="white",
            stopwords=stopwords, colormap="viridis"
        ).generate(text_data)

        # Store Word Cloud in session state
        st.session_state.word_cloud = wordcloud

    # Load Word Cloud from session state
    wordcloud = st.session_state.word_cloud

    # 📊 **Display Word Cloud**
    st.subheader("📊 Word Cloud Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # 🔽 **Dropdown: How to Interpret This Graph?**
    with st.expander("📈 **How to Interpret This Word Cloud?**"):
        st.markdown("""
        - Words that **appear larger** are used **more frequently** in the text data.  
        - **Common themes** can be detected from frequently occurring words.  
        - If certain words dominate the cloud (e.g., "slow", "bad service"), they may indicate problems.  
        """)

    # 🔽 **Dropdown: Business Insights from This Visualization**
    with st.expander("🏆 **Business Insights from This Visualization**"):
        st.markdown("""
        - Identify **customer concerns** by finding frequently mentioned negative words.  
        - Highlight **strong brand attributes** by identifying positive themes.  
        - Use high-frequency words in **SEO & marketing campaigns**.  
        """)

    # 📋 **Word Frequency Table**
    st.subheader("📋 Top 15 Most Frequent Words")
    word_freq = pd.DataFrame(wordcloud.words_.items(), columns=["Word", "Frequency"]).head(15)
    st.write(word_freq)

    # 📊 **Word Frequency Bar Chart**
    st.subheader("📊 Word Frequency Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(word_freq["Word"], word_freq["Frequency"], color="skyblue")
    ax.set_xticklabels(word_freq["Word"], rotation=45, ha="right")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 15 Most Frequent Words")
    st.pyplot(fig)

    # 📥 **Download Option**
    st.subheader("📥 Download Word Frequency Data")
    csv = word_freq.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "word_frequency.csv", "text/csv", key="download-words")
