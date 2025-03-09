import nltk
nltk.download('punkt')   # Required for tokenization
nltk.download('averaged_perceptron_tagger')  # Required for POS tagging
nltk.download('brown')   # Required for text classification
nltk.download('wordnet') # Required for lemmatization
nltk.download('stopwords')  # Required for stopwords filtering
nltk.download('movie_reviews')  # Required for TextBlob's sentiment analysis
nltk.download('vader_lexicon')  # Required for VADER sentiment analysis
# Download required corpora
nltk.download('punkt')
textblob.download_corpora()
import streamlit as st
import pandas as pd
import plotly.express as px
from nrclex import NRCLex

st.markdown("<h1 style='text-align: center;'> 😊 Emotion Detection </h1>", unsafe_allow_html=True)
#st.title("")

# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Emotion Detection?**", expanded=True):
    st.markdown("""
    Emotion Detection identifies **emotions** in text (e.g., Joy, Anger, Sadness).  
    It helps analyze **customer reviews, social media comments, and feedback**.

    **Expected Output:**  
    - Each text entry will be labeled with an **emotion** (Joy, Fear, Sadness, etc.).  
    - A **bar chart** showing the distribution of detected emotions.  

    **How to Use This for Business Insights?**  
    - **Joy/Dominance → Positive impact** → Leverage positive emotions in marketing.  
    - **Anger/Sadness → Negative impact** → Identify pain points and improve services.  
    - **Surprise/Anticipation → Curiosity** → Identify trends for targeted promotions.  

    **Example:**  
    - *"I love this product! It's amazing!"* → **Emotion: Joy**  
    - *"The service was terrible and frustrating!"* → **Emotion: Anger**  
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Detecting Emotions for Column:** `{selected_column}`")

    # Emotion Detection Function
    def get_emotions(text):
        if isinstance(text, str):  # Ensure text is a string
            emotions = NRCLex(text).top_emotions
            return max(emotions, key=lambda x: x[1])[0] if emotions else "Neutral"
        return "Neutral"

    # Recompute if dataset/column changes
    if "emotion_results" not in st.session_state or \
       st.session_state.emotion_results is None or \
       "prev_data" not in st.session_state or \
       st.session_state.prev_data != df[selected_column].tolist():
        
        df_result = df.copy()
        df_result["Emotion"] = df_result[selected_column].apply(get_emotions)

        # Store results in session state
        st.session_state.emotion_results = df_result
        st.session_state.prev_data = df[selected_column].tolist()  # Store dataset snapshot

    df_result = st.session_state.emotion_results  # Use stored results

    # 📊 **Emotion Distribution Plot**
    st.subheader("📊 Emotion Breakdown")
    emotion_counts = df_result["Emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]
    fig = px.bar(emotion_counts, x="Emotion", y="Count", title="Emotion Distribution", color="Emotion")
    st.plotly_chart(fig, use_container_width=True)

    # 📌 **How to Interpret This Graph?**
    with st.expander("📈 **How to Interpret This Graph?**", expanded=True):
        st.markdown("""
        - This bar chart shows the **frequency of emotions** detected in the dataset.
        - **Higher bars** indicate emotions that appear more frequently in the text.
        - A **dominant positive emotion** (e.g., Joy) suggests positive customer sentiment.
        - If **negative emotions (e.g., Anger, Fear, Sadness) dominate**, it may indicate service or product issues.
        - The emotion mix can help businesses **adjust their messaging** to better connect with customers.
        """)

    # 📋 **Sample Data with Emotions**
    st.subheader("📋 Sample Data with Emotions")
    st.write(df_result[[selected_column, "Emotion"]].head())

    # 📥 **Download Option**
    st.subheader("📥 Download Emotion Data")
    csv = df_result[[selected_column, "Emotion"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "emotion_analysis.csv", "text/csv", key="download-csv")

    # 🏆 **Business Insights from This Visualization**
    with st.expander("🏆 **Business Insights from This Visualization**", expanded=True):
        st.markdown("""
        - **Customer Feedback Monitoring**: Identify whether customers are happy or frustrated.
        - **Brand Reputation Management**: Track sentiment trends in **social media mentions**.
        - **Product Improvement**: If anger or sadness dominates, investigate customer complaints.
        - **Marketing Strategy**: Use positive emotions (Joy, Trust) to craft **effective campaigns**.
        - **Customer Support Enhancement**: Adjust support responses based on detected emotions.
        - **Crisis Detection**: A sudden rise in negative emotions can indicate **urgent issues** needing attention.
        """)

