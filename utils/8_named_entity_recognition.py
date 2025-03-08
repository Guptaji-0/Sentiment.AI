'''import streamlit as st
import spacy
import pandas as pd

st.title("📝 Named Entity Recognition (NER)")

# Load spaCy model with caching
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")  # Change to "en_core_web_trf" for better speed

nlp = load_model()

# 📌 **Feature Explanation Card**
with st.expander("ℹ️ **What is Named Entity Recognition (NER)?**", expanded=True):
    st.markdown("""
    Named Entity Recognition (NER) identifies **important entities** in text, such as:
    - **People** (e.g., Elon Musk)
    - **Organizations** (e.g., Google)
    - **Locations** (e.g., New York)
    - **Dates, Products, Events, and More!** 
    """)

# Ensure dataset & column selection exists
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Please upload a dataset in the **Upload Data** page.")
elif "selected_column" not in st.session_state or st.session_state.selected_column is None:
    st.warning("⚠️ Please select a text column in the **Upload Data** page.")
else:
    df = st.session_state.data
    selected_column = st.session_state.selected_column

    st.write(f"✅ **Performing Named Entity Recognition on Column:** `{selected_column}`")

    # Named Entity Recognition Function (Optimized)
    @st.cache_data
    def extract_named_entities(text_list):
        docs = list(nlp.pipe(text_list))  # Process texts in batch
        results = [" | ".join([f"{ent.text} ({ent.label_})" for ent in doc.ents]) if doc.ents else "No Entities Found" for doc in docs]
        return results

    # Perform NER only once and store it in session state
    if "ner_results" not in st.session_state:
        st.session_state.ner_results = df.copy()
        st.session_state.ner_results["Named Entities"] = extract_named_entities(df[selected_column].astype(str))

    df_result = st.session_state.ner_results  # Use stored results

    # 📋 **Sample Data with Named Entities**
    st.subheader("📋 Sample Data with Named Entities")
    if "Named Entities" in df_result.columns:
        st.write(df_result[[selected_column, "Named Entities"]].head())
    else:
        st.error("🚨 Error: Named Entity Recognition was not computed correctly.")

    # 📥 **Download Option**
    st.subheader("📥 Download NER Data")
    csv = df_result[[selected_column, "Named Entities"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "named_entities.csv", "text/csv", key="download-ner-csv")'''
