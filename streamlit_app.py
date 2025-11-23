import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/search"

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Semantic Document Search")
st.subheader("Powered by MiniLM + FAISS + FastAPI + Streamlit")

st.write("---")

# --- Search Input ---
query = st.text_input("Enter your search query:", placeholder="e.g., machine learning basics")

top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query before searching.")
    else:
        with st.spinner("Searching..."):
            payload = {"query": query, "top_k": top_k}

            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                st.error(f"Error contacting API: {e}")
                st.stop()

        results = data.get("results", [])

        st.write(f"### 🔎 Found {len(results)} matching documents")
        st.write("---")

        # --- Render Results ---
        for item in results:
            doc_id = item["doc_id"]
            score = item["score"]
            preview = item["preview"]
            explanation = item["explanation"]

            with st.container():
                st.markdown(f"## 📄 Document **{doc_id}**  — Score: **{score:.4f}**")

                st.markdown(f"**Preview:** {preview}")

                with st.expander("📘 Why was this document matched? (Explanation)"):
                    st.write(f"**Reason:** {explanation['why_matched']}")
                    st.write(f"**Overlap Keywords:** {explanation['overlap_keywords']}")
                    st.write(f"**Overlap Ratio:** {explanation['overlap_ratio']:.4f}")
                    st.write(f"**Document Length Norm:** {explanation['doc_length_norm']:.4f}")

                st.write("---")
