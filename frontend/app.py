import streamlit as st
import requests
import json

# Backend URL
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Academic Abstract Classifier",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Academic Abstract Classifier")
st.markdown("""
This application uses a fine-tuned **DeBERTa-v3-small** model to classify academic paper abstracts into one of 11 categories.
""")

# Input area
abstract_text = st.text_area(
    "Enter Abstract:",
    height=200,
    placeholder="Paste the abstract of the paper here..."
)

if st.button("Classify", type="primary"):
    if not abstract_text.strip():
        st.warning("Please enter an abstract to classify.")
    else:
        with st.spinner("Classifying..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": abstract_text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    label = result["label"]
                    confidence = result["confidence"]
                    
                    st.success("Classification Complete!")
                    
                    # Display result with metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Category", label)
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.2%}")
                        
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

st.markdown("---")
st.caption("Powered by FastAPI & Streamlit")
