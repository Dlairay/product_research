
from retriever import run_suggestions
from productscan import product_scan
import streamlit as st
import os
import uuid
from datetime import datetime

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
st.set_page_config(
    page_title="Product Improvement Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_product" not in st.session_state:
    st.session_state.current_product = None

def save_uploaded_file(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1]
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

st.title("Product Improvement Advisor")
st.write("Upload a product image to get started")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Handle image upload
if uploaded_file and not st.session_state.current_product:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Add to chat history
        st.session_state.messages.append({"role": "user", "type": "image", "content": file_path})
        
        # Process image
        with st.spinner("Analyzing product..."):
            try:
                product_name = product_scan(file_path)
                st.session_state.current_product = product_name
                
                # Get initial suggestions
                response = run_suggestions(product_name)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            if msg["type"] == "image":
                st.image(msg["content"], caption="Uploaded Product Image", width=200)
            else:
                st.write(msg["content"])
    
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write("**Product Analysis Report**")
            st.text(msg["content"])  # For raw formatted text

# Chat input
if prompt := st.chat_input("Ask about product improvements..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    
    # Process query
    with st.spinner("Generating response..."):
        try:
            response = run_suggestions(st.session_state.current_product)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})
            st.rerun()
        except Exception as e:
            st.error(f"Query failed: {str(e)}")