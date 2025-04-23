import streamlit as st
from streamlit_chat import message
from PIL import Image
import io
import os
import uuid
from datetime import datetime

# Add your existing backend imports here
from retriever import run_suggestions
from productscan import product_scan

# Page Config
st.set_page_config(
    page_title="Product Improvement Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Session State Initialization
if 'conversation' not in st.session_state:
    st.session_state.conversation = {
        'past': [],
        'generated': [],
        'product': None,
        'qa_chain': None
    }

# Main Layout
# st.markdown('''
#     <style>
#         .main {
#             background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#             padding: 2rem;
#         }
#         .chat-container {
#             background: white;
#             border-radius: 15px;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             height: 70vh;
#             padding: 2rem;
#             overflow-y: auto;
#         }
#     </style>
# ''', unsafe_allow_html=True)

st.markdown('''
    <style>
        .image-message {
            background: #e6f3ff;
            border-radius: 20px;
            padding: 15px;
            margin: 10px 0;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .image-message img {
            border-radius: 15px;
            max-width: 100%;
            height: auto;
        }
        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-left: 10px;
            float: right;
        }
    </style>
''', unsafe_allow_html=True)

st.markdown('<h1 class="title">Product Improvement Advisor 🔍</h1>', unsafe_allow_html=True)

# Chat Containers
chat_container = st.container()
input_container = st.container()

# Image Processing
def handle_image_upload(uploaded_file):
    try:
        file_ext = uploaded_file.name.split('.')[-1]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        filename = f"{timestamp}_{unique_id}.{file_ext}"
        file_path = os.path.join("img/", filename)

        # Save file to upload directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Store both path and image in session state
        image = Image.open(file_path)
        st.session_state.conversation['product'] = product_scan(file_path)
        # image = Image.open(io.BytesIO(uploaded_fil.read()))
        # st.session_state.conversation['product'] = product_scan(uploaded_file)e
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Chat Functions
def generate_response():
    product = st.session_state.conversation['product']
    if product:
        try:
            return run_suggestions(product)
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"
    return "Please upload a product image first"

# Input Handling
with input_container:
    col1, col2 = st.columns([1, 4])
    with col1:
        uploaded_file = st.file_uploader("📤 Upload Product Image", 
                                      type=["jpg", "png", "jpeg"],
                                      key="file_uploader",
                                      help="Upload a clear image of the product")
    with col2:
        with st.form(key='chat_form'):
            user_input = st.text_input("Your question:", 
                                      placeholder="Ask about product improvements...",
                                      key="user_input")
            submit_btn = st.form_submit_button("Send")

    if uploaded_file and not st.session_state.conversation['product']:
        image = handle_image_upload(uploaded_file)
        if image:
            st.session_state.conversation['past'].append(("image", image))
            response = generate_response()
            st.session_state.conversation['generated'].append(response)

    if submit_btn and user_input:
        st.session_state.conversation['past'].append(("text", user_input))
        response = generate_response()
        st.session_state.conversation['generated'].append(response)

# Display Chat History
with chat_container:
    if st.session_state.conversation['generated']:
        for i in range(len(st.session_state.conversation['generated'])):
            # Display user input (image or text)
            msg_type, content = st.session_state.conversation['past'][i]
            if msg_type == "image":
                st.markdown(f'''
                    <div class="image-message">
                        <img src="{content}" alt="Uploaded product image">
                    </div>
                    <img src="https://api.dicebear.com/7.x/adventurer/svg?seed={i}" class="user-avatar">
                ''', unsafe_allow_html=True)
            else:
                message(content, is_user=True, avatar_style="adventurer")
            
            # Display bot response
            message(st.session_state.conversation['generated'][i], 
                   avatar_style="bottts", 
                   key=str(i))