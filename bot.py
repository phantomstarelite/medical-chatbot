import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Page Configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Custom CSS
st.markdown("""
    <style>
    /* Dark Theme Base */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 2px solid #333;
    }
    
    /* Title Styles */
    h1, h2, h3, h4, h5, h6 {
        color: #4CAF50 !important;
    }
    
    /* Chat Message Styling */
    .human-message {
        background-color: #1f1f1f;
        color: #f0f0f0;
        border-left: 4px solid #2196F3;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    
    .ai-message {
        background-color: #2c2c2c;
        color: #e0e0e0;
        border-left: 4px solid #4CAF50;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    
    /* Input Styling */
    .stTextInput>div>div>input {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
        border: 2px solid #4CAF50 !important;
    }
    
    /* Radio Button Styling */
    .stRadio>div {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #2c2c2c; 
    }
    ::-webkit-scrollbar-thumb {
        background: #4CAF50; 
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
        <h2 style="color: #4CAF50;">🔧 Chatbot Settings</h2>
    </div>
    """, unsafe_allow_html=True)

    model_size = st.radio(
        "Select Model Size:",
        ["1.5B Parameters", "7B Parameters"],
        index=0,
        help="Choose the AI model size for your medical queries"
    )
    
    st.markdown("---")
    st.markdown("🚀 Quick Guide:")
    guide_steps = [
        "Pick model size",
        "Enter medical query",
        "Send & get insights"
    ]
    for step in guide_steps:
        st.markdown(f"• {step}")
    
    st.markdown("---")
    st.caption("💡 Advanced Medical AI")

# Model selection mapping
model_map = {
    "1.5B Parameters": "medllama2:latest",
    "7B Parameters": "ALIENTELLIGENCE/doctorai:latest"
}

# Initialize LangChain components
def setup_chain(model_name):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Respond in a clear and concise manner."),
        ("human", "{input}")
    ])
    
    llm = ChatOllama(
        model=model_name,
        temperature=0.5,
        num_ctx=4096
    )
    
    return prompt | llm | StrOutputParser()

# Main chat interface
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #4CAF50;">🩺 Medical Chatbot</h1>
        <p style="color: #e0e0e0;">Powered by Advanced AI Technology</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(f'<div class="human-message">{message.content}</div>', unsafe_allow_html=True)
    elif isinstance(message, AIMessage):
        with st.chat_message("ai"):
            st.markdown(f'<div class="ai-message">{message.content}</div>', unsafe_allow_html=True)

# User input handling
if prompt := st.chat_input("Type your medical question..."):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("human"):
        st.markdown(f'<div class="human-message">{prompt}</div>', unsafe_allow_html=True)
    
    # Display assistant response
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Get selected model
        selected_model = model_map[model_size]
        
        # Initialize chain
        chain = setup_chain(selected_model)
        
        # Stream response
        for chunk in chain.stream({"input": prompt}):
            full_response += chunk
            response_placeholder.markdown(f'<div class="ai-message">{full_response}▌</div>', unsafe_allow_html=True)
        
        response_placeholder.markdown(f'<div class="ai-message">{full_response}</div>', unsafe_allow_html=True)
    
    # Add AI response to chat history
    st.session_state.messages.append(AIMessage(content=full_response))