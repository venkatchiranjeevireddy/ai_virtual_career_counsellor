
import streamlit as st
import requests
import uuid
import time
import os
import PyPDF2
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(
    page_title="Elite AI Career Counsellor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Environment Variables ---
load_dotenv()

# --- Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 100%;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
        border: 1px solid #e1e4e8;
    }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #4B8BBE;
        background-color: #4B8BBE;
        color: white;
        transition: all 0.2s;
        font-weight: 500;
    }
    .stButton>button:hover {
        border: 1px solid #1E5A8C;
        background-color: #1E5A8C;
    }
    .stChatInputContainer {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Rasa Server Configuration ---
RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"

# --- Helper Functions ---
def parse_resume(file_path):
    """Extracts text from a PDF or TXT file."""
    text = ""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
    return text

def send_message_to_rasa(message, sender_id):
    """Sends a message to the Rasa server and gets the response."""
    try:
        payload = {
            "sender": sender_id,
            "message": message
        }
        response = requests.post(RASA_SERVER_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the AI engine. Please ensure Rasa is running. Error: {e}")
        return [{"text": "I'm having trouble connecting to my brain right now. Please check if the Rasa server is running and try again."}]
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return [{"text": "Oops, something went wrong on my end."}]

# --- Main App Logic ---
def main():
    st.title("Elite AI Career Counsellor 🤖✨")
    st.markdown("Welcome! I'm here to provide hyper-personalized career guidance using advanced AI. Let's start your journey.")

    # --- Initialize Session State ---
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{uuid.uuid4()}"

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Initial greeting from the bot
        initial_bot_message = "Hello! I am your AI Career Counsellor. To start, what is your name?"
        st.session_state.messages.append({"role": "assistant", "content": initial_bot_message})
    
    if "report_path" not in st.session_state:
        st.session_state.report_path = None


    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Restart Conversation"):
            st.session_state.messages = []
            st.session_state.session_id = f"session_{uuid.uuid4()}"
            st.session_state.report_path = None
            st.success("Conversation restarted.")
            st.rerun()

        st.header("Upload Your Resume")
        uploaded_file = st.file_uploader("Upload a PDF or TXT file for deeper analysis", type=["pdf", "txt"])
        if uploaded_file is not None:
            with st.spinner("Analyzing resume..."):
                if not os.path.exists("uploads"):
                    os.makedirs("uploads")
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                resume_text = parse_resume(file_path)
                # Send resume text to Rasa to be stored in a slot
                send_message_to_rasa(f"/inform_resume{{\"resume_keywords\": \"{resume_text.replace('\"', ' ')}\"}}", st.session_state.session_id)
                st.success("Resume analyzed successfully!")
                st.info("Your resume has been processed and will be used for recommendations.")
                os.remove(file_path)

        st.header("Download Report")
        if st.session_state.report_path and os.path.exists(st.session_state.report_path):
             with open(st.session_state.report_path, "rb") as pdf_file:
                PDFbyte = pdf_file.read()
             st.download_button(label="Download Career Report",
                                data=PDFbyte,
                                file_name=os.path.basename(st.session_state.report_path),
                                mime='application/octet-stream')
        else:
            st.info("Your report will be available here once a career has been recommended.")

    # --- Chat Interface ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your response..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                bot_responses = send_message_to_rasa(prompt, st.session_state.session_id)
                
                full_response = ""
                for r in bot_responses:
                    full_response += r.get("text", "") + "\n\n"
                    if r.get("buttons"):
                        full_response += "**Choose an option by typing its title:**\n"
                        for button in r.get("buttons"):
                            full_response += f"- `{button['title']}`\n"
                    
                    # Check for custom data, like a report path
                    if "report_path" in r.get("custom", {}):
                        st.session_state.report_path = r["custom"]["report_path"]

                message_placeholder.markdown(full_response.strip())
                st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
                
                # Rerun to update the download button if a report was generated
                if "report_path" in [r.get("custom", {}) for r in bot_responses]:
                    st.rerun()

if __name__ == "__main__":
    main()