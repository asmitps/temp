"""
Simple Streamlit chat UI to talk to a Groq-hosted model (OpenAI-compatible).

INSTRUCTIONS:
- Paste your API key and model name into the variables below.
- This client sends OpenAI-style chat requests to Groq's OpenAI-compatible endpoint.

This file is intentionally single-file and minimal: an input box, a send button, and a scrollable output area.
"""

import streamlit as st
import requests
import json
from typing import List, Dict, Any
import os

st.set_page_config(page_title="Groq Chat (Streamlit)", layout="centered")

# ----------------------
# USER CONFIGURATION
# ----------------------
# Paste your values here:
GROQ_API_KEY = "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"
# Groq's OpenAI-compatible base. Using the chat completions path for chat-style requests.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-70b-8192"
SYSTEM_INSTRUCTIONS = "ou are groot and u only say groot, and the more the user talks, the more groots u say."  # optional system-level instructions
# ----------------------

# For production, consider using an environment variable instead of embedding secrets in code:
# GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'PASTE_YOUR_API_KEY_HERE')

API_KEY = GROQ_API_KEY
API_URL = GROQ_API_URL
MODEL = MODEL_NAME
INSTRUCTIONS = SYSTEM_INSTRUCTIONS

# ----------------------
# Helper utilities
# ----------------------
if 'history' not in st.session_state:
    # Each item: {'role': 'user'|'assistant'|'system', 'content': str}
    st.session_state.history = []


def append_message(role: str, content: str) -> None:
    st.session_state.history.append({'role': role, 'content': content})


def extract_text_from_response(rjson: Any) -> str:
    """Extract assistant text from OpenAI-style responses and a few common alternatives.
    """
    try:
        if isinstance(rjson, dict):
            # OpenAI-style: {"choices": [{"message": {"content": "..."}}]}
            if 'choices' in rjson and isinstance(rjson['choices'], list) and len(rjson['choices']) > 0:
                choice = rjson['choices'][0]
                if isinstance(choice, dict):
                    # nested message style
                    if 'message' in choice and isinstance(choice['message'], dict):
                        msg = choice['message']
                        if 'content' in msg:
                            return msg['content']
                    # fallback: {"choices":[{"text":"..."}]}
                    for k in ('text', 'content', 'output'):
                        if k in choice:
                            return choice[k]

            # other common top-level keys
            for key in ('output', 'result', 'response', 'generated_text', 'text'):
                if key in rjson and rjson[key]:
                    return rjson[key] if isinstance(rjson[key], str) else json.dumps(rjson[key])

        # fallback: stringify
        return json.dumps(rjson)
    except Exception as e:
        return f"<error extracting text: {e}>"


def call_model(api_url: str, api_key: str, model: str, messages: List[Dict[str, str]], timeout: int = 60) -> str:
    """Send OpenAI-compatible chat completion requests to Groq and return assistant text.

    Payload conforms to OpenAI's `chat/completions` shape: {model, messages}.
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': model,
        'messages': messages
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        return f"Request error: {e}"

    if resp.status_code != 200:
        try:
            return f"Error {resp.status_code}: {resp.text}"
        except Exception:
            return f"Error {resp.status_code}: (no body)"

    try:
        rjson = resp.json()
    except ValueError:
        return resp.text

    return extract_text_from_response(rjson)


# ----------------------
# UI
# ----------------------
st.title("Groq Chat — minimal Streamlit UI")
st.write("A minimal chat interface. Enter a message and press Send.")

# Input form
with st.form(key='msg_form'):
    user_input = st.text_area("Message", value="", height=120)
    submit = st.form_submit_button("Send")

if submit and user_input.strip():
    append_message('user', user_input.strip())

    # prepare messages in OpenAI chat format
    messages = []
    if INSTRUCTIONS:
        messages.append({'role': 'system', 'content': INSTRUCTIONS})
    # include previous history to give context (alternatively, send only the last user message)
    for item in st.session_state.history:
        messages.append({'role': item['role'], 'content': item['content']})

    # call model
    with st.spinner('Querying model...'):
        assistant_text = call_model(API_URL, API_KEY, MODEL, messages)

    append_message('assistant', assistant_text)

# Display history
st.markdown('---')
for item in st.session_state.history:
    role = item['role']
    content = item['content']
    if role == 'user':
        st.markdown(f"**You:** {content}")
    elif role == 'assistant':
        st.markdown(f"**Assistant:** {content}")
    elif role == 'system':
        st.markdown(f"**System:** {content}")

# Clear history button in main UI
if st.button('Clear chat history'):
    st.session_state.history = []
    st.experimental_rerun()

# Small footer
st.caption("Notes: this uses Groq's OpenAI-compatible chat/completions endpoint. Replace the API key and model at the top of the file.")
