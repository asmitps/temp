"""
Simple Streamlit chat UI to talk to a Groq-hosted model.

INSTRUCTIONS:
- Paste your API key, model name and any instructions into the variables below (or use the sidebar override).
- If Groq's API expects a different request shape, adapt the payload construction in `call_model` accordingly.

This file is intentionally single-file and minimal: an input box, a send button, and a scrollable output area.
"""

import streamlit as st
import requests
import json
from typing import List, Dict, Any

st.set_page_config(page_title="Groq Chat (Streamlit)", layout="centered")

# ----------------------
# USER CONFIGURATION
# ----------------------
# Paste your values here (or override them in the sidebar):
GROQ_API_KEY = "PASTE_YOUR_API_KEY_HERE"
GROQ_API_URL = "https://api.groq.ai/v1/generate"  # example; replace with the actual endpoint if different
MODEL_NAME = "your-groq-model-name"
SYSTEM_INSTRUCTIONS = "You are a helpful assistant."  # optional system-level instructions
# ----------------------

# Sidebar overrides (convenience for quick testing without editing the file):
st.sidebar.header("Configuration (overrides)")
api_key_input = st.sidebar.text_input("API Key", value=GROQ_API_KEY, type="password")
api_url_input = st.sidebar.text_input("API URL", value=GROQ_API_URL)
model_input = st.sidebar.text_input("Model name", value=MODEL_NAME)
instructions_input = st.sidebar.text_area("System instructions", value=SYSTEM_INSTRUCTIONS, height=120)

USE_SIDEBAR = st.sidebar.checkbox("Use sidebar values (recommended)", value=True)

API_KEY = api_key_input if USE_SIDEBAR else GROQ_API_KEY
API_URL = api_url_input if USE_SIDEBAR else GROQ_API_URL
MODEL = model_input if USE_SIDEBAR else MODEL_NAME
INSTRUCTIONS = instructions_input if USE_SIDEBAR else SYSTEM_INSTRUCTIONS

# ----------------------
# Helper utilities
# ----------------------
if 'history' not in st.session_state:
    # Each item: {'role': 'user'|'assistant'|'system', 'content': str}
    st.session_state.history = []


def append_message(role: str, content: str) -> None:
    st.session_state.history.append({'role': role, 'content': content})


def extract_text_from_response(rjson: Any) -> str:
    """Try several common patterns used by LLM APIs to extract assistant text.
    This makes the client robust to minor differences in reply shape.
    """
    try:
        if isinstance(rjson, dict):
            # common: {"output": "..."} or {"result": "..."}
            for key in ('output', 'result', 'response', 'generated_text', 'text'):
                if key in rjson and rjson[key]:
                    return rjson[key] if isinstance(rjson[key], str) else json.dumps(rjson[key])

            # common: {"choices": [{"message": {"content": "..."}}]} or {"choices":[{"text":"..."}]}
            if 'choices' in rjson and isinstance(rjson['choices'], list) and len(rjson['choices']) > 0:
                choice = rjson['choices'][0]
                # nested message style
                if isinstance(choice, dict):
                    if 'message' in choice and isinstance(choice['message'], dict):
                        msg = choice['message']
                        for k in ('content', 'text'):
                            if k in msg:
                                return msg[k]
                    for k in ('text', 'content', 'output'):
                        if k in choice:
                            return choice[k]

        # fallback: stringify
        return json.dumps(rjson)
    except Exception as e:
        return f"<error extracting text: {e}>"


def call_model(api_url: str, api_key: str, model: str, messages: List[Dict[str, str]], instructions: str = None, timeout: int = 60) -> str:
    """Send messages to the LLM provider and return assistant text.
    NOTE: The request body below is a best-effort template. Adapt this to Groq's exact schema if needed.
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # Construct a simple payload that many LLM endpoints accept.
    payload = {
        'model': model,
        # Provide messages in a chat-style shape. Many APIs accept 'messages' or an 'input' string.
        'messages': messages,
    }
    if instructions:
        # include system/instructions field if desired by the provider
        payload['instructions'] = instructions

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        return f"Request error: {e}"

    if resp.status_code != 200:
        # try to include response body for debugging
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

with st.form(key='msg_form'):
    user_input = st.text_area("Message", value="", height=120)
    submit = st.form_submit_button("Send")

if submit and user_input.strip():
    append_message('user', user_input.strip())

    # prepare chat-style messages (start with an optional system role)
    messages = []
    if INSTRUCTIONS:
        messages.append({'role': 'system', 'content': INSTRUCTIONS})
    # include previous history to give context (alternatively, send only the last user message)
    for item in st.session_state.history:
        # only include last N messages if you want to limit size
        messages.append({'role': item['role'], 'content': item['content']})

    # call model
    with st.spinner('Querying model...'):
        assistant_text = call_model(API_URL, API_KEY, MODEL, messages, instructions=INSTRUCTIONS)

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

# Clear history button
if st.sidebar.button('Clear chat history'):
    st.session_state.history = []
    st.experimental_rerun()

# Small footer
st.caption("Notes: adapt `API_URL` and payload in call_model(...) if Groq expects a different request shape.")
