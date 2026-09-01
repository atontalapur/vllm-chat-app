"""Streamlit chat interface.

Talks to the application layer, never to the model server directly — that is
deliberate. Routing through the api means auth, validation, and logging are
exercised on every message rather than being decorative.

Conversation history lives here, in session state, and is resent with each
turn. The api is stateless. History is trimmed so a long session cannot grow
the prompt until it overruns the model's context window.

    user types
        |
        v
    history + new message  ---> POST /chat/stream (X-API-Key)
        |                              |
        |                        SSE chunks
        v                              |
    st.write_stream  <-----------------+
"""

import json
import os
from collections.abc import Iterator

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8080")
API_KEY = os.environ.get("API_KEY", "")
# Turns kept in the prompt. Older turns stay on screen but stop being sent.
MAX_TURNS = int(os.environ.get("MAX_TURNS", "20"))
REQUEST_TIMEOUT_S = 300

st.set_page_config(page_title="vllm-chat-app", page_icon="[]", layout="centered")


def fail_fast() -> None:
    """Stop immediately on missing configuration.

    Without this the UI would send an empty key and every message would come
    back 401, which reads like a broken server rather than an unset variable.
    """
    if not API_KEY:
        st.error(
            "**API_KEY is not set.** The UI cannot authenticate to the api service. "
            "Copy `.env.example` to `.env` and set `API_KEY`, then restart."
        )
        st.stop()


def stream_reply(messages: list[dict[str, str]]) -> Iterator[str]:
    """Send the conversation and yield tokens as they arrive.

    Errors are raised as RuntimeError with a human-readable message; the caller
    renders it in an error banner. Silence is the one unacceptable outcome — a
    user staring at an empty box has no idea whether to wait or retry.
    """
    try:
        response = requests.post(
            f"{API_URL}/chat/stream",
            json={"messages": messages, "max_tokens": 1024, "temperature": 0.7},
            headers={"X-API-Key": API_KEY},
            stream=True,
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach the api service at {API_URL}. Is the stack running?"
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError("The api service did not respond in time.") from None

    if response.status_code != 200:
        raise RuntimeError(_describe_error(response))

    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        # The proxy reports a mid-stream upstream failure on the stream itself,
        # since the 200 status has already been sent by that point.
        if "error" in chunk:
            raise RuntimeError(chunk.get("detail", "The model server failed mid-response."))

        choices = chunk.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            yield content


def _describe_error(response: requests.Response) -> str:
    """Turn an api error response into something a person can act on."""
    try:
        detail = response.json().get("detail", "")
    except (ValueError, AttributeError):
        detail = response.text[:300]

    if response.status_code == 401:
        return "Authentication failed. The UI's API_KEY does not match the api service."
    if response.status_code == 413:
        return f"{detail}"
    if response.status_code == 422:
        return f"The request was rejected as invalid: {detail}"
    if response.status_code == 502:
        return f"The model server is unavailable: {detail}"
    return f"Unexpected error {response.status_code}: {detail}"


def main() -> None:
    fail_fast()

    st.title("vllm-chat-app")
    st.caption(f"Chat served by a self-hosted model, proxied through {API_URL}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Only the most recent turns are sent; the rest remain visible on screen.
    history = st.session_state.messages[-(MAX_TURNS * 2) :]

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(history))
        except RuntimeError as exc:
            st.error(str(exc))
            # Drop the unanswered turn so a retry does not resend a dangling
            # user message with no reply after it.
            st.session_state.messages.pop()
            return

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
