import os
from pathlib import PurePosixPath

import requests
import streamlit as st
from databricks.sdk.config import Config
from dotenv import load_dotenv

load_dotenv()

SERVING_ENDPOINT_NAME = os.environ["SERVING_ENDPOINT_NAME"]

cfg = Config()
ENDPOINT_URL = f"{cfg.host}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"
HEADERS = {"Authorization": f"Bearer {cfg.token}", "Content-Type": "application/json"}


def query_endpoint(messages: list) -> dict:
    response = requests.post(
        ENDPOINT_URL,
        headers=HEADERS,
        json={"inputs": {"messages": messages}},
        timeout=300,
    )
    if not response.ok:
        raise ValueError(f"Endpoint error {response.status_code}: {response.text}")
    return response.json()["predictions"]


st.title("Research Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history (answers only, no chunks for past turns)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the publications..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_endpoint(st.session_state.messages)

        answer = result["answer"]
        chunks = result.get("chunks", [])

        st.markdown(answer)

        if chunks:
            # Deduplicate paths while preserving order
            seen, unique_paths = set(), []
            for c in chunks:
                p = c["path"]
                if p not in seen:
                    seen.add(p)
                    unique_paths.append(p)

            with st.expander(f"Sources ({len(chunks)} chunks from {len(unique_paths)} file(s))"):
                for i, chunk in enumerate(chunks, 1):
                    filename = PurePosixPath(chunk["path"]).name
                    st.markdown(f"**[{i}] {filename}**")
                    st.caption(chunk["text"].strip())
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
