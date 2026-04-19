import os
import streamlit as st
import db
from graph import run_rag

st.set_page_config(
    page_title="Gastric-Lipase",
    page_icon=":)",
    layout="centered",
)

if "session_id" not in st.session_state:
    st.session_state.session_id   = db.create_session()
    st.session_state.ui_messages  = []   
 
session_id  = st.session_state.session_id
ui_messages = st.session_state.ui_messages

with st.sidebar:
    st.markdown("## Digest you big fat book")
    st.caption(f"Session `{session_id[:8]}…`")
    st.divider()
 
    st.markdown("**Model:** Qwen2.5 (Ollama)")
    st.markdown("**Embed:** Nomic-embed-text")

    st.divider()
 
    if st.button("New chat", use_container_width=True):
        for k in ("session_id", "ui_messages"):
            st.session_state.pop(k, None)
        st.rerun()
 
    st.caption("New history with each page refresh.")

    import os
    
if not ui_messages:
    ui_messages = db.get_history(session_id)
    st.session_state.ui_messages = ui_messages
    # Debug
    print(f"[app] loaded {len(ui_messages)} history turns for session {session_id[:8]}")

for msg in ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            topic_text = f" · topic: <b>{meta['topic_filter']}</b>" if meta.get("topic_filter") else ""
            reformed   = f" · reformed: <i>{meta['reformed_query']}</i>" if meta.get("reformed_query") else ""
            st.markdown(
                f'<div class="meta-line">'
                f'{topic_text}{reformed}</div>',
                unsafe_allow_html=True,
            )
user_input = st.chat_input("Feed me that doubt📖.")
 
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
 
    db.save_message(session_id, "user", user_input)
    chat_history = db.get_history(session_id)
 
    # Debug
    print(f"[app] passing {len(chat_history)} turns to graph")
    for i, t in enumerate(chat_history):
        print(f"  [{i}] {t['role']}: {t['content'][:80]}")

    with st.chat_message("assistant"):
        with st.spinner("Chewing on that…"):
            result = run_rag(query=user_input, chat_history=chat_history)
 
        answer = result["ans"]
        st.markdown(answer)
 
    db.save_message(session_id, "assistant", answer)
 
    st.session_state.ui_messages.append({"role": "user", "content": user_input})
    st.session_state.ui_messages.append({
        "role":      "assistant",
        "content":   answer,
        "meta": {
            "topic_filter":   result.get("topic_filter"),
            "reformed_query": result["reformed_query"],
        },
    })
 