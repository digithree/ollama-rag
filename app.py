import streamlit as st
from streamlit_chat import message
from converse import Converse
from tinydb import TinyDB

db = TinyDB('./config.json')
agent_table = db.table('agent')
agent_table_row = agent_table.all()[0]
user_name = agent_table_row["user_name"]
agent_name = agent_table_row["agent_name"]

st.set_page_config(page_title="Talking with " + agent_name)

def display_messages(showAll: bool=False, plainDisplay: bool=True):
    st.subheader("Chat")
    list = st.session_state["messages"]
    if not showAll and len(list) >= 2:
        list = list[-2:]
    list.reverse()
    for i, (msg, is_user) in enumerate(list):
        if plainDisplay:
            speaker = user_name if is_user else agent_name
            st.markdown(speaker + " said: " + msg)
        else:
            message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        st.session_state["user_input"] = ""
        with st.session_state["thinking_spinner"], st.spinner(f"Simulating conversation..."):
            agent_text = st.session_state["converse"].ask(user_text)
        if not st.session_state["enable_msg_history"]:
            st.session_state["messages"] = []
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def process_check_box(target_key: str, cb_key: str):
    try:
        st.session_state[target_key] = st.session_state[cb_key]
    except KeyError as e:
        print(e)

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["converse"] = Converse()
        st.session_state["enable_msg_history"] = False
        st.session_state["cb_msg_history"] = False

    st.text_input("Message", key="user_input", on_change=process_input)

    st.checkbox("Show all message history", value=st.session_state["enable_msg_history"], key="cb_msg_history", on_change=process_check_box("enable_msg_history", "cb_msg_history"))

    display_messages(st.session_state["enable_msg_history"])

if __name__ == "__main__":
    page()