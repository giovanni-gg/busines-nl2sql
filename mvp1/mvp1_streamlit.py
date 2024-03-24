from openai import OpenAI
import streamlit as st
import test
# from packages.dev_thesis.gbq_utils import GBQUtils
# from packages.dev_thesis.llm_utils import LLMUtils, FrameworkEval

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You response only in French"}]

# add a sidebar
st.sidebar.title("Settings")
st.sidebar.subheader("OpenAI model")
st.sidebar.write(test.sum(1, 2))
# open a text file
import os

# Build an absolute path to the data file
current_dir = os.path.dirname(__file__)
data_file_path = os.path.join(current_dir, 'test.txt')
with open(data_file_path, "r") as file:
    text = file.read()

st.sidebar.write(text)

# llm = LLMUtils()
# gbq = GBQUtils()

# st.sidebar.write(gbq.test_streamlit())

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)
    
    # with st.sidebar:
        # st.write(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response})
