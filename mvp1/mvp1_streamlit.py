from openai import OpenAI
import streamlit as st
import test
import os
from utils import LLMUtils, GBQUtils, MetadataLoader, Format
from langchain_openai import ChatOpenAI

# Initialize the ChatOpenAI model
model = ChatOpenAI()

# path handling
current_dir = os.path.dirname(__file__)

st.title("Danish Endurance - Amazon Orders & Sales Analyst")
with st.sidebar:
    st.markdown("# Prompt Guidelines")
    st.markdown('''
    1. Specify the product category with the max detail possible. For example, mentioning Marketing Category/ Product Type/ Product Name.
                
    ✅ Good Prompt: "What's the yoy sales for the marketing category "Baselayer"?
                
    ❌ Bad Prompt: "Sales for product Baselayer" - **Not specified if you want marketing category, or product type**
                ''')

llm = LLMUtils()
client_llm = LLMUtils().get_client()
gbq = GBQUtils()

# clint_gbq.get_client()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "sys_messages" not in st.session_state:
    sql_specialist = 'You are an experienced SQL Developer, who is capable of tranlasting natural language into SQL code in Google Big Query Syntax. You job is to understand the Table Schema and user questions and produce a robust SQL query.'

    tabular2sql_translator = '''You're professional data communicator, that takes in an User Question and tabular response formated as csv, and parse the response to the user in simple and objecitve Natural Language. All the monetery rsponses should be in Euros'''

    st.session_state["sys_messages"] = {
                "sql_specialist": sql_specialist,
                "tabular2sql_translator": tabular2sql_translator 
            }

if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = [{"role": "system", "content": st.session_state.sys_messages.get("sql_specialist")}]

# set of llm_messages that will be displayed, to the user - hiding prompt engineering. 
if 'display_messages' not in st.session_state:
    st.session_state.display_messages = []


for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if original_prompt := st.chat_input("What is up?"):    
    if len(st.session_state.llm_messages) == 1: # First User message
        prompt_template = llm.load_template_from_file(os.path.join(current_dir, 'templates/openAI_baseline/openAIBaseline_prompt.txt'))

        table_schema = MetadataLoader.get_all_metadata(os.path.join(current_dir, 'templates/openAI_baseline/openAiBaseline_table_schema.txt'))

        prompt_formated = Format.get_prompt_simple(user_question=original_prompt, prompt_template=prompt_template, table_schema=table_schema)
    else: # not first message
        st.sidebar.write("Not first message")
        prompt_formated = original_prompt
        st.sidebar.write(prompt_formated)

    st.session_state.display_messages.append({"role": "user", "content": original_prompt})
    st.session_state.llm_messages.append({"role": "user", "content": prompt_formated})

    with st.chat_message("user"):
        st.markdown(original_prompt)

    with st.chat_message("assistant"):
        
        ## GET SQL RESPONSE
        response = model.invoke(st.session_state.llm_messages).content
        st.session_state.llm_messages.append({"role": "assistant", "content": response}) # append the response
        formatted_response = Format.format_llm_response(response, original_prompt) # format to extract SQL query

        # RUN SQL QUERY ON DATABASE
        if formatted_response.get('sql_query') != None:
            query_result = gbq.run_query(formatted_response.get('sql_query'))
            
            # FORMAT PROMPT FOR TABULAR TO SQL
            prompt_template_t2nl = llm.load_template_from_file(os.path.join(current_dir, 'templates/tabular2nl/tabular2SQL_promptTemplate.txt'))

            prompt_message_t2nl = Format.get_prompt_tabular2sql(user_question=original_prompt, prompt_template=prompt_template_t2nl, tabular_response=Format.save_row_iterator_to_csv_string(query_result))

            st.session_state.llm_messages.append({"role": "user", "content": prompt_message_t2nl})

            response = model.invoke(st.session_state.llm_messages).content
            st.session_state.llm_messages.append({"role": "assistant", "content": response}) # append the response

            # st.session_state.llm_messages.append() 
        else:
            query_result = None


        st.markdown(response)


    st.session_state.display_messages.append({"role": "assistant", "content": response})
