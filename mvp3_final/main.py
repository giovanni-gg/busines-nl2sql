import streamlit as st

st.set_page_config(
    page_title="DE's Amazon Sales Specialist ",
    page_icon="🦜️️🛠️",
)

from langchain.callbacks.manager import collect_runs
from langchain import memory as lc_memory
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from expression_chain import get_expression_chain
from utils import LLMUtils, GBQUtils, MetadataLoader, Format, LLMTabular2NL, get_lookups, Streamlit
from google.cloud.bigquery.table import RowIterator
import hmac
import json
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(__file__)


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

client = Client()
gbq = GBQUtils()
llm = LLMUtils()
streamlit_utils = Streamlit()
llm_tabular = LLMTabular2NL()


SCHEMA_LINKING_LOOKUP, GROWTH_METRICS_LOOKUP, FINANCIAL_METRICS_LOOKUP, DATERANGE_METRICS_LOOKUP = get_lookups()

def analyze_response(response, prompt):
    formatted_response = Format.format_llm_response(response, prompt) # format to extract SQL query

    if formatted_response.get('sql_query') == None:
        return "Your question didn't produce any results. Please, try another question."
    else: # RUN SQL QUERY ON DATABASE
        with st.spinner('Getting real-time data from Database...'):
            query_result = gbq.run_query(formatted_response.get('sql_query'))
            # st.sidebar.write(query_result)
        
        if "error" in query_result:
            return "SYNTAX ERROR: {}".format(query_result)
        
        else: # didnt produce an error
            if len(query_result) > 50:
                return "", query_result
            else:
                query_results_csv = Format.save_dict_to_string(query_result)
                # st.sidebar.write(query_results_csv)
                print(query_results_csv)

                # Parse Tabular Response to NL and return it
                with st.spinner('Generating Natural Language Response...'):
                    nl_response = llm_tabular.invoke_tabular2sql_chain(user_question=prompt, tabular_response=query_results_csv)

                return nl_response, query_result
        

# set of llm_messages that will be displayed, to the user - hiding prompt engineering. 
if 'display_messages' not in st.session_state:
    st.session_state.display_messages = []
    welcome_message = {"role": "assistant", "content": "Welcome! Ask me almost 😁 any questions about Danish Endurance's Amazon Orders."}
    st.session_state.display_messages.append(welcome_message)

if 'memory_messages_classifier' not in st.session_state:
    st.session_state.memory_messages_classifier = []


st.subheader('''🚵‍♂️🔢 Danish Endurance's GPT''')
memory = lc_memory.ConversationBufferMemory(
    chat_memory=lc_memory.StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)
with st.sidebar:
    st.markdown('## Technical Validation Testing')
    st.markdown("Reload page to clear the chat history")
    st.divider()

    # st.success('Please, leave a feedback after the response')
    # st.markdown('''### Easy Copy-Paste Feedback''')
    # st.markdown('''#### Use the following feedback options:''')
    # st.code('syntax_error')
    # st.markdown('''*failed to execute query*''')
    # st.code('column_mapping')
    # st.markdown('''*maps to wrong column e.g Baselayer -> product_name*''')
    # st.code('question_misinterpretation')
    # st.markdown('''*e.g "ask  for last month" but gets last 30 days*''')
    # st.code('context_misinterpretation')
    # st.markdown('''*e.g "ask  for n orders" but gets number of lines*''')
    # st.code('faulty_logic')
    # st.markdown('''*wrong calculations - e.g YoY, Sum Basket Value*''')
    # st.divider()
    # st.markdown('''
    #             ## What Feedback I'm looking for?
    #             ### Model Accuracy
    #             - Is the generated SQL query correct?
    #             - Did the model correctly capture the intent of the question? (Reflect on the quality of the prompt)

    #             ### UX/UI
    #             *Fundamently, this task is also an UX/UI design problem*
    #             - Is the generated answer well formatted?
    #             - Do you think the model could provide additional information to the user in order to the user understand and pottentially improve the prompt?
    #             ''')

# Add a button to choose between llmchain and expression chain
_DEFAULT_SYSTEM_PROMPT = llm.load_template_from_file(os.path.join(current_dir,"templates/chain_w_classifier/prompt/tf-sql_sysmessage.txt" ))
system_prompt = _DEFAULT_SYSTEM_PROMPT
system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")



# Create Chain
chain = get_expression_chain(system_prompt, memory)

feedback_option = (
   "thumbs"
)

for message in st.session_state.display_messages:
    if message["role"] == "assistant":
        avatar = "🤖"
    else:
        avatar = "🚵‍♂️"
    with st.chat_message(message["role"], avatar=avatar):
        streamlit_utils.get_status_elements(message["content"])
        st.markdown(message["content"])


if prompt := st.chat_input(placeholder="Message Danish Endurance's Amazon Analyst ..."):
    st.session_state.display_messages.append({"role": "user", "content": prompt}) # display message for user
    st.session_state.memory_messages_classifier.append({"role": "user", "content": prompt}) 
    # save context

    st.chat_message("user", avatar='🚵‍♂️').write(prompt)

    # create display message for assistant
    st.session_state.display_messages.append({"role": "assistant", "content": ""})

    with st.chat_message("assistant", avatar = "🤖"):
        memory_query_generator = ""

        with st.spinner("Analysing your question..."):
            answerable, reasons_found, reasons_not_found, classifier_guidelines, question_classfied = llm.invoke_openAI_w_classifier_vanilla(st.session_state.memory_messages_classifier)

            st.session_state.memory_messages_classifier.append({"role": "assistant", "content": json.dumps(question_classfied)})
        
        formatted_reasons = streamlit_utils.format_reasons(reasons_found, reasons_not_found, answerable)
        
        # with st.sidebar:
        #     st.markdown('## Query Classification')
        #     st.markdown(f'### Is the question answerable? {answerable}')
        #     st.markdown(f'### Reasons: {reasons_found}')
        #     st.markdown(f'### Reasons nto found: {reasons_not_found}')
        #     st.markdown('### Classification Results:')
        #     st.markdown(f'### classifierguidelines: {classifier_guidelines}')
        #     st.json(json.dumps(question_classfied))
        #     st.divider()

        if answerable:

            st.info('Please, check if we correctly captured your question intent', icon="ℹ️")

            st.session_state.display_messages[-1]['content'] += formatted_reasons
            st.markdown(formatted_reasons) # display reason as soon as it's generated before the query is processed
            # Define the basic input structure for the chains
            input_dict = [{"input": prompt}, {'classifier_guidelines': classifier_guidelines}]
            with collect_runs() as cb:
                with st.spinner("Processing Your Question") as status:
                    generated_query = chain.invoke(input_dict, config={"tags": ["Streamlit Chat"]}).content
                    # st.session_state.display_messages[-1]['content'] += generated_query
                    # st.markdown(generated_query)
                st.session_state.run_id = cb.traced_runs[0].id
            
            nl_response, query_result_dict = analyze_response(generated_query, prompt) # analyze the response to check if the produced a SQL Query or not
            
            # LLM context
            memory_query_generator += f"\n{nl_response}"
            memory.save_context(input_dict[0], {"output": memory_query_generator})
            
            # Display nl response
            st.session_state.display_messages[-1]['content'] += nl_response
            st.markdown(nl_response)
            
            st.dataframe(pd.DataFrame(query_result_dict))
            st.session_state.display_messages.append({"role": "assistant", "content": memory_query_generator})
        else:
            # st.warning("Our systems couldn't answer your question. Please modify it:", icon="⚠️")
            streamlit_utils.get_status_elements(formatted_reasons)
            st.markdown(formatted_reasons)
            st.session_state.display_messages.append({"role": "assistant", "content": "I'm sorry, I can't answer this question. Please, ask another question."})

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="syntax_error, column_mapping, question_misinterpretation, context_error, faulty_logic",
        key=f"feedback_{run_id}",
    )

    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0, "🔢":2},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            # Formulate feedback type string incorporating the feedback option
            # and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string
            # and optional comment
            feedback_record = client.create_feedback(
                run_id,
                feedback_type_str,
                score=score,
                comment=feedback.get("text"),
            )
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            st.warning("Invalid feedback score.")