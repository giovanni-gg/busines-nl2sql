import streamlit as st
from langchain.callbacks.manager import collect_runs
from langchain import memory as lc_memory
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from expression_chain import get_expression_chain
from utils import LLMUtils, GBQUtils, MetadataLoader, Format, LLMTabular2NL
from google.cloud.bigquery.table import RowIterator
import hmac

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


def analyze_response(response, prompt):
    formatted_response = Format.format_llm_response(response, prompt) # format to extract SQL query

    if formatted_response.get('sql_query') == None:
        return "Your question didn't produce any results. Please, try another question."
    else: # RUN SQL QUERY ON DATABASE
        with st.spinner('Getting real-time data from Database...'):
            query_result = gbq.run_query(formatted_response.get('sql_query'))
            st.sidebar.write(query_result)
        
        if "error" in query_result:
            return "SYNTAX ERROR: {}".format(query_result)
        
        else: # didnt produce an error
            if len(query_result) > 500:
                return "The query result is too large to be displayed. Please, try another question."
            else:
                query_results_csv = Format.save_dict_to_string(query_result)
                st.sidebar.write(query_results_csv)
                print(query_results_csv)

                # Parse Tabular Response to NL and return it
                llm_tabular = LLMTabular2NL()
                with st.spinner('Generating Natural Language Response...'):
                    nl_response = llm_tabular.invoke_tabular2sql_chain(user_question=prompt, tabular_response=query_results_csv)

                return nl_response
        

client = Client()
gbq = GBQUtils()

st.set_page_config(
    page_title="DE's Amazon Sales Specialist ",
    page_icon="🦜️️🛠️",
)

# set of llm_messages that will be displayed, to the user - hiding prompt engineering. 
if 'display_messages' not in st.session_state:
    st.session_state.display_messages = []
    welcome_message = {"role": "assistant", "content": "Welcome! Ask me any questions about Danish Endurance's Amazon Orders."}
    st.session_state.display_messages.append(welcome_message)

st.subheader('''🚵‍♂️🔢 Danish Endurance's Amazon Analyst''')
memory = lc_memory.ConversationBufferMemory(
    chat_memory=lc_memory.StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)
with st.sidebar:
    st.markdown('## Technical Validation Testing')
    st.markdown("Reload page to clear the chat history")
    st.divider()

    st.success('Please, leave a feedback after the response')
    st.markdown('''### Easy Copy-Paste Feedback''')
    st.markdown('''#### Use the following feedback options:''')
    st.code('syntax_error')
    st.markdown('''*failed to execute query*''')
    st.code('column_mapping')
    st.markdown('''*maps to wrong column e.g Baselayer -> product_name*''')
    st.code('question_misinterpretation')
    st.markdown('''*e.g "ask  for last month" but gets last 30 days*''')
    st.code('context_misinterpretation')
    st.markdown('''*e.g "ask  for n orders" but gets number of lines*''')
    st.code('faulty_logic')
    st.markdown('''*wrong calculations - e.g YoY, Sum Basket Value*''')
    st.divider()
    st.markdown('''
                ## What Feedback I'm looking for?
                ### Model Accuracy
                - Is the generated SQL query correct?
                - Did the model correctly capture the intent of the question? (Reflect on the quality of the prompt)

                ### UX/UI
                *Fundamently, this task is also an UX/UI design problem*
                - Is the generated answer well formatted?
                - Do you think the model could provide additional information to the user in order to the user understand and pottentially improve the prompt?
                
                ''')

# Add a button to choose between llmchain and expression chain
_DEFAULT_SYSTEM_PROMPT = (
'''
You are an experienced SQL Developer, who is capable of tranlasting natural language into SQL code in Google Big Query Syntax. You job is to understand the Table Schema and user questions and produce a robust SQL query.

Here's the table schema for Amazon Orders (danish-endurance-analytics.nl2sql.amazon_orders):
danish-endurance-analytics.nl2sql.amazon_orders('order_id', 'purchase_date', 'buyer_email', 'market', 'child_asin', 'e_conomic_number', 'product_marketing_category', 'product_name', 'product_pack', 'product_and_pack', 'product_category', 'product_type', 'product_size', 'product_colour', 'gross_sales', 'units_sold')

Guidelines:
- If the questions is not a business question directly related to the table schema, please simply return to the user "Your question didn't produce any results. Please, try another question."
- When generating the SQL query, you must return the SQL query only, without explanation. If the user requires an explanation, they can ask for it.
'''
)
system_prompt = _DEFAULT_SYSTEM_PROMPT
system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")



# Create Chain
chain = get_expression_chain(system_prompt, memory)

feedback_option = (
   "thumbs"
)

# for msg in st.session_state.langchain_messages:
#     avatar = "🦜" if msg.type == "ai" else None
#     st.sidebar.write(msg.type, msg.content)
#     with st.chat_message(msg.type, avatar=avatar):
#         st.markdown(msg.content)

for message in st.session_state.display_messages:
    if message["role"] == "assistant":
        avatar = "🤖"
    else:
        avatar = "🚵‍♂️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


if prompt := st.chat_input(placeholder="Message Danish Endurance's Amazon Analyst ..."):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar='🚵‍♂️').write(prompt)

    with st.chat_message("assistant", avatar = "🤖"):
        message_placeholder = st.empty()
        full_response = ""
        # Define the basic input structure for the chains
        input_dict = {"input": prompt}

        with collect_runs() as cb:
            # for chunk in chain.stream(input_dict, config={"tags": ["Streamlit Chat"]}):
            #     full_response += chunk.content
            #     message_placeholder.markdown(full_response + "▌")
            with st.spinner("Processing Your Question") as status:
                full_response = chain.invoke(input_dict, config={"tags": ["Streamlit Chat"]}).content
            st.session_state.run_id = cb.traced_runs[0].id
        
        query_results_csv = analyze_response(full_response, prompt) # analyze the response to check if the produced a SQL Query or not
                
        # st.markdown(query_results_csv)
        full_response += f"\n{query_results_csv}"
        memory.save_context(input_dict, {"output": full_response})
        message_placeholder.markdown(full_response)
        st.session_state.display_messages.append({"role": "assistant", "content": full_response})


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