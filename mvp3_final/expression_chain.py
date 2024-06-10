from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable, RunnableMap
from langchain.memory import ConversationBufferMemory
from utils import Format
import os 

current_dir = os.path.dirname(__file__)

def get_expression_chain(
    system_prompt: str, memory: ConversationBufferMemory
) -> Runnable:
    """Return a chain defined primarily in LangChain Expression Language"""
    ingress = RunnableMap(
        {
            "input": lambda x: x[0]["input"],
            "classifier_guidelines": lambda x: x[1]["classifier_guidelines"],
            "chat_history": lambda x: memory.load_memory_variables(x)["chat_history"],
            # "time": lambda _: str(datetime.now()),
        }
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="chat_history"),

            ("human", load_template_from_file(os.path.join(current_dir, "templates/chain_w_classifier/prompt/tf-sql_prompt.txt")))
        ]
    )
    llm = ChatOpenAI(temperature=0, model='gpt-4-turbo-preview')
    chain = ingress | prompt | llm
    return chain

def load_template_from_file(file_path):
    with open(file_path, 'r') as file:
        template_str = file.read()
    return template_str

if __name__ == "__main__":
    chain, _ = get_expression_chain()
    # memory = ConversationBufferMemory()
    # system_prompt = "Your system prompt here"
    # chain = get_expression_chain(system_prompt, memory)
    # input_data = {
    #     "input": "What's your name?",
    #     "classifier_guidelines": "Ensure the query is optimized for speed and includes proper indexing."
    # }
    # for chunk in chain.stream(input_data):
    #     print(chunk.content, end="", flush=True)
    pass