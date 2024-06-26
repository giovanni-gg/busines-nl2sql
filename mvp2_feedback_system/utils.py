# Langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate

# path ans os
from pathlib import Path
import os
import csv
from datetime import datetime, date
import json
from io import StringIO

#openai 
from openai import OpenAI

# gbq
from google.cloud.bigquery.table import RowIterator
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import GoogleCloudError, BadRequest

# streamlit
import streamlit as st
current_dir = os.path.dirname(__file__)

class LLMUtils:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        pass

    def get_client(self):
        return self.client

    def invoke_simple_chain(self, messages:list, model:str = None):
        '''
            Invoke the simple chain using simple_chain.txt template
        '''
        # Load the Templates
        sys_message = self.load_template_from_file(os.path.join(current_dir, 'templates/generic_sys_message.txt'))

        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'templates/simple_chain.txt'))

        table_schema = MetadataLoader.get_all_metadata(os.path.join(current_dir, 'templates/table_schema_full.json'))


        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", sys_message),
        #         ("user", prompt_template)
        #     ]
        # )

            
        return chain.invoke({"user_question": user_question, "table_schema": table_schema})
    
    def invoke_simple_chain_vanilla(self, model:str = None, user_question: str = None):
        sys_message = self.load_template_from_file(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\prompt_template\generic_sys_message.txt')

        prompt_template = self.load_template_from_file(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\prompt_template\simple_chain.txt')

        table_schema = MetadataLoader.get_all_metadata(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\data.json')

        # Messages
        messages=[
        {"role": "system", "content": sys_message},
        {"role": "user", "content": Format.get_prompt_simple(user_question, prompt_template, table_schema)},
        ]
        
        # API call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=messages
        )

        # Response Handling -> Dictionary with SQL query and other metadata
        formatted_response = Format.format_llm_response(response, user_question)

        formatted_response['user_question'] = user_question

        # Run SQL Query on the database
        if formatted_response.get('sql_query') != None:
            print(formatted_response.get('sql_query'))
            gbq = GBQUtils()
            query_result = gbq.run_query(formatted_response.get('sql_query'))
        else:
            query_result = None

        # Log the response
        FrameworkEval.log_response(formatted_response, iterator = query_result)

        return formatted_response

    def load_template_from_file(self, file_path):
        with open(file_path, 'r') as file:
            template_str = file.read()
        return template_str

class LLMTabular2NL(LLMUtils):

    def __init__(self):
        super().__init__()
        print(current_dir)
        pass

    def invoke_tabular2sql_chain(self, user_question: str, tabular_response: str):
        print("Invoking Tabular 2 sql Chain")
        
        sys_message = self.load_template_from_file(os.path.join(current_dir, 'templates/tabular2nl/tabular2sql_sysMessage.txt')) # Load the sys message

        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'templates/tabular2nl/tabular2SQL_promptTemplate.txt'))

        # Messages
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": Format.get_prompt_tabular2sql(user_question=user_question, prompt_template = prompt_template, tabular_response=tabular_response)},
        ]

        # API call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=messages
        )

        print(messages)

        response = response.choices[0].message.content

        # formatted_response['user_question'] = user_question
        print(response)

        return response
    
class Format:
    def __init__(self):
        pass

    @staticmethod
    def get_prompt_simple(user_question, prompt_template, table_schema):
        '''
            Get the formatted prompt for the simple chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question', 'table_schema'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
            table_schema= table_schema
        )
        return prompt_formatted

    @staticmethod
    def get_prompt_tabular2sql(user_question, prompt_template, tabular_response):
        '''
            Get the formatted prompt for the simple chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question', 'tabular_response'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
            tabular_response= tabular_response
        )
        return prompt_formatted
    
    @staticmethod
    def format_llm_response(response, user_question) -> dict:
        '''
            Format the response from the LLM
        '''
        import re
        pattern = r"```sql\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
        else:
            sql_query = None
        
        # Create a dictionary to hold the response and the SQL query
        formatted_response = {
           "sql_query": sql_query,
            "user_question": user_question
        }
        
        return formatted_response
    
    @staticmethod
    def save_row_iterator_to_csv_string(iterator):
        """
        Saves the rows returned by a BigQuery query job to a CSV file.
        
        Parameters:
        - query_job: The query job object from BigQuery.
        - filename: The name of the file to save the CSV data.
        """

        if isinstance(iterator, RowIterator):
                # Create a StringIO object to hold the CSV data
            output = StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow([field.name for field in iterator.schema])

            # Write data rows
            for row in iterator:
                writer.writerow(row.values())

            # Get the CSV string from the StringIO object
            csv_string = output.getvalue()

            return csv_string
        else:
            return "No data returned."
    @staticmethod 
    def custom_serializer(obj):
        if isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()  # or format it as obj.strftime('%Y-%m-%dT%H:%M:%S')
        raise TypeError(f"Type {type(obj)} not serializable")

    @staticmethod
    def save_dict_to_string(dict):
        '''
            Save a dictionary to a string
        '''
        return json.dumps(dict, default=Format.custom_serializer)
    
    def save_row_iterator_to_csv_string_new(iterator):
        """
        Saves the rows returned by a BigQuery query job to a CSV string.
        
        Parameters:
        - iterator: The RowIterator object from BigQuery.
        """

        if not isinstance(iterator, RowIterator):
            return "No data returned."

        # Capture schema from the RowIterator before reading data
        schema = iterator.schema

        # Store data in a list to prevent multiple iterations over the iterator
        data = list(iterator)

        # Create a StringIO object to hold the CSV data
        output = StringIO()
        writer = csv.writer(output)

        # Write header using the schema from the RowIterator
        writer.writerow([field.name for field in schema])

        # Write data rows
        for row in data:
            writer.writerow(row.values())

        # Get the CSV string from the StringIO object
        csv_string = output.getvalue()
        output.close()  # Close the StringIO object to free up resources

        return csv_string

        
    
class MetadataLoader:
    def __init__(self):
        pass

    @staticmethod
    def get_all_metadata(path_to_file):
        '''
            Get the metadta for a table from a json file
        '''
        # if it's JSON file load it
        if path_to_file.endswith('.json'):
            with open(path_to_file, 'r') as file:
                file_content = json.load(file)
            return file_content
        if path_to_file.endswith('.txt'):
            with open(path_to_file, 'r') as file:
                file_content = file.read()
            return file_content
    

    
class FrameworkEval:
    def __init__(self):
        pass

    @staticmethod
    def run_eval(models:list, chains:list, questions:list, hyper_parameters:dict = None):
        '''
            Run the evaluation of the framework
        '''
        for model in models:
            for chain in chains:
                print(f"RUNNIG EVALUATION FOR MODEL: {model} AND CHAIN: {chain}")
                print("=====================================================")
                for question in questions:
                    print(f"\t Question: {question['question']}")
                    llm = LLMUtils()
                    if chain == 'simple_chain':
                        llm.invoke_simple_chain_vanilla(model = model, user_question=question['question'])
                    else:
                        continue

    @staticmethod
    def log_response(formatted_response: dict, iterator):
        '''
            Log the response from the LLM
        '''
        # Define your file path
        file_path = r"G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\logging\login_response.json"

        # Check if the file exists and read its content if it does
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        query_results_file_path = f"G:/My Drive/Profissional & Acadêmico/Mestrados/DTU/5_thesis/dev_thesis/data/de_data/logging/logging_results/{formatted_response.get('id')}.csv"
        # Save Result to CSV
        FrameworkEval().save_row_iterator_to_csv(iterator, query_results_file_path)
        # add file path to the formatted response
        formatted_response['query_results_file_path'] = query_results_file_path

        # Append the new response to the data array
        data.append(formatted_response)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print("Response logged successfully.")
    
    @staticmethod
    def save_row_iterator_to_csv(iterator, filename):
        """
        Saves the rows returned by a BigQuery query job to a CSV file.
        
        Parameters:
        - query_job: The query job object from BigQuery.
        - filename: The name of the file to save the CSV data.
        """        
        with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write headers based on the schema of the query result
            headers = [field.name for field in iterator.schema]
            csv_writer.writerow(headers)
            
            # Write each row from the iterator
            for row in iterator:
                csv_writer.writerow(row.values())


class GBQUtils:
    def __init__(self):
        service_account_info_str = st.secrets["GBQ_SERVICE_ACCOUNT_INFO"]
        service_account_info = json.loads(service_account_info_str)

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
        )
        self.client = bigquery.Client(credentials=credentials)
        
    def get_client(self):
        return self.client

    def get_table_schema(self, table_id):
        if "`" in table_id:
            table_id = table_id.replace("`", "")
        table_ref = self.client.get_table(table_id)
        schema_str = ""
        for field in table_ref.schema:
            schema_str += f"{field.name} ({field.field_type}), "
        return schema_str[:-1]
    
    def format_distinct_values(self, query_job_result):

        ''' Format the results of a query job to a string accepts only one column in the result set'''

        column_name = query_job_result.schema[0].name

        # Initialize an empty string to store the results
        result_string = ""

        # Iterate over each row in the results
        for row in query_job_result:
            # Dynamically access the column value using the column_name
            column_value = getattr(row, column_name)
            # Append the column value to the result string with a newline character
            result_string += f"{column_value},"

        return result_string

    
    def get_distinct_values(self, table_id, fetch_all: list, fetch_partial: list):
        ''' Get the distinct values for the columns in a table'''

        table_ref = self.client.get_table(table_id)


        distinct_values = {} # Initialize an empty dictionary
        distinct_values['table_id'] = table_id
        distinct_values['table_description'] = table_ref.description
        distinct_values['fields'] = [] 

        for field in table_ref.schema:
            if field.name in fetch_all:
                query = f"SELECT DISTINCT {field.name} FROM {table_id} WHERE {field.name} IS NOT NULL"

            elif field.name in fetch_partial:
                query = f"SELECT DISTINCT {field.name} FROM {table_id} WHERE {field.name} IS NOT NULL LIMIT 20"
            else:
                continue

            query_job_result = self.client.query(query).result() # perform the query

            distinct_values['fields'].append({
                        "field_name": field.name,
                        "field_description": None,
                        "field_dataType": field.field_type,
                        "distinct_values": self.format_distinct_values(query_job_result)})

        return distinct_values
    
    @st.cache_data(ttl=600)
    def run_query(_self, query):
        try:
            query_job = _self.client.query(query)
            results = query_job.result()
            print(f"Query executed successfully. {results.total_rows} rows returned.")
            rows = [dict(row) for row in results] 
            return rows
        except BadRequest as error:
            # BadRequest errors provide more detailed information about query issues
            # Attempting to parse error details for more specific feedback
            err_msg = "A query error occurred: "
            if hasattr(error, 'errors') and error.errors:
                for e in error.errors:
                    err_msg += f"{e['message']} "
                    if 'location' in e:
                        err_msg += f"at location {e['location']}. "
            return err_msg
        except GoogleCloudError as error:
            return f"A Google Cloud error occurred: {error}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
    




                    

