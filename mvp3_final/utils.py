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


class CaseInsensitiveDict:
    def __init__(self, original_dict):
        self.original_dict = original_dict
        self.lowercase_mapping = {str(k).lower(): k for k in original_dict}
    
    def get_original_key(self, key):
        lower_key = key.lower()
        print("lower_key", lower_key)
        return self.lowercase_mapping.get(lower_key, None)
    
    def items(self):
        return self.original_dict.items()

@st.cache_data(show_spinner=True)
def get_lookups():
    '''
    Get the lookup data for the Streamlit app.
    '''
    current_dir = os.path.dirname(__file__)  # Ensure current_dir is defined
    lookup_files = {
        'fields_look_up': 'templates/chain_w_classifier/lookups/fields_look_up.json',
        'growth_metrics': 'templates/chain_w_classifier/lookups/growth_metrics.json',
        'financial_metrics': 'templates/chain_w_classifier/lookups/financial_metrics.json',
        'date_range': 'templates/chain_w_classifier/lookups/date_range.json'
    }
    
    with open(os.path.join(current_dir, lookup_files['fields_look_up']), 'r') as f:
        data = json.load(f)

    # Initialize the output dictionary
    schema_linking_lookup = {}

    # Loop through each field in the 'fields' list
    for field in data['fields']:
        field_name = field['field_name']
        # Map each distinct value to the field_name
        for value in field['distinct_values']:
            schema_linking_lookup[value] = field_name
    
    schema_linking_lookup_case_insensitive = CaseInsensitiveDict(schema_linking_lookup)

    with open(os.path.join(current_dir, lookup_files['growth_metrics']), 'r') as f:
        growth_metrics_lookup = json.load(f)
    
    growth_metrics_lookup_case_insensitive = CaseInsensitiveDict(growth_metrics_lookup)

    with open(os.path.join(current_dir, lookup_files['financial_metrics']), 'r') as f:
        financial_metrics_lookup = json.load(f)
    
    financial_metrics_lookup_case_insensitive = CaseInsensitiveDict(financial_metrics_lookup)

    with open(os.path.join(current_dir, lookup_files['date_range']), 'r') as f:
        date_range_lookup = json.load(f)
    
    date_range_lookup_case_insensitive = CaseInsensitiveDict(date_range_lookup)
    
    return schema_linking_lookup_case_insensitive, growth_metrics_lookup_case_insensitive, financial_metrics_lookup_case_insensitive, date_range_lookup_case_insensitive

# Retrieve the case-insensitive lookup dictionaries
SCHEMA_LINKING_LOOKUP, GROWTH_METRICS_LOOKUP, FINANCIAL_METRICS_LOOKUP, DATERANGE_METRICS_LOOKUP = get_lookups()

class LLMUtils:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        pass

    def get_client(self):
        return self.client

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

    def invoke_openAI_w_classifier_vanilla(self, memory_messages_classifier: list, model = 'gpt-4-turbo'):
        '''
        Invoke openai vanilla chain with classfiier
        - Classify the question
        - Format the Prompt and pass it into the normal flow of the openAI chain
        '''
        ## Classify the question ##
        llm_classifier = LLMQueryClassification()
        question_classfied = llm_classifier.invoke_query_classification_chain(memory_context=memory_messages_classifier, model=model)

        # process the classification
        classification_results = llm_classifier.process_classification(question_classfied)

        # define if it's anserable
        answerable, reasons_found, reasons_not_found = llm_classifier.decide_if_question_is_answerable(classification_results)
         
        if answerable:
            classifier_guidelines = Format.get_classification_guidelines(classification_results)
        
        else:
            classifier_guidelines = f"Question is not answerable due to the following reasons: {reasons_not_found}"

        return answerable, reasons_found, reasons_not_found, classifier_guidelines, question_classfied

        #     ## Run the openAI chain
        #     sys_message = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/6_TF-SQL/tf-sql_sysmessage.txt'))

        #     prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/6_TF-SQL/tf-sql_prompt.txt'))

        #     final_prompt = Format.get_prompt_simple_with_classifier(user_question = user_question, prompt_template = prompt_template, classifier_guidelines=classifier_guidelines)


        return formatted_response, response

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
class LLMQueryClassification(LLMUtils):
    def __init__(self):
        super().__init__()
        pass

    def invoke_query_classification_chain(self, memory_context:list, model = 'gpt-4-turbo'):
        print("##Invoking Query Classification Chain##")


        if len(memory_context) == 1: # first message

            sys_message_template = self.load_template_from_file(os.path.join(current_dir, 'templates/chain_w_classifier/prompt/qc_dynamic_system_message.txt'))

            sys_message = Format.get_sysmessage_classification(prompt_template=sys_message_template)

            prompt_template = self.load_template_from_file(os.path.join(current_dir, 'templates/chain_w_classifier/prompt/qc_prompt_template.txt'))

            prompt =  Format.get_prompt_classification(user_question=memory_context[-1].get('content'), prompt_template=prompt_template)

            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": prompt},
            ]
        else: # second message
            messages = memory_context
            # messages.insert(0, {"role": "user", "content": sys_message})

        # API call
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
            response_format={ "type": "json_object" }
        )

        st.session_state.memory_messages_classifier = messages

        return json.loads(response.choices[0].message.content)

    def process_classification(self, classification):
        '''
        Process classification for allowed and not_allowed entries and returns their status and details from the lookup.

        Parameters:
        - classification: Dict with 'allowed' and 'not_allowed' keys, each containing categories like financial_metrics, etc.
        
        Returns:
        - Dictionary of classification with status and additional details for metrics.
        '''

        def process_category(items, lookup):
            results = []
            look_up_lower = lookup.lowercase_mapping
            look_up_original = lookup.original_dict

            for item in items:
                if item.lower() in look_up_lower:
                    original_item = lookup.get_original_key(item.lower())
                    item_details = look_up_original[original_item]
                    if lookup is SCHEMA_LINKING_LOOKUP:
                        results.append({original_item: {'found': 1, 'field_name': item_details}})
                        continue
                    else:
                        results.append({
                            item: {
                                'found': 1,
                                'description': item_details.get('description', 'No description available'),
                                'calculation_guidelines': item_details.get('calculation_guidelines', 'No guidelines available'),
                                'calculation_example': item_details.get('calculation_example', 'No example available')
                            }
                        })
                else:
                    results.append({item: {'found': 0}})
            return results

        results = {}

        # Loop over both 'allowed' and 'not_allowed'
        for status in ['allowed', 'not_allowed']:
            results[status] = {}

            for category, lookup in [
                ('products', SCHEMA_LINKING_LOOKUP),
                ('countries_alpha_2_code', SCHEMA_LINKING_LOOKUP),
                ('financial_metrics', FINANCIAL_METRICS_LOOKUP),
                ('growth_metrics', GROWTH_METRICS_LOOKUP),
                ('date_range', DATERANGE_METRICS_LOOKUP)
            ]:
                items = classification.get(status, {}).get(category, [])
                # print(items)  
                results[status][category] = process_category(items, lookup)

        return results

    def decide_if_question_is_answerable(self, classification: dict):
        '''
        Decide if the question is answerable based on the classification.

        - Returns false if the question contains the following:
            - Not Found Products;
            - Not Found Financial Metrics;
            - Not Found Growth Metrics;
            - Not Found Date Range Metrics;
            - Any Metric on the not allowed key.
        - Returns both FOUND and NOT FOUND metrics in a structured format.
        '''

        def collect_elements(category):
            found_elements = []
            not_found_elements = []
            for items in category:
                for metric, details in items.items():
                    if details['found'] == 0:
                        not_found_elements.append(metric)
                    else:
                        found_elements.append(metric)
            return found_elements, not_found_elements

        reasons_not_found = {}
        reasons_found = {}

        # Check allowed metrics
        for key in ['products', 'financial_metrics', 'growth_metrics', 'date_range', 'countries_alpha_2_code']:
            if key in classification.get('allowed', {}):
                found, not_found = collect_elements(classification['allowed'][key])
                if found:
                    reasons_found[key] = found
                if not_found:
                    reasons_not_found[key] = not_found

        # Check not allowed metrics
        for key in ['products', 'financial_metrics', 'growth_metrics', 'date_range']:
            if key in classification.get('not_allowed', {}):
                _, not_allowed_metrics = collect_elements(classification['not_allowed'][key])
                if not_allowed_metrics or classification['not_allowed'][key]:
                    if key not in reasons_not_found:
                        reasons_not_found[key] = not_allowed_metrics
                    else:
                        reasons_not_found[key].extend(not_allowed_metrics)

        if reasons_not_found:
            return False, reasons_found, reasons_not_found
        elif reasons_found == {}:
            return False, reasons_found, reasons_not_found

        return True, reasons_found, None
    
class Format:
    def __init__(self):
        pass

    @staticmethod
    def get_sysmessage_classification(prompt_template):
        '''
            Get the formatted prompt for the classification chain
        '''
        from datetime import datetime
        today_date = datetime.today().strftime('%A %d %B %Y')
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['dynamic_system_message'],
            template=prompt_template
        )

        dynamic_system_message = ""

        dynamic_system_message += "\n## date_range: Identify and extract the time period mentioned in the question. This can include the two categories below:\n"
        for key, value in DATERANGE_METRICS_LOOKUP.items():
            dynamic_system_message += f"- {key}: {value['description']}\n"
        
        dynamic_system_message += "\n## financial_metrics: Identify and list any financial metrics mentioned in the question. This can include the metrics below:\n"
        for key, value in FINANCIAL_METRICS_LOOKUP.items():
            dynamic_system_message += f"- {key}: {value['description']}\n"

        dynamic_system_message += "\n## growth_metrics: Indicators that measure the growth rate of various aspects of the business, which are period-over-period calculations. This can include the metrics below:\n"
        for key, value in GROWTH_METRICS_LOOKUP.items():
            dynamic_system_message += f"- {key}: {value['description']}\n"

        prompt_formatted = prompt.format(dynamic_system_message=dynamic_system_message)
        return prompt_formatted

    @staticmethod
    def get_prompt_classification(prompt_template, user_question):
        '''
            Get the formatted prompt for the classification chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
        )
        return prompt_formatted

    @staticmethod
    def get_classification_guidelines(classification:dict):
        '''
            Get the classification guidelines for the classification
        '''
        # Parse the input JSON string to a dictionary
        metrics_info = classification
        
        # Start forming the response string
        response = ""
        
        # Process products under the "allowed" category
        if len(metrics_info['allowed']['products']) > 0:
            response += "###Schema Linking Instructions:###\n"
            for product in metrics_info['allowed']['products']:
                for product_name, details in product.items():
                    response += f"- {product_name} maps to the column {details['field_name']}\n"
        
        # Add growth metrics section
        if len(metrics_info['allowed']['growth_metrics']) > 0:
            response += "###Growth Metrics Instructions###\n"
            for metric in metrics_info['allowed']['growth_metrics']:
                for metric_name, details in metric.items():
                    if details['found'] == 1:
                        response += f"{metric_name}:\n{details['calculation_guidelines']}\n {details['calculation_example']}\n"
        
        # Add date_range section
        if len(metrics_info['allowed']['date_range']) > 0:
            response += "###Date Interpretation Instructions:###\n"
            for metric in metrics_info['allowed']['date_range']:
                for metric_name, details in metric.items():
                    if details['found'] == 1:
                        response += f"{metric_name}:\n{details['calculation_guidelines']}\n{details['calculation_example']}\n"
        
                # Add date_range section
        if len(metrics_info['allowed']['financial_metrics']) > 0:
            response += "\n### Financial Metrics instructions###:\n"
            for metric in metrics_info['allowed']['financial_metrics']:
                for metric_name, details in metric.items():
                    if details['found'] == 1:
                        response += f"{metric_name}: \n {details['calculation_guidelines']}\n"
        
        return response

    @staticmethod
    def get_prompt_simple_with_classifier(user_question, classifier_guidelines, prompt_template):
        '''
            Get the formatted prompt for the simple chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question', 'classifier_guidelines'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
            classifier_guidelines = classifier_guidelines
        )
        # append to final prompt the current day including the day of the week
        prompt_formatted += f"\nToday is {datetime.today().strftime('%A %d %B %Y')}"

        return prompt_formatted

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
    
    @st.cache_data(ttl=600, show_spinner=False)
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

class Streamlit:
    def format_reasons(self, reasons_found, reasons_not_found, answerable):
        '''
        Format the reasons for the classification into natural language.
        
        - If answerable is True, returns only the metrics found.
        - If answerable is False, returns the metrics not found.
        '''
        if answerable:
            formatted_reasons = "##### Our system classified your question as:\n"
            # Add found metrics
            for key, metrics in reasons_found.items():
                formatted_reasons += f"- **{key.replace('_', ' ').title()}**: `{', '.join(metrics)}`\n"
        else:
            if reasons_found == {} and reasons_not_found == {}: # No metrics found
                formatted_reasons = "Sorry, but it looks like your questions is not related to any of the metrics we have in our system. Please, try again with a different question."
            else:
                formatted_reasons = "#### Sorry, we coudn't interpret your question: \n"
                formatted_reasons += "Your question contains the following metrics: \n"
                # Add not found metrics
                for key, metrics in reasons_not_found.items():
                    formatted_reasons += f"- **{key.replace('_', ' ').title()}**: `{', '.join(metrics)}`\n"
                formatted_reasons += "\n Which we currently don't support. Look at the tab on your left to see the allowed metrics."

        return formatted_reasons
    
    def get_status_elements(self, message: None):
        if "Sorry" in message:
            return st.error("We couldn't classify your question, look at the instructions below and rephrase it", icon="❗")
        
        elif "Our system" in message:
            return st.info('Please, check if we correctly captured your question intent', icon="ℹ️")