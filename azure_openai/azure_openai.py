import os
import time
import json
import logging
import traceback

from openai import AzureOpenAI

from azure_openai.prompts import *
from azure_logging.azure_logging import logger

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MyAzureOpenAI:
    def __init__(self):
        # ! Have to remove these credentials
        self.api_key      = os.getenv("openaikey1")
        self.api_version  = os.getenv("openapiversion")
        self.api_endpoint = os.getenv("openaiendpoint")

    def _openai_chat_completion(self, messages: list[dict], model_name = "gpt-35-turbo-16k"):
        client = AzureOpenAI(
            api_key        = self.api_key, 
            api_version    = self.api_version,
            azure_endpoint = self.api_endpoint,
        )

        response = client.chat.completions.create(
            model       = model_name,
            messages    = messages,
            temperature = 0.2,
            top_p       = 0.8,
            stop        = None
        )
        return response


    async def generate_greet_messages(self, query, user_fullname: str) -> str:
        start = time.time()
        """
        Generate a greeting message based on the query and user's full name. 
        If the query is a greeting, respond with a greeting message.
        If the query is a request for the application access list, respond with the application access list.
        If the query is not a greeting or a request for the application access list, respond with a generic message.
        
        Args:
        - query: The user's query.
        - application_access_list: The application access list.
        - user_fullname: The user's full name.
        
        Returns:
        - str: The response message.
        """
        try:
            format_dict = {
                # "application_access_list": application_access_list,
                "user_fullname": user_fullname.title(),
                "query": query
            }
            prompt = greetings_prompt.format(**format_dict)
            messages = [
                {"role": "system", "content": prompt},
            ]
            response = self._openai_chat_completion(messages)
            json_obj = json.loads(response.model_dump_json(indent=2))
        
            # Extract the content
            content = json_obj["choices"][0]["message"]["content"]
            end = time.time()
            logger.log(msg= f"Successfully generated greet message..{end-start}",level=logging.INFO)
            return content
        except Exception as e:
            logger.log(msg= f"Greet message not generated..{e}",level=logging.INFO)
    

    def followup_query_modified(self, query, history=None) -> str:
        """
        Checks if a query is a follow-up query based on context.

        Args:
            query (str): The natural language query.
            last_history (str, optional): The last history or context (default is None).

        Returns:
            bool: True if the query is a follow-up, False otherwise.
        """
        start = time.time()
        try:
            messages = [
                        {'role': 'system', 'content': followup_query_prompt},
                        {'role': 'user', 'content': f'''User: Based on the Chat history rephrase the user query. You need to analyse the  user, bot interaction and need to frame the follow up query which carries all the full information (without any other context) to query.
                         Chat history:
                         {history}
                         User Query :{query}
                         Rephrased question:'''}
                        ]
            response = self._openai_chat_completion(messages)
            end = time.time()
            logger.log(msg= f"Successfully modified followup query..{end-start}",level=logging.INFO)
            return response.choices[0].message.content
        except Exception as e:
            logger.log(msg= f"Not able to modify followup query..{e}",level=logging.ERROR)


    def check_followup(self, query, history=None) -> str:
        """
        Checks if a query is a follow-up query based on context.

        Args:
            query (str): The natural language query.
            last_history (str, optional): The last history or context (default is None).

        Returns:
            bool: True if the query is a follow-up, False otherwise.
        """
        start = time.time()
        try:
            messages = [
                {'role': 'system', 'content': check_followup_prompt},
                {'role': 'user', 'content': f'''Based on the given Current User Query and Previous User Query, find the given user query is a follow up query or not to the chat history. Carefully analyse the interaction between Previous Query and Current User Query and check whether the current User Query can be the follow-up query to the previous query. Based on previous query and Bot answer, decide the current query is a follow up or not. If the current query is a follow up then reply with True
                 Output a single word True if it is a follow up query, if it is not a follow up query then output a single query False
                 
                 Chat history:
                 {history}
                 
                 Thought: Based on the given above Previous User Query and below Current User Query (without considering Bot Response) I need to decide the Current User Query is a follow up or not.
                 
                 Current User Query: {query}
                 Response:
                 '''}
                ]
            response = self._openai_chat_completion(messages)
            end = time.time()
            logger.log(msg= f"Successfully checked for followup query..{end-start}",level=logging.INFO)
            return response.choices[0].message.content
        except Exception as e:
            logger.log(msg= f"Not able to check for followup query..{e}",level=logging.ERROR)

    def openai_embeddings(self, query: str) -> list[float]:
        # ! Have to remove these credentials
        start = time.time()
        embed_model_name = "text-embedding-ada"

        client = AzureOpenAI(
            api_key        = self.api_key,
            api_version    = self.api_version,
            azure_endpoint = self.api_endpoint
        )
        end = time.time()
        logger.log(msg= f"Successfully created openai embeddings..{end-start}",level=logging.INFO)
        return client.embeddings.create(input = [query], model=embed_model_name).data[0].embedding
    

    def doc_openai(self, context: list, query: str):
        start = time.time()
        prompt = prompt_doc.format(context = context)
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ]
        response = self._openai_chat_completion(messages)

        json_obj = json.loads(response.model_dump_json(indent = 2))
        content = json_obj["choices"][0]["message"]["content"]
        end = time.time()
        logger.log(msg= f"Successfully fetched unstructured answers..{end-start}",level=logging.INFO)
        return content
    
    def sql_openai(self, query):
        start = time.time()
        messages = [
            {'role': 'system', 'content': prompt_sql},
            {'role': 'user', 'content': query}
        ]

        response = self._openai_chat_completion(messages, model_name = "gpt-4")

        json_obj = json.loads(response.model_dump_json(indent=2))
        content = json_obj["choices"][0]["message"]["content"]
        end = time.time()
        logger.log(msg= f"Successfully fetched sql answers..{end-start}",level=logging.INFO)
        return content
    
    def sql_openai_with_metadata(self, query: str, selected_tables_columns, tables_relations, rules_json: list[str]):
        format = '''
        {    
            "Query": "<SQL Query>"
        }'''

        prompt = promot_for_metadata_dashboards.format(
            selected_tables_columns = selected_tables_columns,
            tables_relations        = tables_relations,
            rules_json              = rules_json,
            query                   = query,
            format                  = format
        )

        messages = [{"role":"system","content": prompt},
                    {"role":"user","content":   query}]

        response = self._openai_chat_completion(messages, model_name = "gpt-4")

        sql_query = response.choices[0].message.content
        try:
            sql_query = json.loads(sql_query)
            return sql_query
        except:
            return {"Query": "Please rephrase the question"}
    
    def modify_sql_query(self, user_question, original_sql_query, rules) -> dict:
        prompt = modify_sql_query.format(user_question = user_question, original_sql_query = original_sql_query, rules = rules)
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': ""}
        ]

        response = self._openai_chat_completion(messages, model_name="gpt-4")
        json_obj = json.loads(response.model_dump_json(indent = 2))
        print(f"{bcolors.OKGREEN}Successfully modified SQL query ==> {json_obj}{bcolors.ENDC}")
        content = json.loads(json_obj["choices"][0]["message"]["content"])
        return content
    
    def dataframe_openai(self, user_question, df) -> str:
        start = time.time()
        sql_answer_desc = dataframe_to_natural_lang.format(final_query = user_question, df = df)
        messages = [
            {'role': 'user', 'content': sql_answer_desc},
            {'role': 'assistant', 'content': ""},
        ]
        response = self._openai_chat_completion(messages)
        json_obj = json.loads(response.model_dump_json(indent=2))
        content = json_obj["choices"][0]["message"]["content"]
        end = time.time()
        print(f"{bcolors.OKGREEN}Successfully dataframe generated in {end-start} seconds{bcolors.ENDC}")
        logger.log(msg= f"Successfully dataframe generated..{end-start}",level=logging.INFO)
        return content
    
    def generate_graph_code(self, df, user_question):
        start_time = time.time()
        prompt = prompt_graph.format(df = df.to_markdown(), user_query = user_question)

        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'assistant', 'content': ""},
        ]

        response = self._openai_chat_completion(messages)
        json_obj = json.loads(response.model_dump_json(indent=2))
    
        # Extract the content
        content = json_obj["choices"][0]["message"]["content"]
        end_time = time.time()
        print(f"{bcolors.OKGREEN}Successfully generated graph code in {end_time-start_time} seconds{bcolors.ENDC}")
        logger.log(msg= f"Successfully generated graph code..{end_time-start_time}", level=logging.INFO)
        return content
    

    def generate_sql_query_with_row_level_access(self, schema_details: str):
        mandatory_rule = """
        MANDATORY RULE:
        - If the table has the ROW_LEVEL_ACCESS, then you always have to add this to your generated SQL query as a INNER JOIN.
        - Always have to do the INNER JOIN by the information given to the ROW_LEVEL_ACCESS based on the column name to the ROW_LEVEL_ACCESS.
        - Use the alliasing ('A', 'B', ...) to the every columns in the SQL query. Also, use this by understanding the QUESTION properly, and in the SELECT parts of the query, always that aliasing which is belongs to the TABLE_NAME only.
        - Always add the 'WHERE ...' conditions of the ROW_LEVEL_ACCESS to your generated SQL query as it is. Don't change the conditions based on the QUESTION.
        - Try to make the SQL query as simple as possible by understanding the QUESTION properly.
        - Always pick the column names for the sql query from the given COLUMN_NAMES list only.
        """
        prompt                = sql_generate_prompt_with_row_level + "\n\n\n\n" + schema_details + mandatory_rule

        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'assistant', 'content': ""},
        ]

        response = self._openai_chat_completion(messages)
        json_obj = json.loads(response.model_dump_json(indent=2))
        content  = json_obj["choices"][0]["message"]["content"]

        return content
    
    def modify_sql_query_with_row_level_access(self, generated_sql_query: str, row_level_access: str, row_level_access_col_name: str):
        prompt = modify_sql_query_with_row_level + "\n\n\n\n" + f"ROW_LEVEL_ACCESS: {row_level_access}\n" + f"ROW_LEVEL_COLUMN_NAME: {row_level_access_col_name}\n" + f"SQL_QUERY: {generated_sql_query}"

        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'assistant', 'content': ""},
        ]

        response = self._openai_chat_completion(messages)
        json_obj = json.loads(response.model_dump_json(indent=2))
        content  = json_obj["choices"][0]["message"]["content"]

        return content
    
    def generate_faqs_based_on_dataframe(self, schema: str):
        prompt = faq_generate_prompt + "\n\n\n\n" + schema
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'assistant', 'content': ""},
        ]

        response = self._openai_chat_completion(messages)
        try:
            json_obj = json.loads(response.model_dump_json(indent = 4))
            content = json_obj["choices"][0]["message"]["content"]
            return json.loads( content )
        except Exception as e:
            print(f"{bcolors.FAIL}Error in GPT for FAQs generation: {e}{bcolors.ENDC}")
            tb = traceback.format_exc()
            logger.log(msg= f"Error: {e}..{tb}",level=logging.ERROR)
            return "Error in generating FAQs"
