import os
import re
import json
import time
import random
import logging
import traceback
import concurrent.futures
from pydantic import BaseModel

import uvicorn
import numpy as np
import pandas as pd
from databricks import sql
import plotly.express as px
import semantic_kernel as sk
from dotenv import load_dotenv
import plotly.graph_objects as go
from fastapi import FastAPI, Response, status
from azure_logging.azure_logging import logger
from databricks.sql.exc import ServerOperationError
from semantic_kernel.planning.basic_planner import BasicPlanner
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion


load_dotenv()

from azure_openai.prompts import prompt_graph
from azure_openai.azure_openai import MyAzureOpenAI
from az_semantic_kernel.prompts import system_prompt
from azure_cosmos_db.azure_cosmos_db import COSMOS_DB
from azure_security.azure_security import AzureKeyVault
from error_handling.error_handling import SqlNotGenerateError
from az_cognitive_search.az_cognitive_search import AZCognitiveSearch
from utils import check_dashboard_name_in_query, get_user_app, process_user_question, generate_sql_first_level_access, generate_answers_for_pdf
from utils import graph_decide, is_valid_python_code, generate_sql_with_row_column_level_access, get_or_generate_faqs
from az_blob_services.az_blob_services import upload_image_to_blob


# set all the environmental variables from the azure key vault
my_key_value = AzureKeyVault().set_environment_from_key_vault()


app = FastAPI()

class QueryRequest(BaseModel):
    user_email: str
    conversation_id: str
    query: str
    user_full_name: str

class QueryRequestNew(BaseModel):
    user_email: str
    conversation_id: str
    query: str
    user_full_name: str
    dashboard_name: str

class WelcomeRequest(BaseModel):
    user_email: str

class QueryAnswer(BaseModel):
    user_email: str
    conversation_id: str
    query: str
    user_full_name: str
    query_type: str
    # dashboard_name: str

class DashboardFAQs(BaseModel):
    user_email: str
    dashboard_name: str

class DeleteItems(BaseModel):
    user_email: str

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



def handle_sql_query(user_details, user_question, return_response, response):
    try:
        generated_sql_query    = generate_sql_first_level_access(
            user_question = user_question,
            user_email = user_details["user_email"], 
            dashboard_name = user_details["dashboard_name"].lower()
        )
        return_response["sql_query"] = generated_sql_query['sql_query']
        print(f"{bcolors.FAIL}Generated SQL Query from the old prompt: {generated_sql_query['sql_query']}{bcolors.ENDC}")

        # row_level_sql_query_start_time = time.time()
        # row_level_access_sql_query = generate_sql_with_row_column_level_access(
        #     user_question = user_question,
        #     user_email    = user_details["user_email"].lower(),
        #     dashboard_name = user_details["dashboard_name"].lower()
        # )
        # print(f"{bcolors.FAIL}Row level access all details: {row_level_access_sql_query.keys()}{bcolors.ENDC}")
        # sql_query = row_level_access_sql_query["sql_query"]
        sql_query = generated_sql_query['sql_query']
        # return_response["row_level_access_sql_query"] = sql_query
        # print(f"{bcolors.FAIL}Generated SQL Query from the new prompt: {row_level_access_sql_query['sql_query']}{bcolors.ENDC}")
        return_response["sql_query"] = sql_query

        if sql_query.lower().replace(".", "") == "NO SQL QUERY CAN BE GENERATED".lower():
            print(f"""{bcolors.FAIL}No SQL query can be generated{bcolors.ENDC}""")
            raise SqlNotGenerateError("No SQL query can be generated")
        elif sql_query.lower().replace(".", "") == "User has no access to browse this type of data".lower():
            print(f"""{bcolors.FAIL}User has no access to browse this type of data{bcolors.ENDC}""")
            return_response["answer"] = "Access Denied: You currently lack the necessary permissions to view this data. Please contact your administrator for further assistance."
            return_response["citation"]      = ""
            return_response["question_type"] = "sql"
            return return_response
        
        row_level_sql_query_end_time = time.time()

        # ! Execute the SQL query and fetch the results
        with sql.connect(
            server_hostname = os.getenv("dbserverhostname"),
            http_path       = os.getenv("dbhttppath"),
            access_token    = os.getenv("dbaccesstoken")
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET spark.sql.legacy.timeParserPolicy = LEGACY")
                # cursor.execute("SELECT * FROM hello;")
                cursor.execute(sql_query)
                result = cursor.fetchall()
                if result is None:
                    print("***No result***")
                else:
                    print("****Fetching result****")

        df         = pd.DataFrame([r.asDict() for r in result])
        columns = df.columns.tolist()
        data = df.head(5)
        # graph_prompt = prompt_graph.format(user_question = user_question, c = columns, data = data)

        data_fetch_from_databricks_end_time = time.time()
        print(f"{bcolors.OKGREEN}Dataframe shape before generating answer: {df.shape} and took {data_fetch_from_databricks_end_time - row_level_sql_query_end_time} seconds.{bcolors.ENDC}")
        # sql_answer = MyAzureOpenAI().dataframe_openai(user_question, df.to_markdown())

        # TODO: Execute the normal language conversation code and graph generation code in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create an instance of MyAzureOpenAI
            azure_openai_instance = MyAzureOpenAI()

            if df.shape[0] >= 2 and df.shape[1] >= 2:
                # Submit the methods to the executor
                future1 = executor.submit(azure_openai_instance.dataframe_openai, user_question, df.to_markdown())
                future2 = executor.submit(azure_openai_instance.generate_graph_code, df, user_question)

                # Wait for both futures to complete
                concurrent.futures.wait([future1, future2])

                # Get the results from the futures
                sql_answer = future1.result()
                graph_code = future2.result()
            else:
                future1 = executor.submit(azure_openai_instance.dataframe_openai, user_question, df.to_markdown())
                concurrent.futures.wait([future1])
                sql_answer = future1.result()
                graph_code = None


        natural_lang_and_graph_code_generation_end_time = time.time()
        print(f"{bcolors.OKGREEN}Natural Language and Graph Code Generation took {natural_lang_and_graph_code_generation_end_time - data_fetch_from_databricks_end_time} seconds{bcolors.ENDC}")
        
        # setup the response format for the Bot & save the conversation to the CosmosDB for the SQL
        try:
            sql_json_answer = json.loads(sql_answer)
            print(f"{bcolors.OKCYAN}Keys in the SQL JSON Answer: {sql_json_answer.keys()}{bcolors.ENDC}")
            return_response["answer"] = sql_json_answer["answer"]
            return_response["similar_queries"] = sql_json_answer["similar_queries"]

            # ! Save the question to the FAQs database in Cosmos DB
            is_saved = COSMOS_DB().save_user_query_for_faqs(
                user_query      = user_details["query"],
                rephrased_query = user_question,
                # dashboard_name  = row_level_access_sql_query['application_name'],
                dashboard_name = generated_sql_query['application_name'],
            )
            print(f"{bcolors.OKGREEN}Is saved to FAQs: {is_saved}{bcolors.ENDC}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error from sql_answer: {str(e)}\nTraceback:\n{tb}")
            logger.log(msg= f"Error in sql_answer...{tb}", level=logging.ERROR)
            return_response["answer"] = sql_answer

        save_to_cosmos_db_for_faqs_end_time = time.time()
        print(f"{bcolors.OKGREEN}Saving to Cosmos DB for FAQs took {save_to_cosmos_db_for_faqs_end_time - natural_lang_and_graph_code_generation_end_time} seconds{bcolors.ENDC}")
        
        ## ! Generate the graphs for the SQL query
        if df.shape[0]<2:
            print(f"{bcolors.FAIL}Dataframe shape: {df.shape} in less than 2{bcolors.ENDC}")
            return_response['table_markdown'] = ""
        elif "table" in user_question.lower() or "tabular" in user_question.lower():
            print(f"{bcolors.OKGREEN}Dataframe shape: {df.shape} in table{bcolors.ENDC}")
            return_response['table_markdown'] = df.to_markdown()
        elif df.shape[0] >= 2 and df.shape[1] >= 2:
            if graph_code is None:
                return_response["table_markdown"] = df.to_markdown()
                # TODO: Have to add the else condition so that after the if block, rest codes are not to be executed
            graph_code = graph_code.replace("`", "").replace("python", "").replace("fig.show()", "").strip()
            print(f"{bcolors.OKGREEN}Graph Code: {graph_code}{bcolors.ENDC}")
            try:
                if is_valid_python_code(graph_code):
                    nan = np.nan
                    local_namespace = {"df": df, "nan": nan}
                    exec(graph_code, {}, local_namespace)
                    
                    print(f"{bcolors.OKGREEN}All local variables: {local_namespace}{bcolors.ENDC}")
                    fig     = local_namespace['fig']
                    sas_url = upload_image_to_blob(fig.to_image(format="png"))
                    return_response['table_markdown'] = sas_url
                else:
                    print(f"Graph code is not executable")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"{bcolors.FAIL}Error from Graph Code: {str(e)}\nTraceback:\n{tb}{bcolors.ENDC}")
                logger.log(msg= f"Error in graph code...{tb}", level=logging.ERROR)


                phrases_to_check = ['graph','plot','chart']
                phrases_to_check_bar = ['bar graph', 'bar chart', 'bar plot']
                phrases_to_check_scatter = ['scatter graph', 'scatter chart', 'scatter plot']
                phrases_to_check_line = ['line graph', 'line chart', 'line plot']
                phrases_to_check_pie = ['pie graph', 'pie chart', 'pie plot']

                # Create a regular expression pattern to match any of the phrases
                basic = '|'.join(re.escape(phrase) for phrase in phrases_to_check)
                bar = '|'.join(re.escape(phrase) for phrase in phrases_to_check_bar)
                scatter = '|'.join(re.escape(phrase) for phrase in phrases_to_check_scatter)
                line = '|'.join(re.escape(phrase) for phrase in phrases_to_check_line)
                pie = '|'.join(re.escape(phrase) for phrase in phrases_to_check_pie)

                print(f"{bcolors.OKGREEN}Dataframe shape: {df.shape} in greater than or equal 2{bcolors.ENDC}")
                # graph_type = graph_decide(graph_prompt)
                # print(f"{bcolors.OKGREEN}Graph Type: {graph_type}{bcolors.ENDC}")
                colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
                if re.search(basic, user_question, re.IGNORECASE):
                    # if "bar" in user_question.lower() or "bar" in graph_type:
                    if re.search(bar, user_question, re.IGNORECASE):
                        # Bar Plot
                        fig = px.bar(df,x=columns[0],y=columns[1],
                                    color = columns[0],
                                    text_auto = '.5s',
                                    title='Bar Plot ' + columns[1])
                        if df.shape[0] > 9:
                            fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35), showlegend=False)
                        else:
                            fig.update_layout(width=650, height=550)
                        # html_format = fig.to_html()
                        image_bytes = fig.to_image(format="png")
                    # Use the regular expression to search for the pie phrases in the user's question
                    # elif "pie" in user_question.lower() or"pie" in graph_type:
                    elif re.search(pie, user_question, re.IGNORECASE):
                        print(f"{bcolors.OKGREEN}Pie Graph columns{columns}{bcolors.ENDC}")
                        # fig = go.Figure(data=[go.Pie(labels=columns,
                        #                             values=columns[1])])
                        fig = px.pie(df, names=columns[0], values=columns[1])
                        fig.update_traces(textinfo='percent+label',textposition='inside',
                                        textfont_size=20,
                                        marker=dict(colors=colors,line=dict(color='#000000', width=2)))
                        if df.shape[0] > 9:
                            fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                        else:
                            fig.update_layout(width=650, height=550)
                        # html_format = fig.to_html()
                        image_bytes = fig.to_image(format="png")
                        
                    # elif "scatter" in user_question.lower() or "scatter" in graph_type:
                    elif re.search(scatter, user_question, re.IGNORECASE):
                        fig = px.scatter(df,x=columns[0],y=columns[1],
                                        # color = columns[0],
                                        size= columns[1],
                                        title='Scatter Plot ' + columns[1])
                        if df.shape[0] > 9:
                            fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                        else:
                            fig.update_layout(width=650, height=550)
                        # html_format = fig.to_html()
                        image_bytes = fig.to_image(format="png")
                    # elif "line" in user_question.lower() or "line" in graph_type:
                    elif re.search(line, user_question, re.IGNORECASE):
                        fig = px.line(df,x=columns[0],y=columns[1],
                                    # color=columns[0],
                                    markers=True,
                                    title='Line Graph ' + columns[1],
                                    text=columns[1]
                                )
                        if df.shape[0] > 9:
                            fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                        else:
                            fig.update_layout(width=650, height=550)
                        # html_format = fig.to_html()
                        image_bytes = fig.to_image(format="png")

                    sas_graph_url = upload_image_to_blob(image_bytes)
                    return_response['table_markdown'] = sas_graph_url
                else:
                    return_response['table_markdown'] = df.to_markdown()
        else:
            print(f"{bcolors.FAIL}Dataframe shape: {df.shape} in else block.{bcolors.ENDC}")
            return_response['table_markdown'] = df.to_markdown()

        graph_generation_with_sas_url_generation_end_time = time.time()
        print(f"{bcolors.OKGREEN}Graph Generation and SAS URL Generation took {graph_generation_with_sas_url_generation_end_time - save_to_cosmos_db_for_faqs_end_time} seconds{bcolors.ENDC}")

        # return_response["citation"] = {
        #     "database_name": row_level_access_sql_query['database_name'],
        #     "table_name": row_level_access_sql_query['table_name'],
        #     "group_name": row_level_access_sql_query["group_name"],
        #     "application_name": row_level_access_sql_query["application_name"].title()
        # }
        return_response["citation"] = {
            "database_name": generated_sql_query['Database_Name'],
            "table_name": generated_sql_query['Table_Name'],
            "group_name": generated_sql_query["group_name"],
            "application_name": generated_sql_query["application_name"].title()
        }
        return_response["question_type"] = "sql"

        ## Save the SQL conversation to the CosmosDB
        item = {
            "rephrased_query": user_question,
            "actual_query": user_details["query"],
            "sql_query": sql_query,
            "answer": sql_answer,
            # "citation": {
            #     "database_name": row_level_access_sql_query['database_name'],
            #     "table_name": row_level_access_sql_query['table_name'],
            #     "group_name": row_level_access_sql_query["group_name"],
            #     "application_name": row_level_access_sql_query["application_name"].title()
            # },
            # "context": row_level_access_sql_query['context'],
            "citation": {
                "database_name": generated_sql_query['Database_Name'],
                "table_name": generated_sql_query['Table_Name'],
                "group_name": generated_sql_query["group_name"],
                "application_name": generated_sql_query["application_name"].title()
            },
            "context": generated_sql_query["context"]
        }

        is_saved = COSMOS_DB()._add_conversation_item(
            user_email = user_details["user_email"].lower(),
            conv_id    = user_details["conversation_id"],
            question_type = "sql",
            item = item
        )

        save_conversation_end_time = time.time()
        print(f"{bcolors.OKGREEN}Saving the conversation to Cosmos DB took {save_conversation_end_time - graph_generation_with_sas_url_generation_end_time} seconds{bcolors.ENDC}")
        if is_saved:
            print("Conversation saved successfully")
        else:
            print("Error saving conversation")

        return return_response

    except ServerOperationError as e:
        tb = traceback.format_exc()
        print(f"Error from ServerOperationError: {str(e)}\nTraceback:\n{tb}")
        logger.log(msg= f"Error in server operation...{tb}", level=logging.ERROR)
        
        if response == "others+sql":
            return_response["answer"] = "An internal server error occurred while processing your request. Please modify your query and try again."
            return_response['table_markdown'] = ""
            return_response["citation"] = ""
            return_response["question_type"] = "sql"
            return return_response
        
        is_saved, return_response = generate_answers_for_pdf(
            user_question   = user_question,
            return_response = return_response,
            conv_id         = user_details["conversation_id"],
            actual_query    = user_details["query"],
            user_email      = user_details["user_email"].lower()
        )
        if is_saved:
            print("Conversation saved successfully")
        else:
            print("Error saving conversation")

        return return_response

    except SqlNotGenerateError as e:
        tb = traceback.format_exc()
        print(f"Error from SqlNotGenerateError: {str(e)}\nTraceback:\n{tb}")
        logger.log(msg= f"Error in sql query not generated...{tb}", level=logging.ERROR)

        if response == "others+sql":
            return_response["answer"] = "An internal server error occurred while processing your request. Please modify your query and try again."
            return_response['table_markdown'] = ""
            return_response["citation"] = ""
            return_response["question_type"] = "sql"
            return return_response
        
        
        is_saved, return_response = generate_answers_for_pdf(
            user_question   = user_question,
            return_response = return_response,
            conv_id         = user_details["conversation_id"],
            actual_query    = user_details["query"],
            user_email      = user_details["user_email"].lower()
        )
        if is_saved:
            print("Conversation saved successfully")
        else:
            print("Error saving conversation")

        return return_response

    except Exception as ex:
        tb = traceback.format_exc()
        print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")

        logger.log(msg= f"Error{tb}", level=logging.ERROR)

        if response == "others+sql":
            return_response["answer"] = "An internal server error occurred while processing your request. Please modify your query and try again."
            return_response['table_markdown'] = ""
            return_response["citation"] = ""
            return_response["question_type"] = "sql"
            return return_response
    
        return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
        return_response['table_markdown'] = ""
        return_response["citation"] = ""
        return_response["question_type"] = "sql"

        return return_response

@app.get("/")
async def read_root():
    return {"message": f"Hey, Welcome to the ACTVET! Your lucky number is: {random.randint(1, 100)}"}


@app.post("/welcome-message")
async def welcome_message(request: WelcomeRequest):
    print(f"Received user email: {request.user_email} for welcome message")
    user_email = request.user_email.lower()
    user_apps  = await get_user_app(user_email)
    if user_apps == "No Record Found":
        return "Access Denied: You currently lack the necessary permissions to view this data. Please contact your administrator for further assistance."
    else:
        return user_apps


@app.post("/api/messages")
# async def submit_query(request: QueryRequest):
async def submit_query(request: QueryRequestNew):
    user_details = {
        "user_email": request.user_email.lower(),
        "conversation_id": request.conversation_id,
        "query": request.query,
        "user_full_name": request.user_full_name,
        "dashboard_name": request.dashboard_name.lower()
    }

    if user_details["query"].strip() == "open dashboard":
        return {"answer": "Please type 'Open Dashboard' to open the dashboard."}

    print(f"Received user query: {user_details['query']}")
    return_response = {
        "user_question": user_details["query"],
    }

    # Check the dashboard name from the user query
    dashboard_name = check_dashboard_name_in_query(user_details["query"])
    print(f"{bcolors.FAIL}Dashboard name found in the user query: {dashboard_name}{bcolors.ENDC}")
    # if dashboard_name == "" or dashboard_name is None:
    #     dashboard_name = user_details["dashboard_name"]
    #     sql_response = True
    # else:
    #     user_details["dashboard_name"] = dashboard_name
    #     sql_response = False

    # if user_details["dashboard_name"] is None or user_details["dashboard_name"] == "":
    #     dashboard_name = check_dashboard_name_in_query(user_details["query"])
    # else:
    #     dashboard_name = user_details["dashboard_name"]

    print(f"{bcolors.FAIL}Dashboard name found from the actual query:\n{dashboard_name=}{bcolors.ENDC}")

    start_time_for_greetings = time.time()
    # reduce the dependency by greetings message or sql query type selection
    if "access list" in user_details["query"].lower():
        response = "application_access_list"
    elif "graph" in user_details["query"].lower() or "chart" in user_details["query"].lower() or "plot" in user_details["query"].lower():
        response = "others+sql"
    elif len(dashboard_name) > 0 or dashboard_name != '':
        # user_details["dashboard_name"] = dashboard_name
        print(f"{bcolors.FAIL}Dashboard name found in the user query and making others+sql unnecessary.{bcolors.ENDC}")
        response = "others+sql"
    else:
        response = None
    

    # If we don't have any response type selected then generate the response by using GPT
    if response is None:
        try:
            response    = await MyAzureOpenAI().generate_greet_messages(user_details["query"], user_details["user_full_name"])
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")
            logger.log(msg= f"Successfully generated greet message.. {tb}", level=logging.ERROR)
            return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
            return return_response

    print(f"{bcolors.FAIL}response = {response}{bcolors.ENDC}")

    if response == "others" or response == "others+sql":
        try:
            # ? Fetch the last conversations and generate the rephrased query
            rephrased_query_time = time.time()
            history = COSMOS_DB().get_last_conversation(user_details["user_email"].lower())
            # print(f"{bcolors.FAIL}Last conversation: {history}{bcolors.ENDC}")

            user_question = process_user_question(
                user_question = user_details["query"],
                conversations = history,
            )

            rephrased_query_end_time = time.time()
            print(f"{bcolors.OKGREEN}Rephrased user question: {user_question} and took {rephrased_query_end_time - rephrased_query_time} seconds {bcolors.ENDC}")

            # if already selected the query is for SQL, then no need to classify again using the semantic kernel
            if response != "others+sql":
                kernel   = sk.Kernel()
                kernel.add_chat_service(
                    "chat_completion",
                    AzureChatCompletion(deployment_name="gpt-35-turbo-16k", endpoint = os.getenv("openaiendpoint"), api_key = os.getenv("openaikey1")),
                )
                skills_directory     = "./az_semantic_kernel/skills"
                summarize_skill      = kernel.import_semantic_plugin_from_directory(skills_directory, "pdf")
                writer_skill         = kernel.import_semantic_plugin_from_directory(skills_directory, "sql")
                classification_skill = kernel.import_semantic_plugin_from_directory(skills_directory, "classification")
                
                planner         = BasicPlanner()
                basic_plan      = await planner.create_plan(user_question, kernel, system_prompt)
                input_json_plan = eval(basic_plan.generated_plan["input"])
                return_response["rephrased_query"] = user_question
                return_response["sk_response"]     = input_json_plan
                print(f"{bcolors.OKGREEN}Semantic Kernel took {time.time() - rephrased_query_end_time} seconds{bcolors.ENDC}")
            else:
                input_json_plan = None
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error: {str(e)}\nTraceback:\n{tb}")
            logger.log(msg= f"Successfully generated greet message.. {tb}", level=logging.ERROR)
            return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
            return return_response

        try:
            if input_json_plan is not None and input_json_plan["subtasks"][1]['function'] == "pdf.pdf_skill":
                is_saved, return_response = generate_answers_for_pdf(
                    user_question   = user_question,
                    return_response = return_response,
                    conv_id         = user_details["conversation_id"],
                    actual_query    = user_details["query"],
                    user_email      = user_details["user_email"].lower()
                )

                print(f"{bcolors.FAIL}Response from the PDF: {return_response['answer']}{bcolors.ENDC}")
                if type(return_response['answer']) == dict and "this information is out of my uploaded knowledge base".lower() in return_response["answer"]['answer'].lower():
                    return handle_sql_query(user_details, user_question, return_response, response)
                elif type(return_response["answer"] == str) and "this information is out of my uploaded knowledge base".lower() in return_response["answer"].lower():
                    return handle_sql_query(user_details, user_question, return_response, response)
                elif type(return_response['answer']['answer']) == dict and "this information is out of my uploaded knowledge base".lower() in return_response["answer"]['answer'].lower():
                    return handle_sql_query(user_details, user_question, return_response, response)
                elif type(return_response["answer"]["answer"] == str) and "this information is out of my uploaded knowledge base".lower() in return_response["answer"]['answer'].lower():
                    return handle_sql_query(user_details, user_question, return_response, response)
                
                if is_saved:
                    print("Conversation saved successfully")
                else:
                    print("Error saving conversation")
                
                return return_response
                
                
            elif response == "others+sql" or (input_json_plan is not None and input_json_plan["subtasks"][1]['function'] == "sql.sql_skill"):
                # 580 to 920 lines
                return handle_sql_query(user_details, user_question, return_response, response)
                # try:
                #     generated_sql_query    = generate_sql_first_level_access(
                #         user_question = user_question,
                #         user_email = user_details["user_email"], 
                #         dashboard_name = user_details["dashboard_name"].lower()
                #     )
                #     return_response["sql_query"] = generated_sql_query['sql_query']
                #     print(f"{bcolors.FAIL}Generated SQL Query from the old prompt: {generated_sql_query['sql_query']}{bcolors.ENDC}")

                #     # row_level_sql_query_start_time = time.time()
                #     # row_level_access_sql_query = generate_sql_with_row_column_level_access(
                #     #     user_question = user_question,
                #     #     user_email    = user_details["user_email"].lower(),
                #     #     dashboard_name = user_details["dashboard_name"].lower()
                #     # )
                #     # print(f"{bcolors.FAIL}Row level access all details: {row_level_access_sql_query.keys()}{bcolors.ENDC}")
                #     # sql_query = row_level_access_sql_query["sql_query"]
                #     sql_query = generated_sql_query['sql_query']
                #     # return_response["row_level_access_sql_query"] = sql_query
                #     # print(f"{bcolors.FAIL}Generated SQL Query from the new prompt: {row_level_access_sql_query['sql_query']}{bcolors.ENDC}")
                #     return_response["sql_query"] = sql_query

                #     if sql_query.lower().replace(".", "") == "NO SQL QUERY CAN BE GENERATED".lower():
                #         print(f"""{bcolors.FAIL}No SQL query can be generated{bcolors.ENDC}""")
                #         raise SqlNotGenerateError("No SQL query can be generated")
                #     elif sql_query.lower().replace(".", "") == "User has no access to browse this type of data".lower():
                #         print(f"""{bcolors.FAIL}User has no access to browse this type of data{bcolors.ENDC}""")
                #         return_response["answer"] = "Access Denied: You currently lack the necessary permissions to view this data. Please contact your administrator for further assistance."
                #         return_response["citation"]      = ""
                #         return_response["question_type"] = "sql"
                #         return return_response
                    
                #     row_level_sql_query_end_time = time.time()

                #     # ! Execute the SQL query and fetch the results
                #     with sql.connect(
                #         server_hostname = os.getenv("dbserverhostname"),
                #         http_path       = os.getenv("dbhttppath"),
                #         access_token    = os.getenv("dbaccesstoken")
                #     ) as connection:
                #         with connection.cursor() as cursor:
                #             cursor.execute("SET spark.sql.legacy.timeParserPolicy = LEGACY")
                #             # cursor.execute("SELECT * FROM hello;")
                #             cursor.execute(sql_query)
                #             result = cursor.fetchall()
                #             if result is None:
                #                 print("***No result***")
                #             else:
                #                 print("****Fetching result****")

                #     df         = pd.DataFrame([r.asDict() for r in result])
                #     columns = df.columns.tolist()
                #     data = df.head(5)
                #     # graph_prompt = prompt_graph.format(user_question = user_question, c = columns, data = data)

                #     data_fetch_from_databricks_end_time = time.time()
                #     print(f"{bcolors.OKGREEN}Dataframe shape before generating answer: {df.shape} and took {data_fetch_from_databricks_end_time - row_level_sql_query_end_time} seconds.{bcolors.ENDC}")
                #     # sql_answer = MyAzureOpenAI().dataframe_openai(user_question, df.to_markdown())

                #     # TODO: Execute the normal language conversation code and graph generation code in parallel
                #     with concurrent.futures.ThreadPoolExecutor() as executor:
                #         # Create an instance of MyAzureOpenAI
                #         azure_openai_instance = MyAzureOpenAI()

                #         if df.shape[0] >= 2 and df.shape[1] >= 2:
                #             # Submit the methods to the executor
                #             future1 = executor.submit(azure_openai_instance.dataframe_openai, user_question, df.to_markdown())
                #             future2 = executor.submit(azure_openai_instance.generate_graph_code, df, user_question)

                #             # Wait for both futures to complete
                #             concurrent.futures.wait([future1, future2])

                #             # Get the results from the futures
                #             sql_answer = future1.result()
                #             graph_code = future2.result()
                #         else:
                #             future1 = executor.submit(azure_openai_instance.dataframe_openai, user_question, df.to_markdown())
                #             concurrent.futures.wait([future1])
                #             sql_answer = future1.result()
                #             graph_code = None


                #     natural_lang_and_graph_code_generation_end_time = time.time()
                #     print(f"{bcolors.OKGREEN}Natural Language and Graph Code Generation took {natural_lang_and_graph_code_generation_end_time - data_fetch_from_databricks_end_time} seconds{bcolors.ENDC}")
                    
                #     # setup the response format for the Bot & save the conversation to the CosmosDB for the SQL
                #     try:
                #         sql_json_answer = json.loads(sql_answer)
                #         print(f"{bcolors.OKCYAN}Keys in the SQL JSON Answer: {sql_json_answer.keys()}{bcolors.ENDC}")
                #         return_response["answer"] = sql_json_answer["answer"]
                #         return_response["similar_queries"] = sql_json_answer["similar_queries"]

                #         # ! Save the question to the FAQs database in Cosmos DB
                #         is_saved = COSMOS_DB().save_user_query_for_faqs(
                #             user_query      = user_details["query"],
                #             rephrased_query = user_question,
                #             # dashboard_name  = row_level_access_sql_query['application_name'],
                #             dashboard_name = generated_sql_query['application_name'],
                #         )
                #         print(f"{bcolors.OKGREEN}Is saved to FAQs: {is_saved}{bcolors.ENDC}")
                #     except Exception as e:
                #         tb = traceback.format_exc()
                #         print(f"Error from sql_answer: {str(e)}\nTraceback:\n{tb}")
                #         logger.log(msg= f"Error in sql_answer...{tb}", level=logging.ERROR)
                #         return_response["answer"] = sql_answer

                #     save_to_cosmos_db_for_faqs_end_time = time.time()
                #     print(f"{bcolors.OKGREEN}Saving to Cosmos DB for FAQs took {save_to_cosmos_db_for_faqs_end_time - natural_lang_and_graph_code_generation_end_time} seconds{bcolors.ENDC}")
                    
                #     ## ! Generate the graphs for the SQL query
                #     if df.shape[0]<2:
                #         print(f"{bcolors.FAIL}Dataframe shape: {df.shape} in less than 2{bcolors.ENDC}")
                #         return_response['table_markdown'] = ""
                #     elif "table" in user_question.lower() or "tabular" in user_question.lower():
                #         print(f"{bcolors.OKGREEN}Dataframe shape: {df.shape} in table{bcolors.ENDC}")
                #         return_response['table_markdown'] = df.to_markdown()
                #     elif df.shape[0] >= 2 and df.shape[1] >= 2:
                #         if graph_code is None:
                #             return_response["table_markdown"] = df.to_markdown()
                #             # TODO: Have to add the else condition so that after the if block, rest codes are not to be executed
                #         graph_code = graph_code.replace("`", "").replace("python", "").replace("fig.show()", "").strip()
                #         print(f"{bcolors.OKGREEN}Graph Code: {graph_code}{bcolors.ENDC}")
                #         try:
                #             if is_valid_python_code(graph_code):
                #                 nan = np.nan
                #                 local_namespace = {"df": df, "nan": nan}
                #                 exec(graph_code, {}, local_namespace)
                                
                #                 print(f"{bcolors.OKGREEN}All local variables: {local_namespace}{bcolors.ENDC}")
                #                 fig     = local_namespace['fig']
                #                 sas_url = upload_image_to_blob(fig.to_image(format="png"))
                #                 return_response['table_markdown'] = sas_url
                #             else:
                #                 print(f"Graph code is not executable")
                #         except Exception as e:
                #             tb = traceback.format_exc()
                #             print(f"{bcolors.FAIL}Error from Graph Code: {str(e)}\nTraceback:\n{tb}{bcolors.ENDC}")
                #             logger.log(msg= f"Error in graph code...{tb}", level=logging.ERROR)


                #             phrases_to_check = ['graph','plot','chart']
                #             phrases_to_check_bar = ['bar graph', 'bar chart', 'bar plot']
                #             phrases_to_check_scatter = ['scatter graph', 'scatter chart', 'scatter plot']
                #             phrases_to_check_line = ['line graph', 'line chart', 'line plot']
                #             phrases_to_check_pie = ['pie graph', 'pie chart', 'pie plot']

                #             # Create a regular expression pattern to match any of the phrases
                #             basic = '|'.join(re.escape(phrase) for phrase in phrases_to_check)
                #             bar = '|'.join(re.escape(phrase) for phrase in phrases_to_check_bar)
                #             scatter = '|'.join(re.escape(phrase) for phrase in phrases_to_check_scatter)
                #             line = '|'.join(re.escape(phrase) for phrase in phrases_to_check_line)
                #             pie = '|'.join(re.escape(phrase) for phrase in phrases_to_check_pie)

                #             print(f"{bcolors.OKGREEN}Dataframe shape: {df.shape} in greater than or equal 2{bcolors.ENDC}")
                #             # graph_type = graph_decide(graph_prompt)
                #             # print(f"{bcolors.OKGREEN}Graph Type: {graph_type}{bcolors.ENDC}")
                #             colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
                #             if re.search(basic, user_question, re.IGNORECASE):
                #                 # if "bar" in user_question.lower() or "bar" in graph_type:
                #                 if re.search(bar, user_question, re.IGNORECASE):
                #                     # Bar Plot
                #                     fig = px.bar(df,x=columns[0],y=columns[1],
                #                                 color = columns[0],
                #                                 text_auto = '.5s',
                #                                 title='Bar Plot ' + columns[1])
                #                     if df.shape[0] > 9:
                #                         fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35), showlegend=False)
                #                     else:
                #                         fig.update_layout(width=650, height=550)
                #                     # html_format = fig.to_html()
                #                     image_bytes = fig.to_image(format="png")
                #                 # Use the regular expression to search for the pie phrases in the user's question
                #                 # elif "pie" in user_question.lower() or"pie" in graph_type:
                #                 elif re.search(pie, user_question, re.IGNORECASE):
                #                     print(f"{bcolors.OKGREEN}Pie Graph columns{columns}{bcolors.ENDC}")
                #                     # fig = go.Figure(data=[go.Pie(labels=columns,
                #                     #                             values=columns[1])])
                #                     fig = px.pie(df, names=columns[0], values=columns[1])
                #                     fig.update_traces(textinfo='percent+label',textposition='inside',
                #                                     textfont_size=20,
                #                                     marker=dict(colors=colors,line=dict(color='#000000', width=2)))
                #                     if df.shape[0] > 9:
                #                         fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                #                     else:
                #                         fig.update_layout(width=650, height=550)
                #                     # html_format = fig.to_html()
                #                     image_bytes = fig.to_image(format="png")
                                    
                #                 # elif "scatter" in user_question.lower() or "scatter" in graph_type:
                #                 elif re.search(scatter, user_question, re.IGNORECASE):
                #                     fig = px.scatter(df,x=columns[0],y=columns[1],
                #                                     # color = columns[0],
                #                                     size= columns[1],
                #                                     title='Scatter Plot ' + columns[1])
                #                     if df.shape[0] > 9:
                #                         fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                #                     else:
                #                         fig.update_layout(width=650, height=550)
                #                     # html_format = fig.to_html()
                #                     image_bytes = fig.to_image(format="png")
                #                 # elif "line" in user_question.lower() or "line" in graph_type:
                #                 elif re.search(line, user_question, re.IGNORECASE):
                #                     fig = px.line(df,x=columns[0],y=columns[1],
                #                                 # color=columns[0],
                #                                 markers=True,
                #                                 title='Line Graph ' + columns[1],
                #                                 text=columns[1]
                #                             )
                #                     if df.shape[0] > 9:
                #                         fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                #                     else:
                #                         fig.update_layout(width=650, height=550)
                #                     # html_format = fig.to_html()
                #                     image_bytes = fig.to_image(format="png")

                #                 sas_graph_url = upload_image_to_blob(image_bytes)
                #                 return_response['table_markdown'] = sas_graph_url
                #             else:
                #                 return_response['table_markdown'] = df.to_markdown()
                #     else:
                #         print(f"{bcolors.FAIL}Dataframe shape: {df.shape} in else block.{bcolors.ENDC}")
                #         return_response['table_markdown'] = df.to_markdown()

                #     graph_generation_with_sas_url_generation_end_time = time.time()
                #     print(f"{bcolors.OKGREEN}Graph Generation and SAS URL Generation took {graph_generation_with_sas_url_generation_end_time - save_to_cosmos_db_for_faqs_end_time} seconds{bcolors.ENDC}")

                #     # return_response["citation"] = {
                #     #     "database_name": row_level_access_sql_query['database_name'],
                #     #     "table_name": row_level_access_sql_query['table_name'],
                #     #     "group_name": row_level_access_sql_query["group_name"],
                #     #     "application_name": row_level_access_sql_query["application_name"].title()
                #     # }
                #     return_response["citation"] = {
                #         "database_name": generated_sql_query['Database_Name'],
                #         "table_name": generated_sql_query['Table_Name'],
                #         "group_name": generated_sql_query["group_name"],
                #         "application_name": generated_sql_query["application_name"].title()
                #     }
                #     return_response["question_type"] = "sql"

                #     ## Save the SQL conversation to the CosmosDB
                #     item = {
                #         "rephrased_query": user_question,
                #         "actual_query": user_details["query"],
                #         "sql_query": sql_query,
                #         "answer": sql_answer,
                #         # "citation": {
                #         #     "database_name": row_level_access_sql_query['database_name'],
                #         #     "table_name": row_level_access_sql_query['table_name'],
                #         #     "group_name": row_level_access_sql_query["group_name"],
                #         #     "application_name": row_level_access_sql_query["application_name"].title()
                #         # },
                #         # "context": row_level_access_sql_query['context'],
                #         "citation": {
                #             "database_name": generated_sql_query['Database_Name'],
                #             "table_name": generated_sql_query['Table_Name'],
                #             "group_name": generated_sql_query["group_name"],
                #             "application_name": generated_sql_query["application_name"].title()
                #         },
                #         "context": generated_sql_query["context"]
                #     }

                #     is_saved = COSMOS_DB()._add_conversation_item(
                #         user_email = user_details["user_email"].lower(),
                #         conv_id    = user_details["conversation_id"],
                #         question_type = "sql",
                #         item = item
                #     )

                #     save_conversation_end_time = time.time()
                #     print(f"{bcolors.OKGREEN}Saving the conversation to Cosmos DB took {save_conversation_end_time - graph_generation_with_sas_url_generation_end_time} seconds{bcolors.ENDC}")
                #     if is_saved:
                #         print("Conversation saved successfully")
                #     else:
                #         print("Error saving conversation")

                #     return return_response

                # except ServerOperationError as e:
                #     tb = traceback.format_exc()
                #     print(f"Error from ServerOperationError: {str(e)}\nTraceback:\n{tb}")
                #     logger.log(msg= f"Error in server operation...{tb}", level=logging.ERROR)
                    
                #     if response == "others+sql":
                #         return_response["answer"] = "An internal server error occurred while processing your request. Please modify your query and try again."
                #         return_response['table_markdown'] = ""
                #         return_response["citation"] = ""
                #         return_response["question_type"] = "sql"
                #         return return_response
                    
                #     is_saved, return_response = generate_answers_for_pdf(
                #         user_question   = user_question,
                #         return_response = return_response,
                #         conv_id         = user_details["conversation_id"],
                #         actual_query    = user_details["query"],
                #         user_email      = user_details["user_email"].lower()
                #     )
                #     if is_saved:
                #         print("Conversation saved successfully")
                #     else:
                #         print("Error saving conversation")

                #     return return_response

                # except SqlNotGenerateError as e:
                #     tb = traceback.format_exc()
                #     print(f"Error from SqlNotGenerateError: {str(e)}\nTraceback:\n{tb}")
                #     logger.log(msg= f"Error in sql query not generated...{tb}", level=logging.ERROR)

                #     if response == "others+sql":
                #         return_response["answer"] = "An internal server error occurred while processing your request. Please modify your query and try again."
                #         return_response['table_markdown'] = ""
                #         return_response["citation"] = ""
                #         return_response["question_type"] = "sql"
                #         return return_response
                    
                    
                #     is_saved, return_response = generate_answers_for_pdf(
                #         user_question   = user_question,
                #         return_response = return_response,
                #         conv_id         = user_details["conversation_id"],
                #         actual_query    = user_details["query"],
                #         user_email      = user_details["user_email"].lower()
                #     )
                #     if is_saved:
                #         print("Conversation saved successfully")
                #     else:
                #         print("Error saving conversation")

                #     return return_response

                # except Exception as ex:
                #     tb = traceback.format_exc()
                #     print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")

                #     logger.log(msg= f"Error{tb}", level=logging.ERROR)

                #     if response == "others+sql":
                #         return_response["answer"] = "An internal server error occurred while processing your request. Please modify your query and try again."
                #         return_response['table_markdown'] = ""
                #         return_response["citation"] = ""
                #         return_response["question_type"] = "sql"
                #         return return_response
                
                #     return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
                #     return_response['table_markdown'] = ""
                #     return_response["citation"] = ""
                #     return_response["question_type"] = "sql"

                #     return return_response

            else:
                is_saved, return_response = generate_answers_for_pdf(
                    user_question   = user_question,
                    return_response = return_response,
                    conv_id         = user_details["conversation_id"],
                    actual_query    = user_details["query"],
                    user_email      = user_details["user_email"].lower()
                )

                print(f"{bcolors.FAIL}Response from the PDF: {return_response['answer']}{bcolors.ENDC}")
                if type(return_response['answer']) == dict and "this information is out of my uploaded knowledge base".lower() in return_response["answer"]['answer'].lower():
                    return handle_sql_query(user_details, user_question, return_response, response)
                elif "this information is out of my uploaded knowledge base".lower() in return_response["answer"].lower():
                    return handle_sql_query(user_details, user_question, return_response, response)
                
                if is_saved:
                    print("Conversation saved successfully")
                else:
                    print("Error saving conversation")

                return return_response
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error: {str(e)}\nTraceback:\n{tb}")
            logger.log(msg= f"Successfully generated greet message.. {tb}", level=logging.ERROR)
            return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
            return return_response
        
    else:
        try:
            if response == "application_access_list":
                user_apps = await get_user_app(user_details["user_email"])
                if user_apps == "No Record Found":
                    return_response["answer"] = "You do not have access to any Group or Dashboard"
                else:
                    return_response["answer"] = user_apps
            else:
                return_response["answer"] = response
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")
            logger.log(msg= f"Successfully generated greet message.. {tb}", level=logging.ERROR)
            return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."

        return return_response


@app.post("/delete-conversation")
def delete_conversation(request: DeleteItems):
    """
    Delete the conversation from the Cosmos DB
    """
    user_email = request.user_email.lower()
    is_deleted = COSMOS_DB().delete_items(user_email=user_email)
    if is_deleted:
        return {"message": "All the conversations are deleted successfully"}
    else:
        return {"message": "Error deleting the conversations"}
    
@app.post("/dashboard-faqs")
async def dashboard_faqs(request: DashboardFAQs):
    """
    Get the FAQs for the dashboard if user selects the dashboard from the dropdown menu
    """
    user_information = {
        "user_email": request.user_email.lower(),
        "dashboard_name": request.dashboard_name.lower()
    }

    print(f"Received user email: {user_information['user_email']} for dashboard FAQs")
    # TODO: Check if the user has access to the dashboard or not
    faqs = get_or_generate_faqs(dashboard_name = user_information["dashboard_name"].lower())

    return faqs

@app.post("/api/messages-new")
async def submit_query(request: QueryRequestNew):
    user_details = {
        "user_email": request.user_email.lower(),
        "conversation_id": request.conversation_id,
        "query": request.query,
        "user_full_name": request.user_full_name,
        "dashboard_name": request.dashboard_name
    }

    print(f"Received user query: {user_details['query']}")
    return_response = {
        "user_question": user_details["query"],
    }

    start_time_for_greetings = time.time()
    # reduce the dependency by greetings message or sql query type selection
    if "access list" in user_details["query"].lower():
        # TODO: Get the list of applications the user has access to
        response = "application_access_list"
        pass
    elif "graph" in user_details["query"].lower() or "chart" in user_details["query"].lower() or "plot" in user_details["query"].lower():
        # TODO: Direct go to the SQL code blocks
        response = "others+sql"
    # TODO: Or if we can find any dashboard name to the user query, then it also has to be considered as SQL query
    else:
        response = None
    

    # If we don't have any response type selected then generate the response by using GPT
    if response is None:
        try:
            response    = await MyAzureOpenAI().generate_greet_messages(user_details["query"], user_details["user_full_name"])
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")
            logger.log(msg= f"Successfully generated greet message.. {tb}", level=logging.ERROR)
            return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
            return return_response

    print(f"{bcolors.FAIL}Greetings response: {response} and took {time.time() - start_time_for_greetings} seconds{bcolors.ENDC}")
    if response == "others" or response == "others+sql":
        # ? Fetch the last conversations and generate the rephrased query
        rephrased_query_time = time.time()
        history = COSMOS_DB().get_last_conversation(user_details["user_email"].lower())
        # print(f"{bcolors.FAIL}Last conversation: {history}{bcolors.ENDC}")

        user_question = process_user_question(
            user_question = user_details["query"],
            conversations = history,
        )

        rephrased_query_end_time = time.time()
        print(f"{bcolors.OKGREEN}Rephrased user question: {user_question} and took {rephrased_query_end_time - rephrased_query_time} seconds {bcolors.ENDC}")

        kernel   = sk.Kernel()
        kernel.add_chat_service(
            "chat_completion",
            AzureChatCompletion(deployment_name="gpt-35-turbo-16k", endpoint = os.getenv("openaiendpoint"), api_key = os.getenv("openaikey1")),
        )
        skills_directory     = "./az_semantic_kernel/skills"
        summarize_skill      = kernel.import_semantic_plugin_from_directory(skills_directory, "pdf")
        writer_skill         = kernel.import_semantic_plugin_from_directory(skills_directory, "sql")
        classification_skill = kernel.import_semantic_plugin_from_directory(skills_directory, "classification")
        
        planner         = BasicPlanner()
        basic_plan      = await planner.create_plan(user_question, kernel, system_prompt)
        input_json_plan = eval(basic_plan.generated_plan["input"])
        return_response["rephrased_query"] = user_question
        return_response["sk_response"]     = input_json_plan
        print(f"{bcolors.OKGREEN}Semantic Kernel took {time.time() - rephrased_query_end_time} seconds{bcolors.ENDC}")
        try:
            if input_json_plan["subtasks"][1]['function'] == "pdf.pdf_skill":
                is_saved, return_response = generate_answers_for_pdf(
                    user_question   = user_question,
                    return_response = return_response,
                    conv_id         = user_details["conversation_id"],
                    actual_query    = user_details["query"],
                    user_email      = user_details["user_email"].lower()
                )
                if is_saved:
                    print("Conversation saved successfully")
                else:
                    print("Error saving conversation")
                
                return return_response
                
            elif input_json_plan["subtasks"][1]['function'] == "sql.sql_skill":
                try:
                    # generated_sql_query    = generate_sql_first_level_access(user_question, user_details["user_email"])
                    # return_response["sql_query"]                  = generated_sql_query['sql_query']

                    # print(f"{bcolors.FAIL}Generated SQL Query from the old prompt: {generated_sql_query['sql_query']}{bcolors.ENDC}")

                    row_level_sql_query_start_time = time.time()
                    row_level_access_sql_query = generate_sql_with_row_column_level_access(
                        user_question = user_question,
                        user_email    = user_details["user_email"].lower(),
                        dashboard_name = user_details["dashboard_name"]
                    )
                    print(f"{bcolors.FAIL}Row level access all details: {row_level_access_sql_query.keys()}{bcolors.ENDC}")
                    sql_query = row_level_access_sql_query["sql_query"]
                    # sql_query = generated_sql_query['sql_query']
                    # return_response["row_level_access_sql_query"] = sql_query
                    print(f"{bcolors.FAIL}Generated SQL Query from the new prompt: {row_level_access_sql_query['sql_query']}{bcolors.ENDC}")
                    return_response["sql_query"] = sql_query

                    if sql_query.lower().replace(".", "") == "NO SQL QUERY CAN BE GENERATED".lower():
                        print(f"""{bcolors.FAIL}No SQL query can be generated{bcolors.ENDC}""")
                        raise SqlNotGenerateError("No SQL query can be generated")
                    elif sql_query.lower().replace(".", "") == "User has no access to browse this type of data".lower():
                        print(f"""{bcolors.FAIL}User has no access to browse this type of data{bcolors.ENDC}""")
                        return_response["answer"] = "Access Denied: You currently lack the necessary permissions to view this data. Please contact your administrator for further assistance."
                        return_response["citation"]      = ""
                        return_response["question_type"] = "sql"
                        return return_response
                    
                    row_level_sql_query_end_time = time.time()
                    print(f"{bcolors.OKGREEN}Row Level SQL Query: {sql_query} and took {row_level_sql_query_end_time - row_level_sql_query_start_time} seconds {bcolors.ENDC}")


                    # ! Execute the SQL query and fetch the results
                    with sql.connect(
                        server_hostname = os.getenv("dbserverhostname"),
                        http_path       = os.getenv("dbhttppath"),
                        access_token    = os.getenv("dbaccesstoken")
                    ) as connection:
                        with connection.cursor() as cursor:
                            cursor.execute("SET spark.sql.legacy.timeParserPolicy = LEGACY")
                            # cursor.execute("SELECT * FROM hello;")
                            cursor.execute(sql_query)
                            result = cursor.fetchall()
                            if result is None:
                                print("***No result***")
                            else:
                                print("****Fetching result****")

                    df         = pd.DataFrame([r.asDict() for r in result])
                    columns = df.columns.tolist()
                    data = df.head(5)
                    # graph_prompt = prompt_graph.format(user_question = user_question, c = columns, data = data)

                    data_fetch_from_databricks_end_time = time.time()
                    print(f"{bcolors.OKGREEN}Dataframe shape before generating answer: {df.shape} and took {data_fetch_from_databricks_end_time - row_level_sql_query_end_time} seconds.{bcolors.ENDC}")
                    # sql_answer = MyAzureOpenAI().dataframe_openai(user_question, df.to_markdown())

                    # TODO: Execute the normal language conversation code and graph generation code in parallel
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Create an instance of MyAzureOpenAI
                        azure_openai_instance = MyAzureOpenAI()

                        if df.shape[0] >= 2 and df.shape[1] >= 2:
                            # Submit the methods to the executor
                            future1 = executor.submit(azure_openai_instance.dataframe_openai, user_question, df.to_markdown())
                            future2 = executor.submit(azure_openai_instance.generate_graph_code, df, user_question)

                            # Wait for both futures to complete
                            concurrent.futures.wait([future1, future2])

                            # Get the results from the futures
                            sql_answer = future1.result()
                            graph_code = future2.result()
                        else:
                            future1 = executor.submit(azure_openai_instance.dataframe_openai, user_question, df.to_markdown())
                            concurrent.futures.wait([future1])
                            sql_answer = future1.result()
                            graph_code = None


                    natural_lang_and_graph_code_generation_end_time = time.time()
                    print(f"{bcolors.OKGREEN}Natural Language and Graph Code Generation took {natural_lang_and_graph_code_generation_end_time - data_fetch_from_databricks_end_time} seconds{bcolors.ENDC}")
                    
                    # setup the response format for the Bot & save the conversation to the CosmosDB for the SQL
                    try:
                        sql_json_answer = json.loads(sql_answer)
                        print(f"Keys in the SQL JSON Answer: {sql_json_answer.keys()}")
                        return_response["answer"] = sql_json_answer["answer"]
                        return_response["similar_queries"] = sql_json_answer["similar_queries"]

                        # ! Save the question to the FAQs database in Cosmos DB
                        is_saved = COSMOS_DB().save_user_query_for_faqs(
                            user_query      = user_details["query"],
                            rephrased_query = user_question,
                            dashboard_name  = row_level_access_sql_query['application_name'],
                        )
                        print(f"{bcolors.OKGREEN}Is saved to FAQs: {is_saved}{bcolors.ENDC}")
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Error from sql_answer: {str(e)}\nTraceback:\n{tb}")
                        logger.log(msg= f"Error in sql_answer...{tb}", level=logging.ERROR)
                        return_response["answer"] = sql_answer

                    save_to_cosmos_db_for_faqs_end_time = time.time()
                    print(f"{bcolors.OKGREEN}Saving to Cosmos DB for FAQs took {save_to_cosmos_db_for_faqs_end_time - natural_lang_and_graph_code_generation_end_time} seconds{bcolors.ENDC}")
                    
                    ## ! Generate the graphs for the SQL query
                    if df.shape[0]<2:
                        print(f"{bcolors.FAIL}Dataframe shape: {df.shape} in less than 2{bcolors.ENDC}")
                        return_response['table_markdown'] = ""
                    elif "table" in user_question.lower() or "tabular" in user_question.lower():
                        print(f"{bcolors.OKGREEN}Dataframe shape: {df.shape} in table{bcolors.ENDC}")
                        return_response['table_markdown'] = df.to_markdown()
                    elif df.shape[0] >= 2 and df.shape[1] >= 2:
                        if graph_code is None:
                            return_response["table_markdown"] = df.to_markdown()
                            # TODO: Have to add the else condition so that after the if block, rest codes are not to be executed
                        graph_code = graph_code.replace("`", "").replace("python", "").replace("fig.show()", "").strip()
                        print(f"{bcolors.OKGREEN}Graph Code: {graph_code}{bcolors.ENDC}")
                        try:
                            if is_valid_python_code(graph_code):
                                nan = np.nan
                                local_namespace = {"df": df, "nan": nan}
                                exec(graph_code, {}, local_namespace)
                                
                                print(f"{bcolors.OKGREEN}All local variables: {local_namespace}{bcolors.ENDC}")
                                fig     = local_namespace['fig']
                                sas_url = upload_image_to_blob(fig.to_image(format="png"))
                                return_response['table_markdown'] = sas_url
                            else:
                                print(f"Graph code is not executable")
                        except Exception as e:
                            tb = traceback.format_exc()
                            print(f"{bcolors.FAIL}Error from Graph Code: {str(e)}\nTraceback:\n{tb}{bcolors.ENDC}")


                            phrases_to_check = ['graph','plot','chart']
                            phrases_to_check_bar = ['bar graph', 'bar chart', 'bar plot']
                            phrases_to_check_scatter = ['scatter graph', 'scatter chart', 'scatter plot']
                            phrases_to_check_line = ['line graph', 'line chart', 'line plot']
                            phrases_to_check_pie = ['pie graph', 'pie chart', 'pie plot']

                            # Create a regular expression pattern to match any of the phrases
                            basic = '|'.join(re.escape(phrase) for phrase in phrases_to_check)
                            bar = '|'.join(re.escape(phrase) for phrase in phrases_to_check_bar)
                            scatter = '|'.join(re.escape(phrase) for phrase in phrases_to_check_scatter)
                            line = '|'.join(re.escape(phrase) for phrase in phrases_to_check_line)
                            pie = '|'.join(re.escape(phrase) for phrase in phrases_to_check_pie)

                            print(f"{bcolors.OKGREEN}Dataframe shape: {df.shape} in greater than or equal 2{bcolors.ENDC}")
                            # graph_type = graph_decide(graph_prompt)
                            # print(f"{bcolors.OKGREEN}Graph Type: {graph_type}{bcolors.ENDC}")
                            colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
                            if re.search(basic, user_question, re.IGNORECASE):
                                # if "bar" in user_question.lower() or "bar" in graph_type:
                                if re.search(bar, user_question, re.IGNORECASE):
                                    # Bar Plot
                                    fig = px.bar(df,x=columns[0],y=columns[1],
                                                color = columns[0],
                                                text_auto = '.5s',
                                                title='Bar Plot ' + columns[1])
                                    if df.shape[0] > 9:
                                        fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35), showlegend=False)
                                    else:
                                        fig.update_layout(width=650, height=550)
                                    # html_format = fig.to_html()
                                    image_bytes = fig.to_image(format="png")
                                # Use the regular expression to search for the pie phrases in the user's question
                                # elif "pie" in user_question.lower() or"pie" in graph_type:
                                elif re.search(pie, user_question, re.IGNORECASE):
                                    print(f"{bcolors.OKGREEN}Pie Graph columns{columns}{bcolors.ENDC}")
                                    # fig = go.Figure(data=[go.Pie(labels=columns,
                                    #                             values=columns[1])])
                                    fig = px.pie(df, names=columns[0], values=columns[1])
                                    fig.update_traces(textinfo='percent+label',textposition='inside',
                                                    textfont_size=20,
                                                    marker=dict(colors=colors,line=dict(color='#000000', width=2)))
                                    if df.shape[0] > 9:
                                        fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                                    else:
                                        fig.update_layout(width=650, height=550)
                                    # html_format = fig.to_html()
                                    image_bytes = fig.to_image(format="png")
                                    
                                # elif "scatter" in user_question.lower() or "scatter" in graph_type:
                                elif re.search(scatter, user_question, re.IGNORECASE):
                                    fig = px.scatter(df,x=columns[0],y=columns[1],
                                                    # color = columns[0],
                                                    size= columns[1],
                                                    title='Scatter Plot ' + columns[1])
                                    if df.shape[0] > 9:
                                        fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                                    else:
                                        fig.update_layout(width=650, height=550)
                                    # html_format = fig.to_html()
                                    image_bytes = fig.to_image(format="png")
                                # elif "line" in user_question.lower() or "line" in graph_type:
                                elif re.search(line, user_question, re.IGNORECASE):
                                    fig = px.line(df,x=columns[0],y=columns[1],
                                                # color=columns[0],
                                                markers=True,
                                                title='Line Graph ' + columns[1],
                                                text=columns[1]
                                            )
                                    if df.shape[0] > 9:
                                        fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                                    else:
                                        fig.update_layout(width=650, height=550)
                                    # html_format = fig.to_html()
                                    image_bytes = fig.to_image(format="png")

                                sas_graph_url = upload_image_to_blob(image_bytes)
                                return_response['table_markdown'] = sas_graph_url
                            else:
                                return_response['table_markdown'] = df.to_markdown()
                    else:
                        print(f"{bcolors.FAIL}Dataframe shape: {df.shape} in else block.{bcolors.ENDC}")
                        return_response['table_markdown'] = df.to_markdown()

                    graph_generation_with_sas_url_generation_end_time = time.time()
                    print(f"{bcolors.OKGREEN}Graph Generation and SAS URL Generation took {graph_generation_with_sas_url_generation_end_time - save_to_cosmos_db_for_faqs_end_time} seconds{bcolors.ENDC}")

                    return_response["citation"] = {
                        "database_name": row_level_access_sql_query['database_name'],
                        "table_name": row_level_access_sql_query['table_name'],
                        "group_name": row_level_access_sql_query["group_name"],
                        "application_name": row_level_access_sql_query["application_name"].title()
                    }
                    return_response["question_type"] = "sql"

                    ## Save the SQL conversation to the CosmosDB
                    item = {
                        "rephrased_query": user_question,
                        "actual_query": user_details["query"],
                        "sql_query": sql_query,
                        "answer": sql_answer,
                        "citation": {
                            "database_name": row_level_access_sql_query['database_name'],
                            "table_name": row_level_access_sql_query['table_name'],
                            "group_name": row_level_access_sql_query["group_name"],
                            "application_name": row_level_access_sql_query["application_name"].title()
                        },
                        "context": row_level_access_sql_query['context'],
                    }

                    is_saved = COSMOS_DB()._add_conversation_item(
                        user_email = user_details["user_email"].lower(),
                        conv_id    = user_details["conversation_id"],
                        question_type = "sql",
                        item = item
                    )

                    save_conversation_end_time = time.time()
                    print(f"{bcolors.OKGREEN}Saving the conversation to Cosmos DB took {save_conversation_end_time - graph_generation_with_sas_url_generation_end_time} seconds{bcolors.ENDC}")
                    if is_saved:
                        print("Conversation saved successfully")
                    else:
                        print("Error saving conversation")

                    return return_response

                except ServerOperationError as e:
                    tb = traceback.format_exc()
                    print(f"Error from ServerOperationError: {str(e)}\nTraceback:\n{tb}")
                    logger.log(msg= f"Error in server operation...{tb}", level=logging.ERROR)
                    is_saved, return_response = generate_answers_for_pdf(
                        user_question   = user_question,
                        return_response = return_response,
                        conv_id         = user_details["conversation_id"],
                        actual_query    = user_details["query"],
                        user_email      = user_details["user_email"].lower()
                    )
                    if is_saved:
                        print("Conversation saved successfully")
                    else:
                        print("Error saving conversation")

                    return return_response

                except SqlNotGenerateError as e:
                    tb = traceback.format_exc()
                    print(f"Error from SqlNotGenerateError: {str(e)}\nTraceback:\n{tb}")
                    logger.log(msg= f"Error in sql query not generated...{tb}", level=logging.ERROR)
                    is_saved, return_response = generate_answers_for_pdf(
                        user_question   = user_question,
                        return_response = return_response,
                        conv_id         = user_details["conversation_id"],
                        actual_query    = user_details["query"],
                        user_email      = user_details["user_email"].lower()
                    )
                    if is_saved:
                        print("Conversation saved successfully")
                    else:
                        print("Error saving conversation")

                    return return_response

                except Exception as ex:
                    tb = traceback.format_exc()
                    print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")
                    logger.log(msg= f"Error{tb}", level=logging.ERROR)
                    return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
                    return_response['table_markdown'] = ""
                    return_response["citation"] = ""
                    return_response["question_type"] = "sql"

                    return return_response

            else:
                is_saved, return_response = generate_answers_for_pdf(
                    user_question   = user_question,
                    return_response = return_response,
                    conv_id         = user_details["conversation_id"],
                    actual_query    = user_details["query"],
                    user_email      = user_details["user_email"].lower()
                )
                if is_saved:
                    print("Conversation saved successfully")
                else:
                    print("Error saving conversation")

                return return_response
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error: {str(e)}\nTraceback:\n{tb}")
            return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
            print(f"Error: {str(e)}\nTraceback:\n{tb}")
            return return_response
        
    else:
        try:
            if response == "application_access_list":
                user_apps = await get_user_app(user_details["user_email"])
                if user_apps == "No Record Found":
                    return_response["answer"] = "You do not have access to any Group or Dashboard"
                else:
                    return_response["answer"] = user_apps
            else:
                return_response["answer"] = response
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")
            return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."

        return return_response



@app.post("/query-answers")
async def query_answers(request: QueryAnswer, response: Response):
    user_details = {
        "user_email": request.user_email.lower(),
        "conversation_id": request.conversation_id,
        "query": request.query,
        "user_full_name": request.user_full_name,
        "query_type": request.query_type,
        # "dashboard_name": request.dashboard_name.lower() if request.dashboard_name else ""
    }

    response    = await MyAzureOpenAI().generate_greet_messages(user_details["query"], user_details["user_full_name"])

    return_response = {
        "user_question": user_details["query"],
    }
    if response == "others":

        #################################### Rephrase the user question ####################################
        print(f"{bcolors.OKGREEN}Received user query: {user_details['query']} & email address: {user_details['user_email']}{bcolors.ENDC}")
        history = COSMOS_DB().get_last_conversation(user_details["user_email"].lower())
        # print(f"{bcolors.FAIL}Last conversation: {history}{bcolors.ENDC}")

        user_question = process_user_question(
            user_question = user_details["query"],
            conversations = history,
        )
        print(f"{bcolors.OKGREEN}Rephrased user question: {user_question}{bcolors.ENDC}")
        return_response["rephrased_query"] = user_question
        #################################### For the PDF ####################################
        if user_details['query_type'] == 'pdf':
            is_saved, return_response = generate_answers_for_pdf(
                user_question   = user_question,
                return_response = return_response,
                conv_id         = user_details["conversation_id"],
                actual_query    = user_details["query"],
                user_email      = user_details["user_email"].lower()
            )
            if is_saved:
                print("Conversation saved successfully")
            else:
                print("Error saving conversation")
            
            return return_response
        
        #################################### For the SQL ####################################
        elif user_details['query_type'] == 'sql':
            print(f"{bcolors.OKGREEN}user's query = {user_question}{bcolors.ENDC}")
            try:
                generated_sql_query    = generate_sql_first_level_access(user_question, user_details["user_email"])

                print(f"{bcolors.FAIL}Generated SQL Query: {generated_sql_query['sql_query']}{bcolors.ENDC}")

                if generated_sql_query['sql_query'].lower().replace(".", "") == "NO SQL QUERY CAN BE GENERATED".lower():
                    print(f"""{bcolors.FAIL}No SQL query can be generated{bcolors.ENDC}""")
                    # raise SqlNotGenerateError("No SQL query can be generated")
                    return_response["answer"]        = "No SQL query can be generated for this question. Please try again with a different question."
                    return_response["citation"]      = ""
                    return_response["question_type"] = "sql"
                    return return_response
                elif generated_sql_query['sql_query'].lower().replace(".", "") == "User has no access to browse this type of data".lower():
                    print(f"""{bcolors.FAIL}User has no access to browse this type of data{bcolors.ENDC}""")
                    return_response["answer"] = "Access Denied: You currently lack the necessary permissions to view this data. Please contact your administrator for further assistance."
                    return_response["citation"]      = ""
                    return_response["question_type"] = "sql"
                    return return_response

                with sql.connect(
                    server_hostname = os.getenv("dbserverhostname"),
                    http_path       = os.getenv("dbhttppath"),
                    access_token    = os.getenv("dbaccesstoken")
                ) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute("SET spark.sql.legacy.timeParserPolicy = LEGACY")
                        # cursor.execute("SELECT * FROM hello;")
                        cursor.execute(generated_sql_query['sql_query'])
                        result = cursor.fetchall()
                        if result is None:
                            print("***No result***")
                        else:
                            print("****Fetching result****")

                df         = pd.DataFrame([r.asDict() for r in result])
                columns = df.columns.tolist()
                data = df.head(5)
                graph_prompt = prompt_graph.format(c = columns, data = data)
                sql_answer = MyAzureOpenAI().dataframe_openai(user_question, df)

                # setup the response format for the Bot
                return_response["answer"] = sql_answer
                return_response["sql_query"] = generated_sql_query['sql_query']
                
                if df.shape[0] < 2:
                    print(f"{bcolors.FAIL}Dataframe shape: {df.shape} in less than 2{bcolors.ENDC}")
                    return_response['table_markdown'] = ""
                elif df.shape[0]>=2 and df.shape[1]==2:
                    print(f"{bcolors.OKGREEN}Dataframe shape: {df.shape} in greater than or equal 2{bcolors.ENDC}")
                    graph_type = graph_decide(graph_prompt)
                    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
                    if "bar" in graph_type:
                        # Bar Plot
                        fig = px.bar(df,x=columns[0],y=columns[1],
                                    color = columns[0],
                                    text_auto = '.5s',
                                    title='Bar Plot ' + columns[1])
                        if df.shape[0] > 9:
                            fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35), showlegend=False)
                        else:
                            fig.update_layout(width=650, height=550)
                        # html_format = fig.to_html()
                        image_bytes = fig.to_image(format="png")
                    # Use the regular expression to search for the pie phrases in the user's question
                    elif "pie" in graph_type:
                        fig = go.Figure(data=[go.Pie(labels=columns[0],
                                                    values=columns[1])])
                        fig.update_traces(textinfo='percent+label',textposition='inside',
                                        textfont_size=20,
                                        marker=dict(colors=colors,line=dict(color='#000000', width=2)))
                        if df.shape[0] > 9:
                            fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                        else:
                            fig.update_layout(width=650, height=550)
                        # html_format = fig.to_html()
                        image_bytes = fig.to_image(format="png")
                        
                    elif "scatter" in graph_type:
                        fig = px.scatter(df,x=columns[0],y=columns[1],
                                        color = columns[0],
                                        size= columns[1],
                                        title='Scatter Plot ' + columns[1])
                        if df.shape[0] > 9:
                            fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                        else:
                            fig.update_layout(width=650, height=550)
                        # html_format = fig.to_html()
                        image_bytes = fig.to_image(format="png")
                    elif "line" in graph_type:
                        fig = px.line(df,x=columns[0],y=columns[1],
                                    color=columns[0],
                                    markers=True,
                                    title='Line Graph ' + columns[1],
                                    text=columns[1])
                        if df.shape[0] > 9:
                            fig.update_layout(width=(df.shape[0]*55), height=(df.shape[0]*35))
                        else:
                            fig.update_layout(width=650, height=550)
                        # html_format = fig.to_html()
                        image_bytes = fig.to_image(format="png")

                    sas_graph_url = upload_image_to_blob(image_bytes)
                    return_response['table_markdown'] = sas_graph_url
                else:
                    print(f"{bcolors.FAIL}Dataframe shape: {df.shape} in else block.{bcolors.ENDC}")
                    return_response['table_markdown'] = df.to_markdown()

                return_response["citation"] = {
                    "database_name":    generated_sql_query['Database_Name'],
                    "table_name":       generated_sql_query['Table_Name'],
                    "group_name":       generated_sql_query["group_name"],
                    "application_name": generated_sql_query["application_name"].title()
                }
                return_response["question_type"] = "sql"

                ## Save the SQL conversation to the CosmosDB
                item = {
                    "rephrased_query": user_question,
                    "actual_query": user_details["query"],
                    "sql_query": generated_sql_query['sql_query'],
                    "answer": sql_answer,
                    "citation": {
                        "database_name": generated_sql_query['Database_Name'],
                        "table_name": generated_sql_query['Table_Name'],
                        "group_name": generated_sql_query["group_name"],
                        "application_name": generated_sql_query["application_name"].title()
                    },
                    "context": generated_sql_query['context'],
                }
                is_saved = COSMOS_DB()._add_conversation_item(
                    user_email = user_details["user_email"].lower(),
                    conv_id    = user_details["conversation_id"],
                    question_type = "sql",
                    item = item
                )
                if is_saved:
                    print("Conversation saved successfully")
                else:
                    print("Error saving conversation")

                return return_response

            except ServerOperationError as e:
                tb = traceback.format_exc()
                print(f"Error from ServerOperationError: {str(e)}\nTraceback:\n{tb}")
                return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."

                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                return return_response

            except Exception as ex:
                tb = traceback.format_exc()
                print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")  
                return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."

                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                return return_response
        
        #################################### For others ####################################
        else:
            return_response["answer"] = "Please enter a valid query type among 'pdf' or 'sql'"
            response.status_code      = status.HTTP_400_BAD_REQUEST
            return return_response
    else:
        try:
            if response == "application_access_list":
                user_apps = await get_user_app(user_details["user_email"])
                if user_apps == "No Record Found":
                    return_response["answer"] = "You do not have access to any Group or Dashboard"
                else:
                    all_dashboards = []
                    for group in user_apps:
                        for key in group.keys():
                            all_dashboards.extend(group[key])
                    
                    print(len(all_dashboards))
                    all_dashboards = [key for group in user_apps for key in group.keys()]
                    print(f"{bcolors.OKGREEN}All Dashboards: {all_dashboards}{bcolors.ENDC} & len = {len(all_dashboards)}")
                    return_response["answer"] = user_apps
            else:
                return_response["answer"] = response
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")
            return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."

        return return_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive = 1500, workers = 10)