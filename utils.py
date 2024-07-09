import os
import re
import ast
import time
import json
import random
import logging
import traceback

import pandas as pd
from databricks import sql
from collections import OrderedDict

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from openai import AzureOpenAI
from sklearn.cluster import KMeans
from azure_logging.azure_logging import logger
from azure_openai.azure_openai import MyAzureOpenAI
from azure_cosmos_db.azure_cosmos_db import COSMOS_DB
from sklearn.feature_extraction.text import TfidfVectorizer
from az_blob_services.az_blob_services import document_blob_url
from az_cognitive_search.az_cognitive_search import AZCognitiveSearch

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

async def rbac_mapping():
    # Define your SQL query to select data from the table
    sql_query = "SELECT * FROM hive_metastore.default.rbac_mapping"

    # Establish SQL connection to Databricks
    # ! TODO: Have to remove the hardcoded credentials
    with sql.connect(
        server_hostname = os.getenv("dbserverhostname"),
        http_path       = os.getenv("dbhttppath"),
        access_token    = os.getenv("dbaccesstoken")
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET spark.sql.legacy.timeParserPolicy = LEGACY")
            cursor.execute(sql_query)
            result = cursor.fetchall()
            if result is None:
                print("****No result****")
            else:
                print("****Fetching Data From RBAC Mapping****")
                # Convert the result to a DataFrame
                df = pd.DataFrame([r.asDict() for r in result])
    return df

async def rbac_groups():
    # Define your SQL query to select data from the table
    sql_query = "SELECT * FROM hive_metastore.default.groups"

    # Establish SQL connection to Databricks
    # ! TODO: Have to remove the hardcoded credentials
    with sql.connect(
        server_hostname = os.getenv("dbserverhostname"),
        http_path       = os.getenv("dbhttppath"),
        access_token    = os.getenv("dbaccesstoken")
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET spark.sql.legacy.timeParserPolicy = LEGACY")
            cursor.execute(sql_query)
            result = cursor.fetchall()
            if result is None:
                print("***No result***")
            else:
                print("****Fetching Data From RBAC Groups****")
                # Convert the result to a DataFrame
                df = pd.DataFrame([r.asDict() for r in result])
    return df

async def get_user_groups_details() -> list[dict]:
    df1 = await rbac_mapping()
    df2 = await rbac_groups()

    df1 = df1.to_dict(orient='records')
    df2 = df2.to_dict(orient='records')

    # Initialize a dictionary to store user details
    user_grp_details = {}

    for row in df2:
        user_email = row['User Email'].lower()
        group_name = row['Group Name']
        
        # Initialize a dictionary to store app names for each group
        if user_email not in user_grp_details:
            user_grp_details[user_email] = {group_name: []}
        elif group_name not in user_grp_details[user_email]:
            user_grp_details[user_email][group_name] = []

        # Iterate over data from table 1 to find corresponding user data
        for data in df1:
            if data['User Email'].lower() == row['User Email'].lower() and data['Group Name'] == group_name:
                app_name = data['App Name']
                user_grp_details[user_email][group_name].append(app_name)

    # Convert the dictionary to the desired format
    user_grp_details = [{user: [{group: user_grp_details[user][group]} for group in user_grp_details[user]]} for user in user_grp_details]
    return user_grp_details

async def get_user_app(user_email: str):
    start = time.time()
    user_grp_details = await get_user_groups_details()
    for user_data in user_grp_details:
        for user, groups in user_data.items():
            if user.strip().replace("gov.ae", "ac.ae") == user_email.strip():
                return groups
    end = time.time()
    logger.log(msg= f"Successfully Fetched welcome access card..{end-start}",level=logging.INFO)
    return "No Record Found"


def process_user_question(user_question: str, conversations: dict) -> str:
    start = time.time()
    history = f"User: {conversations['question']}\nBot: {conversations['answer']}"

    openai_obj = MyAzureOpenAI()
    followup   = openai_obj.check_followup(user_question, history)

    print(f"{bcolors.FAIL}Followup response from GPT: {followup}{bcolors.ENDC}")

    if followup.lower() == "True".lower():
        final_query = openai_obj.followup_query_modified(user_question, history)
    else:
        final_query = user_question
    end = time.time()
    print(f"{bcolors.FAIL}Final Query: {final_query}{bcolors.ENDC}")
    logger.log(msg= f"Successfully processed User Query..{end-start}",level=logging.INFO)

    return final_query


def find_answer_from_pdf(user_question: str) -> dict:
    try:
        start = time.time()
        results = AZCognitiveSearch().azure_vector_search(
            final_query = user_question,
            select      = ["document_name", "content", "page_no"],
            flag        = "pdf"
        )
        context = [{
            'relevant content': result['content'], 
            'document_name':result['document_name'],
            'page_number':result['page_no']
        } for result in results]

        answer = MyAzureOpenAI().doc_openai(context, user_question)
        answer = eval(answer)
        answer["context"] = context
        end = time.time()
        logger.log(msg= f"Successfully find unstructured data ..{end-start}",level=logging.INFO)
        return answer
    except Exception as ex:
        tb = traceback.format_exc()
        logger.log(msg= f"Error in finding unstructured data ..{ex}\n{tb}",level=logging.ERROR)
        return {"error": str(ex), "answer": answer}


def generate_sql_first_level_access(user_question: str, user_email: str, dashboard_name: str) -> dict:
    start = time.time()
    # TODO: Generate SQL query with the metadata dashboards
    dashboard_lists = ["graduates and alumni tracking dashboard", "schools students satisfaction", "post-secondary monthly snapshot", "schools dashboard", "emsat analysis", "eee program"]
    if dashboard_name.lower() in dashboard_lists:
        print(f"{bcolors.FAIL}Dashboard Name for the metadata section: {dashboard_name}{bcolors.ENDC}")
        # find the relations between tables for the dashboard
        with open("./Data/dashboard_tables_relations.json", "r") as f:
            dashboard_tables_relations = json.load(f)

        if dashboard_name.lower() in dashboard_tables_relations.keys():
            tables_relations = dashboard_tables_relations[dashboard_name.lower()]
        else:
            tables_relations = {}
        print(f"{bcolors.FAIL}Tables relations for the metadata dashboard: {tables_relations.keys()}{bcolors.ENDC}")

        # find the metadata for the dashboards
        with open("./Data/dashboard_metadata.json", "r", encoding = "UTF-8") as f:
            dashboard_metadata = json.load(f)

        if dashboard_name.lower() in dashboard_metadata.keys():
            selected_tables_columns = dashboard_metadata[dashboard_name.lower()]
        else:
            selected_tables_columns = ""
        print(f"{bcolors.FAIL}Selected tables and columns for the metadata dashboard: {selected_tables_columns.keys()}{bcolors.ENDC}")

        # find the dashboard wise custom rules
        with open("./Data/dashboard_extras.json", "r") as f:
            dashboard_rules = json.load(f)

        if dashboard_name.lower() in dashboard_rules.keys():
            rules_json = dashboard_rules[dashboard_name.lower()]
        else:
            rules_json = []
        print(f"{bcolors.FAIL}Rules for the metadata dashboard: {rules_json}{bcolors.ENDC}")

        # pass the details and generate the SQL query
        response = MyAzureOpenAI().sql_openai_with_metadata(
            query = user_question,
            selected_tables_columns = selected_tables_columns,
            tables_relations        = tables_relations,
            rules_json              = rules_json
        )

        print(f"{bcolors.FAIL}Generated SQL query for the metadata dashboard: {response}{bcolors.ENDC}")

        if response["Query"].lower() != "Please rephrase the question".lower():
            return {"sql_query": response["Query"], "Database_Name": "actvet_uc.gold", "Table_Name": "", "group_name": "", "application_name": dashboard_name.title(), "context": ""}
        else:
            return {"sql_query": "NO SQL QUERY CAN BE GENERATED", "Database_Name": "", "Table_Name": "", "group_name": "", "application_name": "", "context": ""}
    else:
        print(f"{bcolors.FAIL}Dashboard not found for the metadata pipeline: {dashboard_name}{bcolors.ENDC}")


    if dashboard_name == "" or dashboard_name is None:
        dashboard_name = check_dashboard_name_in_query(user_question)

    select_cols = ["head_content", "table_name", "database", "group_name", "application_name", "users", "columns"]
    
    if len(dashboard_name) > 0:
        tables      = AZCognitiveSearch().azure_vector_search_with_dashboard_filter(final_query = user_question, select = select_cols, dashboard_name = dashboard_name)
    else:
        tables      = AZCognitiveSearch().azure_vector_search(final_query = user_question, select = select_cols, flag = "sql")

    # Load the dashboard extras data
    with open("./Data/dashboard_extras.json", "r") as f:
        dashboard_extras = json.load(f)

    # print(f"{bcolors.FAIL}Total tables coming from the cognitive search: {len(list(tables))}{bcolors.ENDC}")

    table_info, extra_information = OrderedDict(), []
    count = 0
    for i in tables:
        count += 1
        table_name = i['table_name'].split(".")[-1]

        if i['table_name'].lower() == "actvet_uc.certificate_claims":
            continue
        elif i["table_name"].lower() == "actvet_uc.schools_admission_data" or i["table_name"].lower() == "actvet_uc.schools_admission_dim_organization":
            continue

        # if dashboard_name.lower() in dashboard_extras.keys() and table_name.lower() in dashboard_extras[dashboard_name.lower()].keys():
        #     extra_information.extend(dashboard_extras[dashboard_name.lower()][table_name.lower()])
        #     print(f"{bcolors.OKCYAN}Extra information for the table {table_name}: {extra_information}{bcolors.ENDC}")

        print(f"{bcolors.OKCYAN}table name = {i['table_name']} | database = {i['database']} | group name = {i['group_name']} | application name = {i['application_name']}{bcolors.ENDC}")

        table_info[f"{i['database']}.{''.join(i['table_name'].split('.')[1:])}"] = {
            "database_name": i["database"],
            "head_content": i["head_content"],
            "group_name": i["group_name"],
            "application_name": i["application_name"],
            "columns": i["columns"],
            "users": i["users"]
        }
    
    # create schema for the GPT prompt
    schema = ""
    for key in table_info.keys():
        table_name    = key.split(".")[-1]
        database_name = table_info[key]["database_name"]
        schema += f"DATABASE_NAME: {database_name}\nTABLE_NAME: {table_name}\nSAMPLE_DATA: \n{table_info[key]['head_content']}\nGROUP_NAME: {table_info[key]['group_name']}\nAPPLICATION_NAME: {table_info[key]['application_name']}\nCOLUMNS: {table_info[key]['columns']}\n"

    # generate the SQL query
    print(f"{bcolors.FAIL}Extra information list: {extra_information}{bcolors.ENDC}")
    rag_information = ""
    for info in extra_information:
        rag_information += f"- {info}\n"
        
    # rag_information = f"""
    # {rag_information}

    # {schema}

    # QUESTION: {user_question}
    # """
    rag_information = f"""
    {schema}

    QUESTION: {user_question}
    """

    gpt_response = MyAzureOpenAI().sql_openai(rag_information)
    print(f"{bcolors.FAIL}Generated response1: {gpt_response}{bcolors.ENDC}")
    json_match = re.search(r'\{.*\}', gpt_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # Parse the JSON string
        gpt_response = json.loads(json_str)
    else:
        gpt_response = json.loads(gpt_response)

    if gpt_response["sql_query"].lower().replace(".", "") == "NO SQL QUERY CAN BE GENERATED".lower():
        return gpt_response
    
    print(f"{bcolors.FAIL}Generated response: {gpt_response}{bcolors.ENDC}")
    key = f"{gpt_response['Database_Name'].lower()}.{gpt_response['Table_Name'].lower()}"
    user_emails = table_info[f"{key}"]['users']
    user_emails  = [i.lower() for i in user_emails]
    if user_email.lower() not in user_emails or len(user_emails) <= 0:
        gpt_response["sql_query"] = "User has no access to browse this type of data."
        return gpt_response

    


    # ----------------------- Modify the Generated SQL query based on some specific rules --------------------
    print(f"{bcolors.OKBLUE}GPT response SQL query: {gpt_response['Table_Name'].lower()} and the dashboard = {dashboard_name}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}SQL query to modify: {gpt_response['sql_query'].lower()}{bcolors.ENDC}")
    rules = ""
    if dashboard_name.lower() in dashboard_extras.keys():
        for rule in dashboard_extras[dashboard_name.lower()]:
            rules += f"- {rule}\n"

    # for info in extra_information:
    #     rules += f"- {info}\n"
    print(f"{bcolors.OKCYAN}{table_name=} and Rules: {rules}{bcolors.ENDC}")
    modified_sql = MyAzureOpenAI().modify_sql_query(user_question = user_question, original_sql_query = gpt_response["sql_query"], rules = rules)
    print(f"{bcolors.FAIL}Modified SQL Query: {modified_sql}{bcolors.ENDC}")
    gpt_response["sql_query"] = modified_sql["modified_sql_query"]
    # --------------------------------------------------------------------------------------------------------

    gpt_response["context"] = table_info
    # ----------------------- Modify the Generated SQL query based on the ROW Level Access --------------------
    table_info = modify_data_for_row_level_access(user_email = user_email, table_info = table_info)
    row_level_access_query    = table_info[key]['row_level_access']
    row_level_access_col_name = table_info[key]['row_level_access_col_name']

    if len(row_level_access_query) > 0 and row_level_access_query is not None:
        row_level_access_sql_query = MyAzureOpenAI().modify_sql_query_with_row_level_access(
            generated_sql_query = gpt_response["sql_query"],
            row_level_access = row_level_access_query,
            row_level_access_col_name = row_level_access_col_name
        )
        print(f"{bcolors.OKCYAN}Row Level Access Modified SQL Query: {row_level_access_sql_query}{bcolors.ENDC}")
        print(f"{bcolors.OKCYAN}Row Level Access Modified SQL Query type: {type(json.loads(row_level_access_sql_query))}{bcolors.ENDC}")

        gpt_response["sql_query"] = json.loads(row_level_access_sql_query)["sql_query"]
    # -----------------------------------------------------------------------------------------------------------

    # remove "No_" with the "No" in the SQL query
    gpt_response["sql_query"] = gpt_response["sql_query"].replace("No_", "No ")
    end = time.time()
    logger.log(msg= f"Successfully generated SQL First level access..{end-start}",level=logging.INFO)

    return gpt_response


def check_dashboard_name_in_query(user_question: str, threshold: int = 80) -> str:
    # dashboard_table_details = {
    #     "ACTVET partners satisfaction": ["partnerssatisfactionsurvey"],
    #     "actvet staff satisfaction":	["StaffSatisfactionsurvey", "answerslookup_StaffSatisfactionsurvey", "Questionorder_StaffSatisfactionsurvey"],	
    #     "Qualification Department":	["candidates"],
    #     "Schools Parents Satisfaction": ['schools_parents_survey'],
    #     "Schools Students Satisfaction": ["school_students_survey", "AnswersLookup_SchoolStudentsSatisfaction", "QuestionOrder_SchoolStudentsSatisfaction"], 	
    #     "IAT Staff Satisfaction": ["iat_staff_survey"],
    #     "Customer care satisfaction": [
    #         "customer_survey_CustomerCareSatisfaction",	"LinkTable_CustomerCareSatisfaction", 
    #         "Target_CustomerCareSatisfaction", "Cases_CustomerCareSatisfaction"
    #     ],
    #     # "emsat analysis":  	["EMSAT_DATA", "Target_EMSAT"],
    #     "ADVETI Staff Satisfaction":	["ADVETI_Staff_Survey"],
    #     "iat students satisfaction":	["iat_student_survey"],
    #     "vedc dashboard":	["VEDCStudnets_Enrollment"],
    #     "Social Media Dashboard":	["SocialMediaAccounts_Data"],
    #     # "TVET Database Users Satisfaction": [""]
    #     # "EEE Program":	["EEEProgram", "EEE_Candidates"],
    #     "Vedc Students Satisfaction":	["VedcStudentsSurvey"],
    #     "Digital Transformation Dashboard": ["LinkTable_DigitalTransformationDashboard", "Institutions_digital_transformation", "ACTVET_Services_Digial_transformation"],
    #     "schools alumni":	["DIM_ACADEMIC_ORGANIZATION_SCHOOL_ALUMNI", "DIM_STUDENT_SCHOOL_ALUMNI", "schoolsalumni_alumni"],
    #     "World Skills":	["WorldSkillsCompetitors"],
    #     "Emirates Skills":	 ["Asia", "GCC", "Competitors", "Winners"],
    #     # "adveti students satisfaction":	["ADVETI_Student_Surveys"]
    # }

    # dashboard_table_details = {
    #     "schools alumni": ["DIM_ACADEMIC_ORGANIZATION_SCHOOL_ALUMNI", "DIM_STUDENT_SCHOOL_ALUMNI", "schoolsalumni_alumni"],
    #     "vedc dashboard": ["VEDCStudnets_Enrollment"],
    #     "Vedc Students Satisfaction": ["VedcStudentsSurvey"],
    #     "ADVETI Staff Satisfaction": ["StaffSatisfactionsurvey", "answerslookup_StaffSatisfactionsurvey", "Questionorder_StaffSatisfactionsurvey"],
    #     "Schools Parents Satisfaction": ["schools_parents_survey"],
    #     "Schools Students Satisfaction": ["school_students_survey", "AnswersLookup_SchoolStudentsSatisfaction", "QuestionOrder_SchoolStudentsSatisfaction"],
    #     "Social Media Dashboard": ["SocialMediaAccounts_Data"],
    #     "ACTVET partners satisfaction": ["partnerssatisfactionsurvey"],
    #     "adveti students satisfaction": ["ADVETI_Student_Surveys"],
    #     "IAT Staff Satisfaction": ["iat_staff_survey"],
    #     "iat students satisfaction": ["iat_student_survey"],
    #     # "TVET Database Users Satisfaction": ["TVETDBUsersSatisfactionSurvey"]
    # }

    all_dashboard_names = [
        "ACTVET Partners satisfaction", "Schools Students Satisfaction", "Schools Parents Satisfaction",
        "IAT Staff Satisfaction", "IAT Students Satisfaction", "VEDC Dashboard", "Social Media Dashboard",
        "ADVETI Students Satisfaction", "Vedc Students Satisfaction", "TVET Database Users Satisfaction",
        "ACTVET Staff Satisfaction", "Schools Alumni", "World Skills", "EEE Program",
        "advtei certificate claims", "Customer Care Satisfaction", "Emirates Skills", "emsat analysis",
        "Schools Admission Dashboard", "ADVETI Staff Satisfaction", "Post-Secondary Monthly Snapshot",
        "Post-Secondary Weekly Snapshot", "Qualification Department", "Digital Transformation Dashboard",
        "Digital Transformation Dashboard", "schools dashboard", "Post-Secondary Admission Dashboard"
    ]

    # all_dashboards = [dashboard.lower() for dashboard in all_dashboard_names]
    # for dashboard in all_dashboards:
    #     if dashboard in user_question.lower():
    #         return dashboard
    # else:
    #     if "iat student satisfaction" in user_question.lower():
    #         return "iat students satisfaction"
    #     else:
    #         return ""
    user_question_lower = user_question.lower()
    best_match, score = process.extractOne(user_question_lower, all_dashboard_names, scorer=fuzz.partial_ratio)
    print(f"{bcolors.FAIL}Best match: {best_match} and score: {score} for the dashboard name selection{bcolors.ENDC}")
    if score >= threshold:
        return best_match.lower()
    else:
        return ""


def generate_answers_for_pdf(user_question: str, return_response: dict, conv_id: str, actual_query: str, user_email: str) -> tuple:
    start  = time.time()
    answer = find_answer_from_pdf(user_question)

    if "error" in answer.keys():
        return False, {"answer": answer["answer"]}
    
    if "document_name & page_no" in answer.keys():
        return_response["answer"] = {
            "answer": answer["answer"],
            "citation": answer["document_name & page_no"],
            "similar_queries": answer["similar_queries"]
        }
    elif "document_name : page_no" in answer.keys():
        return_response["answer"] = {
            "answer": answer["answer"],
            "citation": answer["document_name : page_no"],
            "similar_queries": answer["similar_queries"]
        }
    else:
        return False, {"answer": "I do not know the answer to your question. Enter the question properly!"}


    # generate the URL for the pdf
    doc_url = document_blob_url(answer['document_name & page_no'])
    return_response["doc_url"] = doc_url
    return_response["question_type"] = "pdf"

    # save the conversation to the CosmosDB
    item = {
        "rephrased_query": user_question,
        "actual_query": actual_query,
        "answer": answer["answer"],
        "citation": answer["document_name & page_no"],
        "similar_queries": answer["similar_queries"],
        "doc_url": doc_url,
        "context": answer["context"],
    }
    is_saved = COSMOS_DB()._add_conversation_item(
        user_email = user_email.lower(),
        conv_id    = conv_id,
        question_type = "pdf",
        item = item
    )
    end = time.time()
    logger.log(msg= f"Successfully generated unstructured answer..{end-start}",level=logging.INFO)
    return is_saved, return_response

def graph_decide(graph_prompt):
    client = AzureOpenAI(
        api_key = "3426761932354d4f8a8fc9bcb1950388", 
        api_version = "2024-02-01",
        azure_endpoint = "https://act-oai-dev-uaen-001.openai.azure.com/"
        )
    completion =  client.chat.completions.create(
        model="gpt-35-turbo-16k",
        messages=[
            {'role': 'user', 'content': graph_prompt},
            {'role': 'assistant', 'content': ""},
            ],
        temperature=0.5,
        top_p = 1
    )

    # Load the JSON string
    json_obj = json.loads(completion.model_dump_json(indent=2))
    
    # Extract the content
    content = json_obj["choices"][0]["message"]["content"]
    print(f"{bcolors.FAIL}Graph Content: {content}{bcolors.ENDC}")
    graph_types = ["scatter plot", "scatter graph","scatter chart","bar graph","bar plot","bar chart","pie chart","pie graph","pie plot","line chart","line plot","line graph", "table"]
    for graph_type in graph_types:
        if graph_type.lower() in content.lower():
            return graph_type
    return None

def is_valid_python_code(code_string):
    try:
        ast.parse(code_string)
        logger.log(msg= f"Successfully checked valid python code ..",level=logging.INFO)
        return True
    except SyntaxError:
        tb = traceback.format_exc()
        logger.log(msg= f"Error in checking valid python code ..{tb}",level=logging.ERROR)
        return False

def generate_answers_for_sql(user_question: str, user_email: str, conversations: dict):
    generated_sql_query    = generate_sql_first_level_access(user_question, user_email.lower())

    print(f"{bcolors.FAIL}Generated SQL Query: {generated_sql_query['sql_query']}{bcolors.ENDC}")

    # if the generated SQL query is not valid or no SQL query can be generated
    if generated_sql_query['sql_query'].lower().replace(".", "") == "NO SQL QUERY CAN BE GENERATED".lower():
        # user_details["answer"] = "I do not know the answer to your question. Enter the question properly!"
        print(f"""{bcolors.FAIL}No SQL query can be generated{bcolors.ENDC}""")
        # raise SqlNotGenerateError("No SQL query can be generated")
        return {"answer": "I do not know the answer to your question. Enter the question properly!", "citation": ""}
    # if the user does not have access to the data
    elif generated_sql_query['sql_query'].lower().replace(".", "") == "User has no access to browse this type of data".lower():
        print(f"""{bcolors.FAIL}User has no access to browse this type of data{bcolors.ENDC}""")
        return {"answer": "Access Denied: You currently lack the necessary permissions to view this data. Please contact your administrator for further assistance.", "citation": ""}

    # connect to the databricks and fetch the data using the generated SQL query
    with sql.connect(
        server_hostname = os.getenv("dbserverhostname"),
        http_path       = os.getenv("dbhttppath"),
        access_token    = os.getenv("dbaccesstoken")
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET spark.sql.legacy.timeParserPolicy = LEGACY")
            cursor.execute(generated_sql_query['sql_query'])
            result = cursor.fetchall()
            if result is None:
                print("***No result***")
            else:
                print("****Fetching result****")

    # convert the fetched data to a pandas dataframe and pass it to the OpenAI model to generate the answer
    df         = pd.DataFrame([r.asDict() for r in result])
    sql_answer = MyAzureOpenAI().dataframe_openai(user_question, df)

    # save the conversation to the CosmosDB
    conversations["value"][0]["conversation"]["structured"].append({"Question": user_question, "Answer": sql_answer,"SQL_QUERY": generated_sql_query['sql_query'], "context": generated_sql_query['context']})
    conversations['value'][0]['conversation']['last_question'] = user_question
    conversations['value'][0]['conversation']['last_answer']   = sql_answer
    is_saved = COSMOS_DB().save_conversation_item(conversations, "structured")
    if is_saved:
        print("Conversation saved successfully")
    else:
        print("Error saving conversation")

    return {
        "answer":         sql_answer,
        "sql_query":      generated_sql_query['sql_query'],
        "table_markdown": df.to_markdown() if df.shape[0] > 2 and df.shape[1] > 2 else "",
        "citation": {
            "database_name":    generated_sql_query['Database_Name'],
            "table_name":       generated_sql_query['Table_Name'],
            "group_name":       generated_sql_query["group_name"],
            "application_name": generated_sql_query["application_name"].title()
        },
    }

    # except ServerOperationError as e:
    #     tb = traceback.format_exc()
    #     print(f"Error from ServerOperationError: {str(e)}\nTraceback:\n{tb}")
    #     return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
    #     return_response['sql_query'] = generated_sql_query['sql_query']

    #     response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    #     return return_response

    # except Exception as ex:
    #     tb = traceback.format_exc()
    #     print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")  
    #     return_response["answer"] = "An internal server error occurred while processing your request. Rest assured, we are actively working to resolve this issue. Thank you for your patience and understanding."
    #     return_response['table_markdown'] = ""
    #     return_response["citation"] = ""
    #     return_response["question_type"] = "sql"

    #     response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    #     return return_response

def modify_data_for_row_level_access(user_email: str, table_info: dict):
    row_level_access_data = pd.read_csv("./Data/rbac_row_level_access.csv")
    schema_mapping        = pd.read_csv("./Data/schema_mapping.csv")

    # Normalize columns to lowercase
    row_level_access_data['Table Name'] = row_level_access_data['Table Name'].str.lower()
    row_level_access_data['Email']      = row_level_access_data['Email'].str.lower()
    schema_mapping['SourceSchemaName']  = schema_mapping['SourceSchemaName'].str.lower()

    # Get unique source schema names
    unique_schemas = schema_mapping['SourceSchemaName'].unique().tolist()

    # Generate schema_type column
    row_level_access_data['schema_type'] = row_level_access_data['Table Name'].apply(
        lambda table: ", ".join([schema for schema in unique_schemas if schema in table])
    )

    our_data = pd.DataFrame({
        "User ID": ["aishwarya.walkar-v@actvet.ac.ae", "rupam.kumari-v@actvet.ac.ae", "rushikesh.goski-v@actvet.ac.ae"],
        "Email": ["aishwarya.walkar-v@actvet.ac.ae", "rupam.kumari-v@actvet.ac.ae", "rushikesh.goski-v@actvet.ac.ae"],
        "Table Name": ["all facts and dimensions under institute and school schemas", "all facts and dimensions under institute and school schemas", "all facts and dimensions under institute and school schemas"],
        "Column Name for Column Level Restriction": [None, None, None],
        "Column Name for Row Level Access": ["ACDORG_SRGT", "ACDORG_SRGT", "ACDORG_SRGT"],
        "Row Value for Row Level Access": ["SELECT ACDORG_SRGT FROM DIM_ACADEMIC_ORGANIZATION WHERE ACDORG_ENTITY = 'ADPOLY'" for i in range(3)],
        "schema_type": ["institute, school", "institute, school", "institute, school"]
    })

    row_level_access_data = pd.concat([row_level_access_data, our_data], axis=0).reset_index().drop(columns = ["index"])

    # Filter user data
    filtered_user          = row_level_access_data[row_level_access_data["Email"] == user_email]
    all_cols_for_row_level_access = filtered_user["Column Name for Row Level Access"].tolist()
    all_row_level_accesses = filtered_user["Row Value for Row Level Access"].tolist()
    accessable_tables      = filtered_user["Table Name"].tolist()
    inaccessable_coumns    = filtered_user["Column Name for Column Level Restriction"].tolist()

    # Check if user has no row level access
    if not all_row_level_accesses:
        print("The user does not have any row-level access!")

    print(f"{all_row_level_accesses=}")
    print(f"{accessable_tables=}")
    print(f"{inaccessable_coumns=}")

    # Modify the tables accordingly if the user has row level access
    for key, value in table_info.items():
        group_for_the_table = value["group_name"]
        table_name = key.split(".")[-1]
        row_level_access, row_level_access_col_name = "", ""

        for idx, row_access in enumerate(all_row_level_accesses):
            # if any(schema in group_for_the_table for schema in filtered_user["schema_type"].iloc[idx].split(", ")):
            if table_name.startswith("fact") or table_name.startswith("dim"):
                if all_cols_for_row_level_access[idx].lower() in map(str.lower, value["columns"]):
                    row_level_access = row_access
                    row_level_access_query = all_cols_for_row_level_access[idx].lower()

                print(f"\n\nGot the row level access for the key --> {key}")
                # # TODO: Check if the table has the column level access
                # table_name = key.lower().split(".")[-1]
                # inaccessable_column_value = inaccessable_coumns[idx]
                # if inaccessable_column_value and inaccessable_column_value.strip().lower() == 'SELECT EMP_BIRTH_DATE FROM HR.DIM_EMPLOYEE'.lower() and table_name.lower() == "DIM_EMPLOYEE".lower():
                #     if "EMP_BIRTH_DATE".lower() in map(str.lower, table_info[key]['columns']):
                #         df = spark.sql(f"SELECT * FROM actvet_uc.gold.{table_name}").toPandas().head(5)
                #         column_name = next((col for col in df.columns if col.lower() == "emp_birth_date"), "")
                #         table_info[key]['head_content'] = df.drop(columns=[column_name]).to_markdown()
                break

        table_info[key]['row_level_access'] = row_level_access
        table_info[key]["row_level_access_col_name"] = row_level_access_col_name

    ## Check which tables have the row level access
    for key, value in table_info.items():
        print(f"Table name = {key.split('.')[-1]}")
        print(f"Group name = {table_info[key]['group_name']}")
        print(f"Row level access = {table_info[key]['row_level_access']}")
        print(f"Row level access column name = {table_info[key]['row_level_access_col_name']}")
        print("--"*50)

    return table_info

def generate_sql_with_row_column_level_access(user_question: str, user_email: str, dashboard_name: str) -> dict:
    if dashboard_name == "" or dashboard_name is None:
        dashboard_name = check_dashboard_name_in_query(user_question)
        
    select_cols = ["head_content", "table_name", "database", "group_name", "application_name", "users", "columns"]

    print(f"{bcolors.FAIL}Dashboard Name from the final query: {dashboard_name}{bcolors.ENDC}")
    if len(dashboard_name) > 0:
        tables      = AZCognitiveSearch().azure_vector_search_with_dashboard_filter(final_query = user_question, select = select_cols, dashboard_name = dashboard_name)
    else:
        tables      = AZCognitiveSearch().azure_vector_search(final_query = user_question, select = select_cols, flag = "sql")

    table_info = OrderedDict()
    for i in tables:
        print(f"{bcolors.OKCYAN}table name = {i['table_name']} | database = {i['database']} | group name = {i['group_name']} | application name = {i['application_name']}{bcolors.ENDC}")
        if i['table_name'] == 'actvet_uc.eee_candidates':
            print(f"{bcolors.OKBLUE}All columns for the competitors table are: {i['columns']}{bcolors.ENDC}")

        table_info[f"{i['database']}.{''.join(i['table_name'].split('.')[1:])}"] = {
            "database_name": i["database"],
            "head_content": i["head_content"],
            "group_name": i["group_name"],
            "application_name": i["application_name"],
            "columns": i["columns"],
            "users": i["users"]
        }

    row_level_access_data = pd.read_csv("./Data/rbac_row_level_access.csv")
    schema_mapping        = pd.read_csv("./Data/schema_mapping.csv")

    # Normalize columns to lowercase
    row_level_access_data['Table Name'] = row_level_access_data['Table Name'].str.lower()
    row_level_access_data['Email']      = row_level_access_data['Email'].str.lower()
    schema_mapping['SourceSchemaName']  = schema_mapping['SourceSchemaName'].str.lower()

    # Get unique source schema names
    unique_schemas = schema_mapping['SourceSchemaName'].unique().tolist()

    # Generate schema_type column
    row_level_access_data['schema_type'] = row_level_access_data['Table Name'].apply(
        lambda table: ", ".join([schema for schema in unique_schemas if schema in table])
    )

    our_data = pd.DataFrame({
        "User ID": ["aishwarya.walkar-v@actvet.ac.ae", "rupam.kumari-v@actvet.ac.ae", "rushikesh.goski-v@actvet.ac.ae"],
        "Email": ["aishwarya.walkar-v@actvet.ac.ae", "rupam.kumari-v@actvet.ac.ae", "rushikesh.goski-v@actvet.ac.ae"],
        "Table Name": ["all facts and dimensions under institute and school schemas", "all facts and dimensions under institute and school schemas", "all facts and dimensions under institute and school schemas"],
        "Column Name for Column Level Restriction": [None, None, None],
        "Column Name for Row Level Access": ["ACDORG_SRGT", "ACDORG_SRGT", "ACDORG_SRGT"],
        "Row Value for Row Level Access": ["SELECT ACDORG_SRGT FROM DIM_ACADEMIC_ORGANIZATION WHERE ACDORG_ENTITY = 'ADPOLY'" for i in range(3)],
        "schema_type": ["institute, school", "institute, school", "institute, school"]
    })

    row_level_access_data = pd.concat([row_level_access_data, our_data], axis=0).reset_index().drop(columns = ["index"])

    # Filter user data
    filtered_user          = row_level_access_data[row_level_access_data["Email"] == user_email]
    all_cols_for_row_level_access = filtered_user["Column Name for Row Level Access"].tolist()
    all_row_level_accesses = filtered_user["Row Value for Row Level Access"].tolist()
    accessable_tables      = filtered_user["Table Name"].tolist()
    inaccessable_coumns    = filtered_user["Column Name for Column Level Restriction"].tolist()
    # print(f"{bcolors.FAIL}Row level access for the filtered_user:\n {filtered_user.columns}{bcolors.ENDC}")

    # Check if user has no row level access
    if not all_row_level_accesses:
        print("The user does not have any row-level access!")

    print(f"{all_row_level_accesses=}")
    print(f"{accessable_tables=}")
    print(f"{inaccessable_coumns=}")

    # Modify the tables accordingly if the user has row level access
    for key, value in table_info.items():
        group_for_the_table = value["group_name"]
        table_name = key.split(".")[-1]
        row_level_access = ""

        for idx, row_access in enumerate(all_row_level_accesses):
            # if any(schema in group_for_the_table for schema in filtered_user["schema_type"].iloc[idx].split(", ")):
            if table_name.startswith("fact") or table_name.startswith("dim"):
                if all_cols_for_row_level_access[idx].lower() in map(str.lower, value["columns"]):
                    row_level_access = row_access

                print(f"\n\nGot the row level access for the key --> {key}")
                # # TODO: Check if the table has the column level access
                # table_name = key.lower().split(".")[-1]
                # inaccessable_column_value = inaccessable_coumns[idx]
                # if inaccessable_column_value and inaccessable_column_value.strip().lower() == 'SELECT EMP_BIRTH_DATE FROM HR.DIM_EMPLOYEE'.lower() and table_name.lower() == "DIM_EMPLOYEE".lower():
                #     if "EMP_BIRTH_DATE".lower() in map(str.lower, table_info[key]['columns']):
                #         df = spark.sql(f"SELECT * FROM actvet_uc.gold.{table_name}").toPandas().head(5)
                #         column_name = next((col for col in df.columns if col.lower() == "emp_birth_date"), "")
                #         table_info[key]['head_content'] = df.drop(columns=[column_name]).to_markdown()
                break

        table_info[key]['row_level_access'] = row_level_access

    ## Check which tables have the row level access
    for key, value in table_info.items():
        print(f"Table name = {key.split('.')[-1]}")
        print(f"Group name = {table_info[key]['group_name']}")
        print(f"Row level access = {table_info[key]['row_level_access']}")
        print("--"*50)


    schema_details = "\n\n"
    for key in table_info.keys():
        schema_details += f"""DATABASE_NAME: {table_info[key]['database_name']}
    TABLE_NAME: {key.split('.')[-1]}
    ROW_LEVEL_ACCESS: {table_info[key]['row_level_access']}
    COLUMN_NAMES: {table_info[key]['columns']}
    SCHEMA:\n{table_info[key]['head_content']}\n\n
    """
    schema_details += f"QUESTION: {user_question}"

    ## Generate the SQL Query using the GPT model
    try:
        gpt_response = json.loads(MyAzureOpenAI().generate_sql_query_with_row_level_access(schema_details))
        print(f"{bcolors.FAIL}Generated SQL Query for row level access: {gpt_response['sql_query']}{bcolors.ENDC}")
        print(f"{bcolors.OKCYAN}All details generated: {gpt_response}{bcolors.ENDC}")

        # find out the dashboard/application name and group name for the table
        gpt_response["application_name"] = table_info[f"{gpt_response['database_name']}.{gpt_response['table_name']}"]['application_name']
        gpt_response["group_name"]       = table_info[f"{gpt_response['database_name']}.{gpt_response['table_name']}"]['group_name']
        users_list                       = table_info[f"{gpt_response['database_name']}.{gpt_response['table_name']}"]['users']

        gpt_response["context"] = table_info

        # check if the user has access to the data
        if user_email.lower() not in users_list or len(users_list) <= 0:
            gpt_response["sql_query"] = "User has no access to browse this type of data"
            return gpt_response
        
        # remove "No_" with the "No" in the SQL query
        gpt_response["sql_query"] = gpt_response["sql_query"].replace("No_", "No ")
        return gpt_response
    except Exception as ex:
        tb = traceback.format_exc()
        logger.log(msg= f"Error generating SQL Query with row level access: {str(ex)}\nTraceback:\n{tb}", level=logging.ERROR)
        print(f"{bcolors.FAIL}Error generating SQL Query with row level access: {str(ex)}{bcolors.ENDC}")
        return {"error": str(ex)}


def get_or_generate_faqs(dashboard_name: str):
    try:
        questions = COSMOS_DB().get_questions_for_faqs(dashboard_name)
        print(f"{bcolors.FAIL}Total questions found in Cosmos DB for FAQs: {questions}{bcolors.ENDC}")
        if len(questions) <= 0:
            ##### ! Generate FAQs based on the details of the dashboard using GPT#####
            select_cols = ["head_content", "application_name"]
            results = list(AZCognitiveSearch().azure_vector_search_with_dashboard_filter(final_query = dashboard_name, select = select_cols, dashboard_name = dashboard_name))
            length_of_result = len(list(results))
            print(f"{bcolors.FAIL}Results for FAQs for checking: {length_of_result} and condition = {length_of_result > 0}{bcolors.ENDC}")
            print(f"{bcolors.FAIL}Results for FAQs and length = {length_of_result}: {length_of_result > 0}{bcolors.ENDC}")
            if length_of_result <= 0:
                return {"error": "No data found for the dashboard name"}

            schema = ""
            for i in list(results):
                schema += f"DATAFRAME: {i['head_content']}\DASHBOARD_NAME: {i['application_name']}\n\n"
            
            faqs = MyAzureOpenAI().generate_faqs_based_on_dataframe(schema = schema)
            if type(faqs) == str:
                return {"error": faqs}
            return faqs

        #### !group the similar questions #####
        ## vectorize the questions
        vectorizer = TfidfVectorizer(stop_words='english')
        X          = vectorizer.fit_transform(questions)

        # Use KMeans clustering to group similar questions
        n_clusters = min(len(questions) // 2, 10) if min(len(questions) // 2, 10) > 0 else 1
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(X)

        clusters = {}
        for i, label in enumerate(kmeans.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(questions[i])

        ##### ! Randomly select one question from each cluster until we have enough questions #####
        selected_questions = []
        cluster_keys = list(clusters.keys())
        random.shuffle(cluster_keys)

        for key in cluster_keys:
            if len(selected_questions) < 5:
                selected_questions.append(random.choice(clusters[key]))
            else:
                break

        return {
            "FAQs": selected_questions
        }
    except Exception as ex:
        tb = traceback.format_exc()
        print(f"{bcolors.FAIL}Error getting or generating FAQs: {str(ex)}\nTraceback:\n{tb}{bcolors.ENDC}")
        logger.log(msg= f"Successfully generated greet message.. {tb}", level=logging.ERROR)
        return {"FAQs": "An error occurred while generating FAQs. Please try again later."}