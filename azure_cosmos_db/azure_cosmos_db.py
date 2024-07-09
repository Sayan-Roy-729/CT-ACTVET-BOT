import os
import uuid
import time
import logging
import traceback

from azure.cosmos import CosmosClient, ContainerProxy
from azure_logging.azure_logging import logger

class COSMOS_DB:
    def __init__(self):
        # ! TODO: Remove these credentials
        self.cosmos_endpoint  = os.getenv("cosmosdbendpoint")
        self.cosmos_key       = os.getenv("cosmosdbkey")
        self.cosmos_database  = "ToDoList"
        self.cosmos_container = "Items"
        self.cosmos_database_faqs = "dashboard_faqs"
        self.cosmos_container_faqs = "items"

    def _get_container_client(self, database_name: str, container_name: str) -> ContainerProxy:
        client    = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
        database  = client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        return container
    
    def _add_conversation_item(self, user_email: str, conv_id: str, question_type: str, item: dict) -> bool:
        details = {
            "id": uuid.uuid4().hex,
            "partitionKey": user_email,
            "conv_id": conv_id,
            "question_type": question_type
        }

        try:
            self._get_container_client(database_name = self.cosmos_database, container_name = self.cosmos_container).upsert_item({**details, **item})
            return True
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error adding conversation item: {str(ex)} Traceback:\n{tb}")
            logger.log(msg=f"Error adding conversation item: {str(ex)}", level=logging.ERROR)
            return False
        
    def get_last_conversation(self, user_email: str) -> list[dict]:
        container = self._get_container_client(database_name = self.cosmos_database, container_name = self.cosmos_container)
        queryText  = "SELECT * FROM c WHERE c.partitionKey = @user_id ORDER BY c.createdTime DESC OFFSET 0 LIMIT 1"
        parameters = [{"name": "@user_id", "value": user_email.lower()}]

        results = list(container.query_items(
            query                        = queryText,
            parameters                   = parameters,
            enable_cross_partition_query = True,
        ))
        if len(results) <= 0:
            return {"question": "", "answer": ""}
        else:
            return {"question": results[0]["rephrased_query"], "answer": results[0]["answer"]}

    
    def save_user_query_for_faqs(self, user_query: str, rephrased_query: str, dashboard_name: str) -> bool:
        """
        Save the user query to Cosmos DB for FAQs.

        Args:
            user_query (str): The user query.
            rephrased_query (str): The rephrased query.
            dashboard_name (str): The dashboard name.

        Returns:
            bool: True if the user query was saved successfully, False otherwise.
        """
        new_item = {
            "id": uuid.uuid4().hex,
            "user_query": user_query,
            "rephrased_query": rephrased_query,
            "dashboard_name": dashboard_name
        }
        try:
            container = self._get_container_client(database_name = self.cosmos_database_faqs, container_name = self.cosmos_container_faqs)
            container.upsert_item(new_item)
            return True
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error saving user query: {str(ex)}\nTraceback:\n{tb}")
            logger.log(msg=f"Error saving user query: {str(ex)}", level=logging.ERROR)
            return False
        

    def get_questions_for_faqs(self, dashboard_name: str) -> list[str]:
        """
        Get the questions for the FAQs.

        Args:
            dashboard_name (str): The dashboard name.

        Returns:
            list[str]: The list of questions.
        """
        container = self._get_container_client(database_name = self.cosmos_database_faqs, container_name = self.cosmos_container_faqs)
        queryText  = "SELECT * FROM c WHERE c.dashboard_name = @dashboard_name"
        parameters = [{"name": "@dashboard_name", "value": dashboard_name}]

        results = list(container.query_items(
            query                        = queryText,
            parameters                   = parameters,
            enable_cross_partition_query = True,
        ))

        if len(results) <= 0:
            return []
        questions = [result["rephrased_query"] for result in results]
        return questions

    def delete_items(self, user_email: str) -> bool:
        """
        Delete all items for a user from Cosmos DB.

        Args:
            user_email (str): The user email.

        Returns:
            bool: True if the items were deleted successfully, False otherwise.
        """
        try:
            container = self._get_container_client(database_name = self.cosmos_database, container_name = self.cosmos_container)
            queryText  = "SELECT * FROM c WHERE c.partitionKey = @user_id"
            parameters = [{"name": "@user_id", "value": user_email.lower()}]

            results = list(container.query_items(
                query                        = queryText,
                parameters                   = parameters,
                enable_cross_partition_query = True,
            ))

            for result in results:
                container.delete_item(item=result["id"], partition_key=user_email.lower())
            return True
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error deleting items: {str(ex)}\nTraceback:\n{tb}")
            logger.log(msg=f"Error deleting items: {str(ex)}", level=logging.ERROR)
            return False


    def get_history_for_user(self, user_email: str, conv_id: str) -> dict:
        start = time.time()
        container = self._get_container_client()

        # Query to fetch structured and unstructured data
        query = f"SELECT * FROM c WHERE c.id = '{user_email}'"
        items = list(container.query_items(query, enable_cross_partition_query=True))

        conversations = {
            "id": user_email,
            "value": [{
                "conv_id": conv_id,
                "conversation": {
                    "last_question": "",
                    "last_answer": ""
                },
            }]
        }
        try:
            if items:
                # Assuming there's only one matching document (key is unique)
                item   = items[0]
                values = item.get("value")
                structured_data   = values[0]["conversation"]["structured"][-2:]
                unstructured_data = values[0]["conversation"]["unstructured"][-2:]

                last_two_structured_ques   = [
                    {"Question": entry["Question"], "Answer": entry["Answer"]} for entry in structured_data
                ]
                last_two_unstructured_ques = [
                    {"Question": entry["Question"], "Answer": entry["Answer"]} for entry in unstructured_data
                ]
                conversations["value"][0]["conversation"]["structured"] = last_two_structured_ques
                conversations["value"][0]["conversation"]["unstructured"] = last_two_unstructured_ques
                conversations["value"][0]["conversation"]["last_question"] = values[0]["conversation"]["last_question"]
                conversations["value"][0]["conversation"]["last_answer"] = values[0]["conversation"]["last_answer"]
            else:
                conversations["value"][0]["conversation"]["structured"] = []
                conversations["value"][0]["conversation"]["unstructured"] = []
            
            end = time.time()
            logger.log(msg= f"Successfully Fetched conversation history..{end-start}",level=logging.INFO)
            return conversations
        except Exception as e:
            logger.log(msg= f"Error while fetching conversation history..{e}",level=logging.ERROR)
    

    def save_conversation_item(self, conversation_item: dict, query_type: str) -> bool:
        """
        Save the conversation item to Cosmos DB.
        """
        try:
            start = time.time()
            container = self._get_container_client()
            query = f"SELECT * FROM c WHERE c.id = '{conversation_item['id']}'"
            existing_items = list(container.query_items(query, enable_cross_partition_query=True))
            if len(existing_items) > 0:
                # If the ID exists, append the last conversation entry
                existing_item = existing_items[0]
                if query_type == "structured":
                    # Extract the last conversation entry
                    existing_item['value'][0]['conversation']['structured'].append(conversation_item['value'][0]['conversation']['structured'][-1])
                    existing_item['value'][0]['conversation']['last_answer']   = conversation_item['value'][0]['conversation']['last_answer']
                elif query_type == "unstructured":
                    existing_item['value'][0]['conversation']['unstructured'].append(conversation_item['value'][0]['conversation']['unstructured'][-1])
                    existing_item['value'][0]['conversation']['last_answer']   = conversation_item['value'][0]['conversation']['last_answer']

                # Update the last question and answer
                existing_item['value'][0]['conversation']['last_question'] = conversation_item['value'][0]['conversation']['last_question']
                container.replace_item(item=existing_item['id'], body=existing_item)
            else:
                if query_type == "structured":
                    conversation_item['value'][0]['conversation']['structured']  = [conversation_item['value'][0]['conversation']['structured'][-1]]
                    conversation_item['value'][0]['conversation']['last_answer'] = conversation_item['value'][0]['conversation']['last_answer']
                elif query_type == "unstructured":
                    conversation_item['value'][0]['conversation']['unstructured']  = [conversation_item['value'][0]['conversation']['unstructured'][-1]]
                    conversation_item['value'][0]['conversation']['last_answer']   = conversation_item['value'][0]['conversation']['last_answer']
                
                # Add the last question and answer
                conversation_item['value'][0]['conversation']['last_question'] = conversation_item['value'][0]['conversation']['last_question']
                container.create_item(body=conversation_item)
            
            end = time.time()
            logger.log(msg= f"Successfully Saved Data to Cosmosdb..{end-start}",level=logging.INFO)
        
            return True
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"Error saving conversation item: {str(ex)}\nTraceback:\n{tb}")
            logger.log(msg=f"Error occured when reading  the word file: {str(ex)}", level=logging.ERROR)