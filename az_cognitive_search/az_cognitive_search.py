import os

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from azure_openai.azure_openai import MyAzureOpenAI
from azure_logging.azure_logging import logger
import logging
import time


class AZCognitiveSearch:
    def __init__(self) -> None:
        # ! Have to remove these credentials
        self.endpoint       = os.getenv("searchendpoint")
        self.key            = os.getenv("searchservicekey1")
        self.index_name_sql = "actvet-index-sheet-databricks2"
        self.index_name_pdf = "actvet-docu-search-databricks"

    def azure_vector_search(self,  final_query: str, select: list[str], flag: str):
        start = time.time()
        vector_query = VectorizedQuery(
            vector              = MyAzureOpenAI().openai_embeddings(final_query),
            k_nearest_neighbors = 5,
            fields              = "contentVector"
        )
        
        if flag == "sql":
            index_name = self.index_name_sql
        else:
            index_name = self.index_name_pdf

        search_client = SearchClient(
            endpoint   = self.endpoint,
            index_name = index_name,
            credential = AzureKeyCredential(self.key)
        )
        
        results = search_client.search(  
            search_text    = final_query,  
            vector_queries = [vector_query],
            select         = select,
            top            = 5
        )
        end = time.time()
        logger.log(msg= f"Azure Vector Search ended..{end-start}",level=logging.INFO)
        return results
    
    def azure_vector_search_with_dashboard_filter(self, final_query: str, select: list[str], dashboard_name: str):
        start = time.time()
        vector_query = VectorizedQuery(
            vector              = MyAzureOpenAI().openai_embeddings(final_query),
            k_nearest_neighbors = 5,
            fields              = "contentVector"
        )

        search_client = SearchClient(
            endpoint   = self.endpoint,
            index_name = self.index_name_sql,
            credential = AzureKeyCredential(self.key)
        )
        
        results = search_client.search(  
            search_text    = final_query,  
            vector_queries = [vector_query],
            select         = select,
            filter         = f"application_name eq '{dashboard_name.lower()}'",
            top           = 5
        )
        end = time.time()
        logger.log(msg= f"Azure Vector Search with filter ended..{end-start}",level=logging.INFO)
        return results
    