import os
from datetime import datetime, timedelta
import traceback
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure_logging.azure_logging import logger
import logging
import time

def document_blob_url(data):
    start = time.time()
    connection_string = os.getenv("storageconnstring1")
    container_name    = 'actcntdevuaen01/Staging_Storage/SFTP_PDF'
    file_url_with_sas = ""
    response_text = data
    # Extract document name using regex
    print(f"Response text: {response_text}")
    match = response_text.split(' : ')[0]
    try:
        if match:
            # Generate SAS token for the blob
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=container_name, blob = match)
            # Define the start and expiry time for the SAS token
            start_time = datetime.utcnow()
            expiry_time = start_time + timedelta(hours=1)  # Adjust the expiry time as needed
            # Define permissions for the SAS token
            permissions = BlobSasPermissions(read=True, list=True)  # Adjust permissions as needed
            # Generate the SAS token
            sas_token = generate_blob_sas(
                blob_client.account_name,
                container_name,
                match,
                account_key=blob_service_client.credential.account_key,
                permission=permissions,
                start=start_time,
                expiry=expiry_time
            )
            # Generate the URL with the SAS token
            file_url_with_sas = f"https://{blob_client.account_name}.blob.core.windows.net/{container_name}/{match}?{sas_token}"
            end = time.time()
            logger.log(msg= f"Successfully Fetched Document blob url..{end-start}",level=logging.INFO)
    except Exception as e:
        print("Document name not found in the response.")
        logger.log(msg= f"Document blob url not generated successfully ..",level=logging.ERROR)
        logger.log(msg=e,level=logging.ERROR)
    return file_url_with_sas


def upload_image_to_blob(image_bytes):

    # Set up Azure Blob Storage connection details
    account_name = os.getenv("storageaccname")
    account_key = os.getenv("storageacckey1")
    container_name = "graphs"
    start = time.time()
    # Generate a unique blob name based on the current date and time
    current_datetime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    blob_name = f"plot_{current_datetime}.png"
 
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
 
    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)
 
    # Upload the image bytes to Azure Blob Storage
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(image_bytes)

    # Get the SAS token for the image blob with read permissions and a custom expiration time
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
 
    sas_token = generate_blob_sas(
        container_client.account_name,
        container_name,
        blob_name,
        account_key=container_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=50)
    )
 
    # Construct the full URL with the SAS token
    sas_url = f"{blob_client.url}?{sas_token}"
    end = time.time()
    logger.log(msg=f"Image uploded to blob storage successfully.... {end-start}",level=logging.INFO)
    # return blob_client.url
    return sas_url
