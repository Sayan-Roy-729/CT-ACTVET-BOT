import os
import traceback
import logging
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure_logging.azure_logging import logger

class AzureKeyVault:
    """
    Class representing a security connection.

    This class provides methods to create and manage connections to security-related services.
    """

    def __init__(self):
        try:
            #logger.log(message="Initialization of Azure Keyvault client", level=logging.INFO)
            key_vault_url = "https://act-kv-dev-uaen-01.vault.azure.net/"
            credential = InteractiveBrowserCredential(additionally_allowed_tenants=['*'])
            # credential = DefaultAzureCredential()
            self.client = SecretClient(vault_url=key_vault_url, credential=credential)
            #logger.log(message="STATUS: Successful", level=logging.INFO)
        except Exception as ex:
            tb = traceback.format_exc()
            logger.log(message=f"STATUS: Failed,   ERROR: {ex},   TRACEBACK: {tb}", level=logging.INFO)
            print("Exception ", ex)
            #logger.log(message=f"STATUS: Failed,   ERROR: {ex}", level=logging.INFO)



    def fetch_secret(self, name):
        """
        Fetches a secret from Key Vault. Retrieves a secret from the Key Vault using the provided secret name.

        Args:
            name (str): The name of the secret to fetch from the Key Vault.

        Returns:
            str or bool: If the secret is successfully retrieved, returns the secret value as a string.
                If an error occurs during retrieval, returns False.

        Raises:
            None
        """
        try:
            #logger.log(message=f"Fetching keyvault secret: {name}", level=logging.INFO)
            secret = str(self.client.get_secret(name).value)
            logger.log(message=f"STATUS: Fetching keyvault secret Successful", level=logging.INFO)
            return secret
        except Exception as ex:
            logger.log(message=f"STATUS: Fetching keyvault secret Failed,  ERROR:  {ex}", level=logging.INFO)
            return False

    def set_environment_from_key_vault(self):
        secrets = self.client.list_properties_of_secrets()
        for secret in secrets:
            try:
                #print(f"fetching credential {secret.name}")
                logger.log(message=f"Adding {secret.name} into environment", level=logging.INFO)
                os.environ[str(secret.name)] = self.fetch_secret(
                    str(secret.name)
                )
                logger.log(message=f"STATUS: Adding {secret.name} Sucessful", level=logging.INFO)
            except Exception as ex:
                logger.log(message=f"STATUS: Adding {secret.name},  ERROR: {ex}", level=logging.INFO)
                print("Exception ", ex)
                

