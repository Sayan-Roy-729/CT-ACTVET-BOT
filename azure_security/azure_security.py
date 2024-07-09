import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

class AzureKeyVault:
    """
    Class representing a security connection.

    This class provides methods to create and manage connections to security-related services.
    """

    def __init__(self):
        try:
            key_vault_url = "https://act-kv-dev-uaen-01.vault.azure.net/"
            if os.getenv("APPLICATION_RUNNING") == "dev":
                credential = InteractiveBrowserCredential(additionally_allowed_tenants=['*'])
            elif os.getenv("APPLICATION_RUNNING") == "prod":
                credential = DefaultAzureCredential()
            else:
                credential = DefaultAzureCredential()

            kay_vault_name = "act-kv-dev-uaen-01"
            key_vault_url  = f"https://{kay_vault_name}.vault.azure.net"

            self.client = SecretClient(vault_url = key_vault_url, credential = credential)
        except Exception as ex:
            print("Exception ", ex)

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
            secret = str(self.client.get_secret(name).value)
            return secret
        except Exception as ex:
            return False

    def set_environment_from_key_vault(self):
        print("Setting environment variables from Key Vault")
        secrets = self.client.list_properties_of_secrets()
        for secret in secrets:
            try:
                os.environ[str(secret.name)] = self.fetch_secret(
                    str(secret.name)
                )
            except Exception as ex:
                print("Exception ", ex)
        print("Environment variables set from Key Vault")
                

