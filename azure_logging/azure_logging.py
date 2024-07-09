import os
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

def setup_logger():
    """
    Sets up the logger configuration and adds an Azure Log handler.

    This method configures the logging settings for the application, including
    the log level and format. It also adds an AzureLogHandler to send logs to
    Azure Application Insights.

    Returns:
        logging.Logger: A logger instance configured with the specified settings.

    Example:
        logger = setup_logger()
        logger.info("Application started.")
        logger.error("An error occurred.")
    """
    # Configure the logging settings
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create a azure  handler to write logs to the JSON file
    azure_handler = AzureLogHandler(
        instrumentation_key="0e6c53f8-f145-402e-9da1-12d8bae9f4f2", connection_string = "InstrumentationKey=0e6c53f8-f145-402e-9da1-12d8bae9f4f2;IngestionEndpoint=https://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope.livediagnostics.monitor.azure.com/;ApplicationId=841dd870-be12-47a9-a371-d02254bb3207" 
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(azure_handler)

    return logger


logger = setup_logger()
