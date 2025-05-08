import logging
import logging.config
import sys
import anyio
import uvicorn

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from starlette.applications import Starlette
from starlette.routing import Mount, Host, Route

from functions import LightsPlugin

load_dotenv()  # Load environment variables from .env file

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,  # Allow existing loggers to propagate
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        },
    },
    'handlers': {
        'file_handler': {
            'class': 'logging.FileHandler',
            'filename': 'output.log',
            'mode': 'a',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'console_handler': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'standard',
            'level': 'ERROR',
        },
    },
    'root': {
        'handlers': ['file_handler', 'console_handler'],
        'level': 'DEBUG',
    },
    'loggers': {
        # Explicitly configure external loggers to propagate to root
        'shared.util': {
            'handlers': ['file_handler'],  # Only file handler
            'level': 'INFO',
            'propagate': True,
        },
        # Add more external loggers here if needed
    },
}

# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)  # Use a module-specific logger

from configuration import Configuration
config = Configuration()

mcp_port = config.get_value("MCP_PORT", default=5000)

deployment_name = config.get_value("AZURE_OPENAI_DEPLOYMENT_NAME", default="chat")
api_version = config.get_value("AZURE_OPENAI_API_VERSION")
endpoint = config.get_value("AZURE_OPENAI_ENDPOINT")

kernel = Kernel()
kernel.add_service(AzureChatCompletion(service_id="default", deployment_name=deployment_name, api_version=api_version, endpoint=endpoint))

# Add functions and prompts as usual
# Add the plugin to the kernel
kernel.add_plugin(
    LightsPlugin(),
    plugin_name="Lights",
)

server = kernel.as_mcp_server(server_name="sk")

# Run as stdio server
async def handle_stdin():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

#for local testing
#anyio.run(handle_stdin)

# Define handler functions
async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )

#https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.tools.mcp.html
# Create an SSE transport at an endpoint
sse = SseServerTransport("/messages/")

# Create Starlette routes for SSE and message handling
routes = [
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse.handle_post_message),
]

# Create and run Starlette app
starlette_app = Starlette(routes=routes)
uvicorn.run(starlette_app, host="0.0.0.0", port=mcp_port)