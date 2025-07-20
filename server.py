import os
import contextlib
import json
import asyncio
import logging
import logging.config
import sys
import anyio
import uvicorn
import fastapi

from connectors import CosmosDBClient
from collections.abc import AsyncIterator
from contextvars import ContextVar
from datetime import datetime
from telemetry import Telemetry

from fastapi import FastAPI, Depends, APIRouter
#from fastapi_mcp import FastApiMCP
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

from typing import Optional

from util.tools import is_azure_environment

from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import \
    SessionsPythonTool
from semantic_kernel.exceptions.function_exceptions import \
    FunctionExecutionException

from starlette.applications import Starlette
from starlette.endpoints import WebSocketEndpoint
from starlette.routing import Mount, Host, Route, WebSocketRoute
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.middleware import Middleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Receive, Scope, Send
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential,get_bearer_token_provider

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from lifespan_manager import lifespan

from middleware.authentication_middleware import AuthenticationMiddleware
from sse import FastApiSseServerTransport

from event_store import InMemoryEventStore

from dependencies import API_NAME, get_config, validate_api_key_header

class WebsocketServer(WebSocketEndpoint):
    encoding = "bytes"

    async def on_connect(self, websocket):
        print("on connect")

    async def on_receive(self, websocket, data):
        print("on receive")        

    async def on_disconnect(self, websocket, close_code):
        print("on disconnect")

def auth_callback_factory(scope):
    auth_token = None
    async def auth_callback() -> str:
        """Auth callback for the SessionsPythonTool.
        This is a sample auth callback that shows how to use Azure's DefaultAzureCredential
        to get an access token.
        """
        nonlocal auth_token
        current_utc_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

        if not auth_token or auth_token.expires_on < current_utc_timestamp:
            credential = DefaultAzureCredential()

            try:
                auth_token = credential.get_token(scope)
            except ClientAuthenticationError as cae:
                err_messages = getattr(cae, "messages", [])
                raise FunctionExecutionException(
                    f"Failed to retrieve the client auth token with messages: {' '.join(err_messages)}"
                ) from cae

        return auth_token.token
    
    return auth_callback

config = get_config()

#configure basic logging
Telemetry.configure_basic(config)

#MCP Settings
mcp_port = config.get_value("AZURE_MCP_SERVER_PORT", default=5000, type=int)
mcp_timeout = config.get_value("AZURE_MCP_SERVER_TIMEOUT", default=240, type=int)
mcp_mode = config.get_value("AZURE_MCP_SERVER_MODE", default="fastapi")
mcp_transport = config.get_value("AZURE_MCP_SERVER_TRANSPORT", default="sse")
mcp_host = config.get_value("AZURE_MCP_SERVER_HOST", default="local")
mcp_enable_auth = config.get_value("AZURE_MCP_SERVER_ENABLE_AUTH", default=True, type=bool)
json_response = config.get_value("AZURE_MCP_SERVER_JSON", default=True, type=bool)

#OPEN AI SETTTINGS
deployment_name = config.get_value("AZURE_OPENAI_DEPLOYMENT_NAME", default="chat")
api_version = config.get_value("AZURE_OPENAI_API_VERSION")
endpoint = config.get_value("AZURE_OPENAI_ENDPOINT")

use_code_interpreter = config.get_value("USE_CODE_INTERPRETER", default=False, type=bool)
pool_management_endpoint = config.get_value("POOL_MANAGEMENT_ENDPOINT", default="https://dynamicsessions.io")

kernel = Kernel()
token_provider = get_bearer_token_provider(
            config.credential, 
            "https://cognitiveservices.azure.com/.default"
        )
kernel.add_service(AzureChatCompletion(service_id="default", deployment_name=deployment_name, api_version=api_version, endpoint=endpoint, ad_token_provider=token_provider))

cosmos = CosmosDBClient(config=config)

def load_tools_from_json(tool_config):
    loaded_tools = []
    agents_to_load = tool_config.get("agents", [])
    tools_to_load = tool_config.get('tools', [])

    for agent in agents_to_load:
        #TODO
        pass

    #for each tool, load the class and add it to the kernel
    for tool in tools_to_load:
        try:
            logging.info(f"Loading tool: {tool['name']}")
            module = __import__(tool["module"], fromlist=[tool["class"]])
            tool_class = getattr(module, tool["class"])
            tool['settings']['config'] = config
            tool_instance = tool_class(settings=tool["settings"])
            
            # If the class has a plug_in attribute, use it
            if 'plug_in' in tool_instance.__dict__:
                kernel.add_plugin(tool_instance.plug_in, tool["name"])
            else:
                kernel.add_plugin(tool_instance, tool["name"])

            loaded_tools.append(tool_instance)
            
            logging.info(f"Loaded tool: {tool['name']}")
        except Exception as e:
            logging.error(f"Failed to load tool {tool['name']}: {e}")
            continue

    return loaded_tools

def load_tools_from_cosmos(container_name, document_id):
    loaded_tools = []
    logging.info(f"Loading tools from cosmos {container_name} : {document_id}")
    try:
        tool_config = cosmos.get_document(container_name, document_id)
        
        if tool_config is None:
            logging.error(f"Tool configuration document {document_id} not found in container {container_name}.")
            return loaded_tools

        loaded_tools.extend(load_tools_from_json(tool_config))

    except Exception as e:
        logging.error(f"Error loading tools from Cosmos DB: {e}")

    return loaded_tools

def load_tools_from_file(file_name):
    loaded_tools = []
    logging.info(f"Loading tools from {file_name}")
    #load the tool_config.json
    tool_config = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(tool_config):
        with open(tool_config, "r") as f:
            tool_config = json.load(f)

        loaded_tools.extend(load_tools_from_json(tool_config))

    return loaded_tools

loaded_tools = []
loaded_tools.extend(load_tools_from_cosmos("mcp", "tool_config"))
loaded_tools.extend(load_tools_from_cosmos("mcp", "tool_config_custom"))

if (use_code_interpreter):
    logging.info("Loading Code Interpreter")
    sessions_tool = SessionsPythonTool(
        pool_management_endpoint=pool_management_endpoint,
        auth_callback=auth_callback_factory("https://dynamicsessions.io/.default")
    )
    kernel.add_plugin(sessions_tool, "SessionsTool")
    
#A2A future
#https://github.com/google/A2A/blob/main/samples/python/agents/semantickernel/agent.py

server = kernel.as_mcp_server(server_name="sk")

fastapi_request_var: ContextVar[Optional[Request]] = ContextVar('fastapi_request', default=None)

logging.info(f"Starting MCP server in {mcp_mode} mode on port {mcp_port}")
if (mcp_mode == "stdio"):
    # Run as stdio server
    async def handle_stdin():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    #for local testing
    anyio.run(handle_stdin)

elif (mcp_transport == "stateless"):
    #https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/servers/simple-streamablehttp-stateless/mcp_simple_streamablehttp_stateless/server.py
    event_store = InMemoryEventStore()

    # Create the session manager with our app and event store
    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=event_store,  # Enable resumability
        json_response=json_response,
    )

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        dependencies=[Depends(validate_api_key_header)],
        lifespan=lifespan,
    )

elif (mcp_mode == "fastapi" and mcp_transport == "sse"):
    # Add the MCP server to your FastAPI app
    app = FastAPI(
        title=API_NAME,
        description="MCP Server for GPT RAG",
        version="1.0.0",
        lifespan=lifespan,
        #dependencies=[Depends(validate_api_key_header)],
        config=config
    )

    message_router = APIRouter(
        dependencies=[Depends(validate_api_key_header)],
        responses={404: {'description':'Not found'}}
    )

    FastAPIInstrumentor.instrument_app(app)

    def set_fastapi_request(request: fastapi.Request):
        fastapi_request_var.set(request)

    #https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.tools.mcp.html
    # Create an SSE transport at an endpoint
    sse = FastApiSseServerTransport("/messages/")

    #async def handle_post_message(scope, recieve, send):
    @message_router.post('/messages')
    async def handle_post_message(request: fastapi.Request) -> fastapi.Response:
        #set the request context for downstream tools
        set_fastapi_request(request.scope)

        #check the api-key header
        if mcp_enable_auth == True:
            if not request.headers.get("X-API-Key") == config.get_value("AZURE_MCP_SERVER_APIKEY"):
                return PlainTextResponse("Unauthorized", status_code=401)

        try:
            return await sse.handle_post_message(request)
        except Exception as e:
            logging.error(f"Error in handle_post_message: {e}")
            return PlainTextResponse(str(e), status_code=500)

    async def handle_sse(request : fastapi.Request):
        #set the request context for downstream tools
        set_fastapi_request(request)

        #check the api-key header
        if mcp_enable_auth == True:
            if not request.headers.get("X-API-Key") == config.get_value("AZURE_MCP_SERVER_APIKEY"):
                return PlainTextResponse("Unauthorized", status_code=401)

        try:
            async with sse.connect_sse(
                request.scope, request._receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options(), raise_exceptions=True
                )
        except Exception as e:
            #reinit server
            #server = kernel.as_mcp_server(server_name="sk")
            while isinstance(e, Exception) and hasattr(e, 'exceptions') and e.exceptions:
                e = e.exceptions[0]
                logging.error(f"Exception details: {str(e)}")

    async def handle_root(request):
        return PlainTextResponse("Welcome to the GPT RAG MCP Server.")

    app.add_route("/", route=handle_root, methods=["GET", "POST"])
    app.add_route("/sse", route=handle_sse, methods=["GET", "POST"])
    #app.include_router(message_router)
    app.add_route("/messages", route=handle_post_message, methods=["POST"])

    for tool in loaded_tools:
        if hasattr(tool, "has_oauth_endpoint") and tool.has_oauth_endpoint:
            app.add_route(
                tool.oauth_token_endpoint,
                tool.handle_oauth_token,
                methods=["GET", "POST"])
            
            app.add_route(
                tool.oauth_authorize_endpoint,
                tool.handle_oauth_authorize,
                methods=["GET", "POST"])

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return PlainTextResponse(str(exc), status_code=400)
    
elif (mcp_mode == "sse"):
    #https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.tools.mcp.html
    # Create an SSE transport at an endpoint
    sse = SseServerTransport("/messages/")

    def set_fastapi_request(request: Request):
        fastapi_request_var.set(request)

    async def handle_post_message(scope, recieve, send):
        #set the request context for downstream tools
        set_fastapi_request(scope)

        try:
            return await sse.handle_post_message(scope, recieve, send)
        except Exception as e:
            #reinit server
            #server = kernel.as_mcp_server(server_name="sk")
            while isinstance(e, Exception) and hasattr(e, 'exceptions') and e.exceptions:
                e = e.exceptions[0]
                logging.error(f"Exception details: {str(e)}")

    # Define handler functions
    async def handle_sse(request):

        set_fastapi_request(request)

        #check the api-key header
        if mcp_enable_auth == True:
            if not request.headers.get("X-API-Key") == config.get_value("AZURE_MCP_SERVER_APIKEY"):
                return PlainTextResponse("Unauthorized", status_code=401)

        try:
            async with sse.connect_sse(
                request.scope, request._receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options(), raise_exceptions=True
                )
        except Exception as e:
            #reinit server
            #server = kernel.as_mcp_server(server_name="sk")
            while isinstance(e, Exception) and hasattr(e, 'exceptions') and e.exceptions:
                e = e.exceptions[0]
                logging.error(f"Exception details: {str(e)}")

    async def homepage(request: Request):
        return PlainTextResponse("Homepage")

    async def not_found(request: Request, exc: StarletteHTTPException):
        return HTMLResponse(content='404', status_code=exc.status_code)

    async def server_error(request: Request, exc: StarletteHTTPException):
        return HTMLResponse(content='500', status_code=exc.status_code)

    async def http_exception(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    exception_handlers = {
        404: not_found,
        500: server_error,
        StarletteHTTPException: http_exception
    }

    middleware = [
        Middleware(
            TrustedHostMiddleware,
            allowed_hosts=['*'],
        )
        #Middleware(HTTPSRedirectMiddleware)
    ]

    # Create Starlette routes for SSE and message handling
    routes = [
        Route("/", endpoint=homepage, methods=["GET"]),
        Route("/sse", endpoint=handle_sse, methods=["GET", "POST"]),
        Mount("/messages/", app=handle_post_message),
        #WebSocketRoute("/ws", endpoint=WebsocketServer)
    ]

    # Create and run Starlette app
    app = Starlette(debug=True,routes=routes) #, exception_handlers=exception_handlers, middleware=middleware)   
    app.add_middleware(AuthenticationMiddleware)

if (not is_azure_environment()):
    # Run the app locally
    #asyncio.run(uvicorn.run(app, host="0.0.0.0", port=mcp_port, log_level="debug", timeout_keep_alive=60))
    uvicorn.run(app, host="0.0.0.0", port=mcp_port, log_level="debug", timeout_keep_alive=60)
    
