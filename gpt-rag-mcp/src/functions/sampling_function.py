from semantic_kernel.functions import kernel_function
from typing import Annotated
from mcp.server.lowlevel import Server

@kernel_function(
    name="run_prompt",
    description="Run the prompts for a full set of release notes based on the PR messages given.",
)
async def sampling_function(
    messages: Annotated[str, "The list of PR messages, as a string with newlines"],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    server: Annotated[Server | None, "The server session", {"include_in_function_choices": False}] = None,
) -> str:
    if not server:
        raise ValueError("Request context is required for sampling function.")
    sampling_response = await server.request_context.session.create_message(
        messages=[
            types.SamplingMessage(role="user", content=types.TextContent(type="text", text=messages)),
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        model_preferences=types.ModelPreferences(
            hints=[types.ModelHint(name="gpt-4o-mini")],
        ),
    )
    return sampling_response.content.text