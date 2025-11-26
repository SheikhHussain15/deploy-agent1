from agents import Agent, RunConfig, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv
import chainlit as cl
from agents.tool import function_tool

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Safety check
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing. Add it in Railway environment variables.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

agent = Agent(
    name="Shariah advisor",
    instructions="You are a shariah advisor who solves problems about Islamic finance and shariah compliance."
)


@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the Shariah advisor chat! Ask me anything related to shariah.").send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    result = await Runner.run(
        agent,
        input=history,
        run_config=config
    )

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()
