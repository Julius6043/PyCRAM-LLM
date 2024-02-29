from langchain_openai import ChatOpenAI
from langchain.agents import tool
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)


@tool
def retrive_info(input: str) -> str:
    """Returns context for the pyCramPlan from a VectorStore Database"""
    return input


tools = [retrive_info]

llm_with_tools = llm.bind_tools(tools)

MEMORY_KEY = "chat_history"
prompt_agent = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are very powerful assistant for creating pyCramPlans as Python Code.
            Generate with just the python code, given the provided world_knowledge, the chat_history and the user input. You can also use the tools when needed, to provide you with additional knowledge and examples to the specific problem. 
            World_Knowledge: {world_knowledge}""",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

chat_history = []

agent = (
    {
        "input": lambda x: x["input"],
        "world_knowledge": lambda x: x["world_knowledge"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt_agent
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def chat_with_agent(input: str, world_knowledge: str):
    global chat_history
    result = agent_executor.invoke(
        {
            "input": input,
            "world_knowledge": world_knowledge,
            "chat_history": chat_history,
        }
    )
