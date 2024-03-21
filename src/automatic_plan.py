from vector_store_SB import get_retriever, load_in_vector_store
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs):
    return "\n\n<next example>\n".join([d.page_content for d in docs])


llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
###
retriever = get_retriever(3, 3)
re_chain = retriever | format_docs

prompt_retriever_chain = ChatPromptTemplate.from_template(
    """You are a renowned AI engineer and programmer. You receive PyCramPlanCode, world knowledge and a task. A task could be a question or a command. Use these information to generate a plan which can be used by a LLM to reverse enginere the PyCramPlanCode. 
PyCramPlanCode is a code that enables a robot to perform the task. For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You can store the evidence in a variable #E, which can be called upon by other tools later. (Plan, #E1, Plan, #E2, Plan, ...).
Use a format that can be recognized by the following regex pattern:
Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*([^]+)]

The tools can be one of the following:
(1) Retrieve[input]: A vector database retrieval system containing the documentation of PyCram. Use this tool when you need information about PyCram functions.
The input should be a specific search query as a detailed question.
(2) LLM[input]: A pre-trained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prefer it if you are confident you can solve the problem yourself. The input can be any statement.
(3) Statement[state]: No tool is used, simply formulate a statement directly. Useful when you just need to extract information from the input.

PyCramPlanCode follow the following structure:
Imports
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot:'-Block (defines the Aktions and moves of the Robot)
    AktionDesignators
    SematicCostmapLocation
BulletWorld Close


Here are some examples of PyCramPlanCode with its corresponding building plan (use them just as examples to learn the code format and the plan structure):
{examples}

--- end of examples ---

Begin!
Describe your plans with rich details. Each plan should follow only one #E. You do not need to consider how PyCram is installed and set up in the plans, as this is already given.

PyCramPlanCode: {pycram_plan_code}
World Knowledge: {world_knowledge}
Task: {task}
"""
)
chain_docs = (
    {
        "examples": itemgetter("task") | re_chain,
        "world_knowledge": itemgetter("world"),
        "pycram_plan_code": itemgetter("pycram_plan_code"),
        "task": itemgetter("task"),
    }
    | prompt_retriever_chain
    | llm
    | StrOutputParser()
)


def generate_plan(task, pycram_plan_code, world_knowledge, load_vector_store=False):
    result = chain_docs.invoke(
        {"task": task, "pycram_plan_code": pycram_plan_code, "world": world_knowledge}
    )
    if load_vector_store:
        full_result = (
            "PyCramPlanCode:\n"
            + "<code>\n"
            + pycram_plan_code
            + "\n</code>\n"
            + "World Knowledge:\n"
            + "<knowledge>\n"
            + world_knowledge
            + "\n</knowledge>\n"
            + "Task:"
            + task
        )
        load_in_vector_store([full_result], 3)
    return result


def load_case(case_plan):
    load_in_vector_store([case_plan], 3)
