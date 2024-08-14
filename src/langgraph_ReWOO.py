# Import necessary libraries and modules
import os
import openai
from dotenv import load_dotenv
from typing import TypedDict, List

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from vector_store_SB import get_retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from helper_func import format_docs, format_code, format_examples, format_example, llm, llm_GP, llm_mini

# from run_llm_local import run_llama3_remote
import sys
import tiktoken

# Load environment variables for secure access to configuration settings
load_dotenv()


# Count the Tokens in a string
def count_tokens(model_name, text):
    # Lade das Tokenizer-Modell
    encoding = tiktoken.encoding_for_model(model_name)

    # Tokenisiere den Text
    tokens = encoding.encode(text)

    # Anzahl der Tokens
    return len(tokens)


# TypedDict for structured data storage, defining the structure of the planning state
class ReWOO(TypedDict):
    task: str
    world: str
    plan_string: str
    steps: List
    results: dict
    result: any


###PyDanticToolParser
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Import statements of the code")
    code: str = Field(description="Just the Code block without including import statements")
    description = "Schema for code solutions for robot tasks."


# LLM with tool and enforce invocation
llm_with_tool = llm_GP.with_structured_output(code)

# Parser
parser_tool = PydanticToolsParser(tools=[code])

# Define a long and complex prompt template for generating plans...
prompt = r"""You are a renowned AI engineer and programmer. You receive world knowledge and a task. You use them to develop a detailed sequence of plans to creat PyCramPlanCode for a robot that enables the robot to perform the 
task step by step. Concentrate on using Action Designators over MotionDesignators. For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You 
can store the evidence in a variable #E, which can be called upon by other tools later. (Plan, #E1, Plan, #E2, Plan, 
...). Don't use **...** to highlight anything.

The tools can be one of the following: 
(1) Retrieve[input]: A vector database retrieval system containing the documentation of PyCram. Use this tool when you need information about PyCram functions. The input should be a specific search query as a detailed question. 
(2) LLM[input]: A pre-trained LLM like yourself. Useful when you need to act with general information and common sense. Prefer it if you are confident you can solve the problem yourself. The input can be any statement.
(3) Code[input]: A LLM Agent with a database retriever for the PyCRAM code. Returns a function from the code base and provides a tutorial for it. Provide a function as input
(4) URDF[input]: A database retriver which returns the URDF file text. Use this tool when you need information about the URDF files used in the world. Provide the URDF file name as input.

PyCramPlanCode follow the following structure:
Imports
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot:'-Block (defines the Actions and moves of the Robot)
    Start with this two Code lines in this block:
        'ParkArmsAction([Arms.BOTH]).resolve().perform()
        MoveTorsoAction([0.25]).resolve().perform()'
    ActionDesignators
    SematicCostmapLocation
BulletWorld Close


Here are some examples of PyCramPlanCode with its corresponding building plan (use them just as examples to learn the 
code format and the plan structure): 
{examples}

--- end of examples ---

Below you find all the Infos for your current task.
Describe your plans with rich details. Each plan should follow only one #E and it should be exactly in the given structure. Don't use any highlighting with markdown and co. You do not need to consider how 
PyCram is installed and set up in the plans, as this is already given.

World knowledge: {world}
Task: {task}"""

# Regex to match expressions of the form E#... = ...[...]
# Regex pattern to extract information from the plan format specified in the prompt
regex_pattern = (
    r"\**Plan\s*\d*:\**\s*(.+?)\s*\**(#E\d+)\**\s*=\s*(\w+)\s*\[([^\]]+)\].*"
)
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | llm_GP

# Retriever function for the examples
retriever_examples = get_retriever(3, 1)
re_chain_examples = retriever_examples | format_examples


# Function to get the plan from the state using regex pattern matching
def get_plan(state: ReWOO):
    task = state["task"]
    world = state["world"]
    examples = re_chain_examples.invoke(task)
    result = planner.invoke({"task": task, "world": world, "examples": examples})

    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    return {"steps": matches, "plan_string": result.content}


# Function to determine the current task based on state
def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1



# More complex template for tutorial writing, generating comprehensive documentation
prompt_docs = """You are an professional tutorial writer and coding educator specially for the PyCRAM toolbox. Given the search query, write a detailed, structured, and comprehensive coding documentation for this question and the topic based on all the important information from the following context from the PyCRAM documentation:
{context}
--- Context End ---

Search query: {task}

Use at 4000 tokens for the output and adhere to the provided information. Incorporate important 
code examples in their entirety. Think step by step and make sure the a other llm agent can produce correct code based on your output.
"""
prompt_retriever_chain = ChatPromptTemplate.from_template(prompt_docs)


# GPT
# Chain to retrieve documents using a vector store retriever and formatting them
retriever_gpt = get_retriever(2, 7)

# More complex template for tutorial writing, generating comprehensive documentation
chain_docs_gpt = (
    {"context": retriever_gpt | format_docs, "task": RunnablePassthrough()}
    | prompt_retriever_chain
    | llm_mini
    | StrOutputParser()
)

# PyCram Code Retriever
prompt_code = """You are an professional tutorial writer and coding educator specially for the PyCRAM toolbox. You get a function, 
your task is to search for it in the provided context code and write a tutorial for the function and likewise and near other functions.
Provide the full code of the provided function in your output.
Explain also the general functioning of PyCram in relation to this function with the Context Code.
Context: {context}
--- Context End ---

Function: {task} 

Use at 4000 tokens for the output and adhere to the provided information. Incorporate important 
code examples in their entirety. Think step by step and make sure the a other llm agent can produce correct code based on your output."""
prompt_retriever_code = ChatPromptTemplate.from_template(prompt_code)

retriever_code = get_retriever(1, 6)

chain_docs_code = (
    {"context": retriever_code | format_code, "task": RunnablePassthrough()}
    | prompt_retriever_code
    | llm_mini
    | StrOutputParser()
)


# Function to execute tools as per the generated plan
def tool_execution(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = state["results"] or {}
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    if tool == "LLM":
        result = llm.invoke(tool_input)
    elif tool == "Retrieve":
        # retriever_llama3 = get_retriever(2, 3)
        # re_llama = retriever_llama3 | format_docs
        # prompt_filled = prompt_docs.format(task=tool_input, context=re_llama.invoke(tool_input))
        result = chain_docs_gpt.invoke(tool_input)
    elif tool == "Code":
        result = chain_docs_code.invoke(tool_input)
    elif tool == "URDF":
        urdf_retriever = get_retriever(4, 1)
        result = urdf_retriever.invoke(tool_input)
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


# Solve function to generate PyCramPlanCode based on the plan and its steps...
solve_prompt = """You are a professional programmer, specialized on writing PycramRoboterPlans. To write the code, 
we have made a step-by-step Plan and retrieved corresponding evidence to each Plan. The evidences are examples and information to write PyCramPlanCode so 
use them with caution because long evidence might contain irrelevant information and only use the world knowledge 
for specific world information. Also be conscious about you hallucinating and therefore use evidence and example code 
as strong inspiration.

{code_example}

Plan with evidence and examples:
<Plan>
{plan}
</Plan>

Now create the PyCramPlanCode for the task according to provided evidence above and the world knowledge. 
Respond with nothing other than the generated PyCramPlan python code.
PyCramPlanCode follow the following structure:
<Plan structure>
Imports #Import Designators with *
#clear separation between code block
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot:'-Block (defines the Actions and moves of the Robot)
    Start with this two Code lines in this block:
        'ParkArmsAction([Arms.BOTH]).resolve().perform()
        MoveTorsoAction([0.25]).resolve().perform()'
    ActionDesignators
    SemanticCostmapLocation
BulletWorld Close
</Plan structure>


Task: {task}
World knowledge: {world}
Code Response (Response structure: prefix("Description of the problem and approach"); imports()"Import statements of the code"); code(Code block not including import statements)):
"""

retriever_example_solve = get_retriever(3, 1)
re_chain_example_solve = retriever_examples | format_example


# Function to solve the task using the generated plan
def solve(state: ReWOO):
    plan = ""
    task = state["task"]
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    code_example = re_chain_example_solve.invoke(task)
    code_example_filler = (
        """Here is also an related example of a similar PyCRAM plan code (use this as a semantic and syntactic example for the code structure and not for the world knowledge):
    <Code example>"""
        + code_example
        + "\n</Code example>"
    )
    # code_example_filler = ""
    prompt_solve = solve_prompt.format(
        plan=plan, code_example=code_example_filler, task=task, world=state["world"]
    )
    result_chain = llm_with_tool
    # result_chain = llm_GP
    result = result_chain.invoke(prompt_solve)
    return {"result": result}


# Function to route the graph based on the current state
def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"


# Initialize the state graph and add nodes for planning, tool execution, and solving
graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.set_entry_point("plan")

# Compile the graph into an executable application
app = graph.compile()


def _sanitize_output(text: str):
    if text.startswith("```python"):
        _, after = text.split("```python")
        return after.split("```")[0]
    else:
        return text


# Function to stream the execution of the application
def stream_rewoo(task, world):
    plan = ""
    for s in app.stream({"task": task, "world": world}):
        if "plan" in s:
            plan = s["plan"]["plan_string"]
        print(s)
        print("---")
    if "plan_string" in s and "result" in s:
        final_result = s["result"]
        plan = s["plan_string"]
    else:
        final_result = s["solve"]["result"]
    result_print = final_result.imports + "\n" + final_result.code
    print(result_print)
    return final_result, plan


# Example task and world knowledge strings...
task_test = """Kannst du das Müsli aufnehmen und neben den Kühlschrank abstellen?"""
world_test = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
"""
##result = chain_docs.invoke("PyCram Grundlagen")
# result = chain_docs.invoke("PyCram Grundlagen")
# result = re_chain_example_solve.invoke(task_test)
#result = stream_rewoo(task_test, world_test)
# result = count_tokens("gpt-4", task_test)
#print(result)
