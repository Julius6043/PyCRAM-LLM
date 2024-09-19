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
from helper_func import (
    format_docs,
    format_code,
    format_examples,
    format_example,
    llm,
    llm_GP,
    llm_mini,
)
from prompts import rewoo_planner, chain_docs_docu, chain_docs_code, rewoo_solve_prompt

# from run_llm_local import run_llama3_remote
import sys


# Load environment variables for secure access to configuration settings
load_dotenv()


# TypedDict for structured data storage, defining the structure of the planning state
class ReWOO(TypedDict):
    task: str
    world: str
    plan_string: str
    steps: List
    results: dict
    result: any
    result_plan: str


###PyDanticToolParser
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Import statements of the code")
    code: str = Field(
        description="Just the Code block without including import statements"
    )
    description = "Schema for code solutions for robot tasks."


# LLM with tool and enforce invocation
llm_with_tool = llm.with_structured_output(code)


# Regex to match expressions of the form E#... = ...[...]
# Regex pattern to extract information from the plan format specified in the prompt
regex_pattern = (
    r"\**Plan\s*\d*:\**\s*(.+?)\s*\**(#E\d+)\**\s*=\s*(\w+)\s*\[([^\]]+)\].*"
)


# Retriever function for the examples
retriever_examples = get_retriever(3, 1)  # the num variable can be changed for few-shot
re_chain_examples = retriever_examples | format_examples


# Function to get the plan from the state using regex pattern matching
def get_plan(state: ReWOO):
    task = state["task"]
    world = state["world"]
    examples = re_chain_examples.invoke(task)
    result = rewoo_planner.invoke({"task": task, "world": world, "examples": examples})

    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    return {"steps": matches, "plan_string": result.content, "results": None}


# Function to determine the current task based on state
def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


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
        result = chain_docs_docu.invoke(tool_input)
    elif tool == "Code":
        result = chain_docs_code.invoke(tool_input)
    elif tool == "URDF":
        urdf_retriever = get_retriever(4, 1)
        result = urdf_retriever.invoke(tool_input)
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


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
    prompt_solve = rewoo_solve_prompt.format(
        plan=plan, code_example=code_example_filler, task=task, world=state["world"]
    )
    result_chain = llm_with_tool
    # result_chain = llm_GP
    result = result_chain.invoke(prompt_solve)
    return {"result": result, "result_plan": plan}


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
        filled_plan = s["result_plan"]
    else:
        final_result = s["solve"]["result"]
        filled_plan = s["solve"]["result_plan"]
    result_print = final_result.imports + "\n" + final_result.code
    print(result_print)
    return final_result, plan, filled_plan


# Example task and world knowledge strings...
task_test = """Kannst du das Müsli aufnehmen und neben den Kühlschrank abstellen?"""
world_test = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
"""
##result = chain_docs.invoke("PyCram Grundlagen")
# result = chain_docs.invoke("PyCram Grundlagen")
# result = re_chain_example_solve.invoke(task_test)
# result = stream_rewoo(task_test, world_test)
# result = count_tokens("gpt-4", task_test)
# print(result)
