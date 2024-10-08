# Import necessary libraries and modules
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, List

import re
from langgraph.graph import StateGraph, END
from vector_store_SB import get_retriever
from langchain_core.pydantic_v1 import BaseModel, Field
from helper_func import (
    format_docs,
    format_code,
    format_examples,
    format_example,
    llm,
    llm_solver
)
from prompts import rewoo_planner, chain_docs_docu, chain_docs_code, rewoo_solve_prompt, urdf_tool

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
llm_with_tool = llm_solver.with_structured_output(code)

# Regex to match expressions of the form E#... = ...[...]
# Regex pattern to extract information from the plan format specified in the prompt
regex_pattern = (
    r"\**Plan\s*\d*:\**\s*(.+?)\s*\**(#E\d+\s)\**\s*=\s*(\w+)\s*\[([^\]]+)\].*"
)

# planer and solver definition
planner = rewoo_planner
solve_prompt = rewoo_solve_prompt


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
    print(matches)
    return {"steps": matches, "plan_string": result.content, "results": None}


# Function to determine the current task based on state
def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


async def async_tool_execution(state: ReWOO):
    """Worker node that executes the tools concurrently."""
    _step = _get_current_task(state)

    async def run_tool(step):
        _, step_name, tool, tool_input = step
        _results = state["results"] or {}
        task = state["task"]
        world = state["world"]
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)

        if tool == "LLM":
            result = await llm.ainvoke(tool_input)
        elif tool == "Retrieve":
            result = await chain_docs_docu.ainvoke({"task": tool_input, "instruction": task, "world": world})
        elif tool == "Code":
            result = await chain_docs_code.ainvoke({"task": tool_input, "instruction": task, "world": world})
        elif tool == "URDF":
            urdf_retriever = get_retriever(4, 1, {"source": tool_input})
            files = await urdf_retriever.ainvoke(tool_input)
            if len(files) >= 1:
                file = files[0].page_content
                result = await urdf_tool.ainvoke({"prompt": task, "world": world, "urdf": file})
            else:
                result = f"The URDF {tool_input} is not in the database."
        else:
            raise ValueError(f"Unknown tool: {tool}")

        return step_name, str(result)

    _results = state["results"] or {}
    tasks = [
        run_tool(step) for step in state["steps"][_step - 1 :]
    ]  # Adjust range based on the task steps.

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Update results in the state
    for step_name, result in results:
        _results[step_name] = result

    return {"results": _results}


retriever_example_solve = get_retriever(3, 1)
re_chain_example_solve = retriever_examples | format_example


# Function to solve the task using the generated plan
def solve(state: ReWOO):
    plan = ""
    logging_plan = ""
    task = state["task"]
    i = 1
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan {i}: {_plan}\n{step_name} = {tool}[{tool_input}]\n"
        logging_plan += f"Plan {i}: {_plan}\n{step_name} = {tool}[{tool_input}]\n\n--Next PLAN--\n"
        i += 1
    code_example = re_chain_example_solve.invoke(task)
    code_example_filler = (
        """Here is also an example of a similar PyCRAM plan code with the corresponding example plan (use this as a semantic and syntactic example for the code structure of a PyCRAM Plan and NOT for the world knowledge AND NOT as the task):
    <Code example>"""
        + code_example
        + "\n</Code example>"
    )
    # code_example_filler = ""
    prompt_solve = solve_prompt.format(
        plan=plan, code_example=code_example_filler, task=task, world=state["world"]
    )
    result_chain = llm_with_tool
    result = result_chain.invoke(prompt_solve)
    return {"result": result, "result_plan": logging_plan}


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
graph.add_node("tool", async_tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.set_entry_point("plan")

# Compile the graph into an executable application
app = graph.compile()


# Function to stream the execution of the application
async def stream_rewoo(task, world):
    plan = ""
    async for s in app.astream({"task": task, "world": world}):
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
    print("-------------")
    print(result_print)
    print("-------------")
    return final_result, plan, filled_plan


# Example task and world knowledge strings...
task_test = """Kannst du das Müsli aufnehmen und neben den Kühlschrank abstellen?"""
world_test = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
"""
##result = chain_docs.invoke("PyCram Grundlagen")
# result = chain_docs.invoke("PyCram Grundlagen")
# result = re_chain_example_solve.invoke(task_test)

# result = count_tokens("gpt-4", task_test)


#result, plan, filled_plan = asyncio.run(stream_rewoo(task_test, world_test))

#print(result)
