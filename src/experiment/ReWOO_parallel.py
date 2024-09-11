# Import necessary libraries and modules
import os
import asyncio
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


# Define a long and complex prompt template for generating plans...
prompt = """You are a renowned AI engineer and programmer. You receive world knowledge and a task. You use them to develop a detailed sequence of plans to creat PyCramPlanCode for a robot that enables the robot to perform the 
task step by step. Concentrate on using Action Designators over MotionDesignators. For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You 
can store the evidence in a variable #E, which can be called upon by other tools later. (Plan, #E1, Plan, #E2, Plan, 
...). Don't use **...** to highlight anything.

The tools can be one of the following: 
(1) Retrieve[input]: A vector database retrieval system containing the documentation of PyCram. Use this tool when you need information about PyCram functionality. The input should be a specific search query as a detailed question. 
(2) Code[input]: A LLM Agent with a database retriever for the PyCRAM code. Returns a function from the code base and provides a tutorial for it. Provide a function as input.
(3) URDF[input]: A database retriver which returns the URDF file text. Use this tool when you need information about the URDF files used in the world. Provide the URDF file name as input.

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
prompt_docs = """You are an experienced technical writer and coding educator, specializing in creating comprehensive guides for implementing specific tasks and workflows using the PyCram framework. 
Your task is to thoroughly explain how to accomplish the given task within PyCram, based on the provided context. 
You should research and extract all relevant information, summarizing and organizing it in a way that enables another LLM agent to efficiently implement the workflow in code.

Context:
{context}
--- End of Context ---

Task:
{task}

Task Overview and Objectives: Start by clearly defining the task or workflow in question. Explain the goal of the task and its relevance within the PyCram framework. Discuss any prerequisites or necessary setup steps.

Detailed Workflow Explanation: Provide a step-by-step guide on how to achieve the task using PyCram. Break down the process into logical steps, detailing each one. Include explanations of key concepts, relevant functions, and how they integrate with each other within the framework.

Code Examples and Implementation Guidance: Where applicable, include relevant code snippets or pseudocode that illustrates how each step of the process can be implemented in PyCram. These examples should be clear and fully explained so that they can be easily adapted to similar tasks.

Framework Integration and Concepts: Discuss how the task fits within the broader PyCram framework. Explain any essential concepts, components, or tools within PyCram that are crucial for understanding and completing the task.

Best Practices and Considerations: Provide best practices for implementing the task, including any potential challenges or common pitfalls. Offer recommendations on how to overcome these challenges and ensure optimal implementation.

Extensions and Alternatives: Explore possible extensions or variations of the task. Suggest alternative approaches if applicable, especially if the standard method may not suit all scenarios.

Important Notes:

Use 4000 tokens for your explanation.
Ensure that all necessary code examples are complete and well-explained.
Organize the information in a clear, logical order to facilitate implementation by another LLM agent.
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
prompt_code = """You are an experienced technical writer and coding educator, specializing in creating detailed and precise tutorials.
Your task is to craft a comprehensive guide on how to use the provided function within an PyCram framework, based on the given documentation and code context. 
You should not only explain the function itself but also describe its relationship with other relevant functions and components within the context.

Context:
{context}
--- End of Context ---

Task:
Function: {task}

Function Explanation and Contextualization: Begin with a detailed description of the function, including its syntax, parameters, and return values. Explain how this function is integrated into the framework and what role it plays within the overall context.

Code Examples and Implementation: Provide the full code of the function. Include relevant code snippets from the context that demonstrate the function in action. Explain step-by-step how the code works and how it can be adapted to solve similar tasks.

General Framework Functionality: Explain the fundamental functionality of the framework in relation to the given function. Discuss key concepts and principles of the framework that are necessary to understand the function and its application.

Best Practices and Recommendations: Provide guidance and best practices for effectively using the function and the framework. Mention potential pitfalls and how to avoid them.

Planning and Implementation for Developers: Design a clear plan for developers on how to implement the function in their own projects. Outline the necessary steps to correctly integrate and customize the function.

Extensions and Alternatives: Discuss possible extensions of the function as well as alternatives if the given function does not meet all requirements.

Important Notes:

Use up to 4000 tokens for the tutorial.
Incorporate all essential code examples in their entirety.
Think systematically and ensure that another LLM agent can produce correct code based on your output."""
prompt_retriever_code = ChatPromptTemplate.from_template(prompt_code)

retriever_code = get_retriever(1, 6)

chain_docs_code = (
    {"context": retriever_code | format_code, "task": RunnablePassthrough()}
    | prompt_retriever_code
    | llm_mini
    | StrOutputParser()
)


async def async_tool_execution(state: ReWOO):
    """Worker node that executes the tools concurrently."""
    _step = _get_current_task(state)

    async def run_tool(step):
        _, step_name, tool, tool_input = step
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)

        if tool == "LLM":
            result = await llm.invoke_async(tool_input)
        elif tool == "Retrieve":
            result = await chain_docs_gpt.invoke_async(tool_input)
        elif tool == "Code":
            result = await chain_docs_code.invoke_async(tool_input)
        elif tool == "URDF":
            urdf_retriever = get_retriever(4, 1)
            result = await urdf_retriever.invoke_async(tool_input)
        else:
            raise ValueError(f"Unknown tool: {tool}")

        return step_name, str(result)

    tasks = [
        run_tool(step) for step in state["steps"][_step - 1 :]
    ]  # Adjust range based on the task steps.

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Update results in the state
    for step_name, result in results:
        state["results"][step_name] = result

    return {"results": state["results"]}


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
solve_prompt = """You are a professional programmer, specialized on writing PyCram-Roboter-Plans. To write the code, 
we have made a step-by-step Plan and retrieved corresponding evidence to each Plan. The evidences are examples and information to write PyCramPlanCode so 
use them with caution because long evidence might contain irrelevant information and only use the world knowledge 
for specific world information. Also be conscious about you hallucinating and therefore use evidence and example code 
as strong inspiration.

Example of PyCRAM Plan Code with the corresponding example plan (use this only as a example how the PyCRAM Code to a plan should look like):
<Code_example>
{code_example}
</Code_example>

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
graph.add_node("tool", async_tool_execution)
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
