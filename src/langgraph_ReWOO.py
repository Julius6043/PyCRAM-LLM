# Import necessary libraries and modules
import os
import openai
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from vector_store_SB import get_retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
import requests
import anthropic

# Load environment variables for secure access to configuration settings
load_dotenv()


# Function to clean and format documents, removing unwanted patterns and reducing whitespace
def format_docs(docs):
    text = "\n\n---\n\n".join([d.page_content for d in docs])
    pattern = r"Next \n\n.*?\nBuilds"
    pattern2 = r"pycram\n          \n\n                latest\n.*?Edit on GitHub"
    filtered_text = re.sub(pattern, "", text, flags=re.DOTALL)
    filtered_text2 = re.sub(pattern2, "", filtered_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", filtered_text2)
    return cleaned_text


# Define a function to format documents for better readability
def format_examples(docs):
    # Join documents using a specified delimiter for separation
    return "\n\n<next example>\n".join([d.page_content for d in docs])


# Define a function to format documents for better readability
def format_example(example):
    text = example[0].page_content
    code_example = text.split("The corresponding plan")[0]
    return code_example


# TypedDict for structured data storage, defining the structure of the planning state
class ReWOO(TypedDict):
    task: str
    world: str
    plan_string: str
    steps: List
    results: dict
    result: any


# Instantiate Large Language Models with specific configurations
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
llm3 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_AH = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
llm_AO = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
llm_AS = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

is_retriever_model_haiku = True

###PyDanticToolParser
class code(BaseModel):
    """Code output"""

    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block without the import statements")


# Tool
code_tool_oai = convert_to_openai_tool(code)

# LLM with tool and enforce invocation
llm_with_tool = llm.bind(
    tools=[code_tool_oai],
    tool_choice={"type": "function", "function": {"name": "code"}},
)

# Parser
parser_tool = PydanticToolsParser(tools=[code])

# Define a long and complex prompt template for generating plans...
prompt = r"""You are a renowned AI engineer and programmer. You receive world knowledge and a task, command, 
or question and are to develop a plan for creating PyCramPlanCode for a robot that enables the robot to perform the 
task. Concentrate on using Action Designators over MotionDesignators. For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You 
can store the evidence in a variable #E, which can be called upon by other tools later. (Plan, #E1, Plan, #E2, Plan, 
...). Don't use **...** to highlight something.
Use a format that can be recognized by the following regex pattern: 
\**Plan\s*\d*:\**\s*(.+?)\s*\**(#E\d+)\**\s*=\s*(\w+)\s*\[([^\]]+)\].*

The tools can be one of the following: 
(1) Retrieve[input]: A vector database retrieval system containing the documentation of PyCram. Use this tool when you need information about PyCram functions. The input should be a specific search query as a detailed question. 
(2) LLM[input]: A pre-trained LLM like yourself. Useful when you need to act with general information and common sense. Prefer it if you are confident you can solve the problem yourself. The input can be any statement.

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

Begin! Describe your plans with rich details. Each plan should follow only one #E and it should be exactly in the given structure. Don't use any highlighting with markdown and co. You do not need to consider how 
PyCram is installed and set up in the plans, as this is already given.

World knowledge: {world}
Task: {task}"""

# Regex to match expressions of the form E#... = ...[...]
# Regex pattern to extract information from the plan format specified in the prompt
regex_pattern = r"\**Plan\s*\d*:\**\s*(.+?)\s*\**(#E\d+)\**\s*=\s*(\w+)\s*\[([^\]]+)\].*"
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | llm

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


# Instantiate a DuckDuckGo search tool
search = DuckDuckGoSearchResults()


# Function to determine the current task based on state
def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


# Haiku
# Chain to retrieve documents using a vector store retriever and formatting them
retriever_haiku = get_retriever(2, 7)

# More complex template for tutorial writing, generating comprehensive documentation
prompt_retriever_chain = ChatPromptTemplate.from_template(
    """You are an professional tutorial writer and coding educator specially for the PyCRAM toolbox. Given the search query, write a detailed, structured, and comprehensive coding documentation for this question and the topic based on all the important information from the following context from the PyCRAM documentation:
{context}

Search query: {task}
Use at 4000 tokens for the output and adhere to the provided information. Incorporate important code examples in their entirety. The installation and configuration of pycram is not important, because it is already given.
Think step by step and make sure the user can produce correct code based on your output.
"""
)
chain_docs_haiku = (
        {"context": retriever_haiku | format_docs, "task": RunnablePassthrough()}
        | prompt_retriever_chain
        | llm_AH
        | StrOutputParser()
)

# GPT
# Chain to retrieve documents using a vector store retriever and formatting them
retriever_gpt = get_retriever(2, 5)

# More complex template for tutorial writing, generating comprehensive documentation
chain_docs_gpt = (
        {"context": retriever_gpt | format_docs, "task": RunnablePassthrough()}
        | prompt_retriever_chain
        | llm3
        | StrOutputParser()
)


# Function to execute tools as per the generated plan
def tool_execution(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    global is_retriever_model_haiku
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = state["results"] or {}
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    if tool == "Google":
        result = search.invoke(tool_input)
    elif tool == "LLM":
        result = llm.invoke(tool_input)
    elif tool == "Retrieve":
        if is_retriever_model_haiku:
            try:
                result = chain_docs_haiku.invoke(tool_input)
            except anthropic.RateLimitError as e:
                is_retriever_model_haiku = False
                result = chain_docs_gpt.invoke(tool_input)
        else:
            trying = True
            result = retriever_gpt.invoke(tool_input)
            i = 6
            retriever_gpt_temp = get_retriever(2, i)
            chain_docs_gpt_temp = (
                    {"context": retriever_gpt_temp | format_docs, "task": RunnablePassthrough()}
                    | prompt_retriever_chain
                    | llm3
                    | StrOutputParser()
            )
            while trying:
                try:
                    result = chain_docs_gpt_temp.invoke(tool_input)
                    trying = False
                except openai.BadRequestError as e:
                    i -= 1
                    retriever_gpt_temp = get_retriever(2, i)
                    chain_docs_gpt_temp = (
                            {"context": retriever_gpt_temp | format_docs, "task": RunnablePassthrough()}
                            | prompt_retriever_chain
                            | llm3
                            | StrOutputParser()
                    )
    elif tool == "Statement":
        result = tool_input
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


# Solve function to generate PyCramPlanCode based on the plan and its steps...
solve_prompt = """You are a professional programmer, specialized on writing PycramRoboterPlans. To write the code, 
we have made step-by-step Plan and retrieved corresponding evidence to each Plan. Use them with caution since long 
evidence might contain irrelevant information. The evidences are examples and information to write PyCramPlanCode so 
use them with caution because long evidence might contain irrelevant information and only use the world knowledge 
for specific world information. Also be conscious about you hallucinating and therefore use evidence and example code 
as strong inspiration.

Plan with evidence and examples:
<Plan>
{plan}
</Plan>

Now create the PyCramPlanCode for the task according to provided evidence above and the world knowledge. 
Respond with nothing other than the generated PyCramPlan python code.
PyCramPlanCode follow the following structure:
<Plan structure>
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
</Plan structure>

{code_example}


Task: {task}
World knowledge: {world}
Code Response:
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
    #code_example_filler = """Here is also an example of a similar PyCRAM plan code (use this as a semantic and syntactic example for the code structure and not for the world knowledge):
    #<Code example>""" + code_example + "\n</Code example>"
    code_example_filler = ""
    prompt_solve = solve_prompt.format(plan=plan, code_example=code_example_filler, task=task, world=state["world"])
    result_chain = llm_with_tool | parser_tool
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

# Example task and world knowledge strings...
task_test = """Kannst du das Müsli aufnehmen und 3 Schritte rechts wieder abstellen?"""
world_test = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
"""


def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]


# Function to stream the execution of the application
def stream_rewoo(task, world):
    for s in app.stream({"task": task, "world": world}):
        print(s)
        print("---")
    result = s[END]["result"]
    result_print = result[0].imports + "\n" + result[0].code
    print(result_print)
    return result

##result = chain_docs.invoke("PyCram Grundlagen")
# result = chain_docs.invoke("PyCram Grundlagen")
#result = re_chain_example_solve.invoke(task_test)
#print(result)
#stream_rewoo(task, world)
