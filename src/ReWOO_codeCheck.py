# Import necessary libraries and modules
import os
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


# TypedDict for structured data storage, defining the structure of the planning state
class ReWOO(TypedDict):
    task: str
    world: str
    code: str
    error: str
    plan_string: str
    steps: List
    results: dict
    result: str


# Instantiate Large Language Models with specific configurations
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
llm3 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_AH = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)


###pyDanticToolParser
class code(BaseModel):
    """Code output"""

    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


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
prompt = """You are a renowned AI engineer and programmer. You receive world knowledge, a task, an error-code and a code solution. The code solution was created by another LLM Agent like you to the given task and world knowledge. The code was already executed resulting in the provided error message. 
Your task is to develop a plan to geather rescourcess and correct given PyCramPlanCode. PyCramPlanCode is a plan instruction for a robot that should enable the robot to perform the provided high level task. 
For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You can store the evidence in a variable #E, which can be called upon by other tools later. (Plan, #E1, Plan, #E2, Plan, ...).
Use a format that can be recognized by the following regex pattern:
Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*([^]+)]

The tools can be one of the following:
(1) Retrieve[input]: A vector database retrieval system containing the documentation of PyCram. Use this tool when you need information about PyCram functions.
The input should be a specific search query as a detailed question.
(2) LLM[input]: A pre-trained LLM like yourself. Useful when you need to act with general information and common sense. Prefer it if you are confident you can solve the problem yourself. The input can be any statement.


PyCramPlanCode follow the following structure:
Imports
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot:'-Block (defines the Aktions and moves of the Robot)
    Start with this two Code lines in this block:
        'ParkArmsAction([Arms.BOTH]).resolve().perform()
        MoveTorsoAction([0.25]).resolve().perform()'
    AktionDesignators
    SematicCostmapLocation
BulletWorld Close


Here is an PyCramPlanCode with its corresponding correction plan (use them just as examples to learn the plan structure):
Failed PyCramPlanCode: 
---
Corresponding error: 
---
World knowledge: 
---
Task: 
---
Corresponding output plan:

--- end of example ---

Begin!
Describe your plans with rich details. Each plan should follow only one #E. You do not need to consider how PyCram is installed and set up in the plans, as this is already given.

Failed PyCramPlanCode: {code}
---
Corresponding error: {error}
---
World knowledge: {world}
---
Task: {task}"""


# Regex to match expressions of the form E#... = ...[...]
# Regex pattern to extract information from the plan format specified in the prompt
regex_pattern = r"Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | llm

# Retriever function for the examples
retriever_examples = get_retriever(3, 2)
re_chain_examples = retriever_examples | format_examples


# Function to get the plan from the state using regex pattern matching
def get_plan(state: ReWOO):
    task = state["task"]
    world = state["world"]
    code = state["code"]
    error = state["error"]
    result = planner.invoke(
        {
            "task": task,
            "world": world,
            "code": code,
            "error": error,
        }
    )

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


# Chain to retrieve documents using a vector store retriever and formatting them
retriever = get_retriever(2, 8)
re_chain = retriever | format_docs

# More complex template for tutorial writing, generating comprehensive documentation
prompt_retriever_chain = ChatPromptTemplate.from_template(
    """You are an professional tutorial writer and coding educator. Given the search query, write a detailed, structured, and comprehensive coding documentation for this question and the topic based on all the important information from the following context:
{context}

Search query: {task}
Use at least 4000 tokens for the output and adhere to the provided information. Incorporate important code examples in their entirety. The installation and configuration of pycram is not important, because it is already given.
"""
)
chain_docs = (
    {"context": retriever | format_docs, "task": RunnablePassthrough()}
    | prompt_retriever_chain
    | llm_AH
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
    if tool == "Google":
        result = search.invoke(tool_input)
    elif tool == "LLM":
        result = llm.invoke(tool_input)
    elif tool == "Retrieve":
        result = chain_docs.invoke(tool_input)
    elif tool == "Statement":
        result = tool_input
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


# Solve function to generate PyCramPlanCode based on the plan and its steps...
solve_prompt = """You are a professional programmer, specialized on correcting PycramRoboterPlanCode. To repair the code, we have made step-by-step Plan and \
retrieved corresponding evidence to each Plan. The evidence are examples and information to write PyCramCode so use them with caution because long evidence might \
contain irrelevant information and only use the world knowledge for specific world information. Also be conscious about you hallucinating and therefore use evidence and example code as strong inspiration.

Plan with evidence and examples:
<Plan>
{plan}
</Plan>

Now create the new properly functioning PyCramPlanCode Version for the task according to provided evidence above and the world knowledge. Respond with nothing other than the generated PyCramPlan python code.
PyCramPlanCode follow the following structure:
<Plan structure>
Imports
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot:'-Block (defines the Aktions and moves of the Robot)
    Start with this two Code lines in this block:
        'ParkArmsAction([Arms.BOTH]).resolve().perform()
        MoveTorsoAction([0.25]).resolve().perform()'
    AktionDesignators
    SematicCostmapLocation
BulletWorld Close
</Plan structure>


Failed PyCramPlanCode: {code}
---
Corresponding error: {error}
---
World knowledge: {world}
---
Task: {task}
---
Response:
"""


# Function to solve the task using the generated plan
def solve(state: ReWOO):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = solve_prompt.format(
        plan=plan,
        error=state["error"],
        code=state["code"],
        task=state["task"],
        world=state["world"],
    )
    result = llm.invoke(prompt)
    return {"result": result.content}


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
task = """ Kannst du das Müsli aufnehmen und 3 Schritte rechts wieder abstellen?"""
world = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
"""


def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]


# Function to stream the execution of the application
def stream_rewoo_check(task, world, code, error):
    for s in app.stream({"task": task, "world": world, "code": code, "error": error}):
        print(s)
        print("---")
    result = _sanitize_output(s[END]["result"])
    print(result)
    return result


##result = chain_docs.invoke("PyCram Grundlagen")
# result = chain_docs.invoke("PyCram Grundlagen")
# print(result)
# stream_rewoo(task, world)
