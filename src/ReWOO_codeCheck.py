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
import requests

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
prompt = """You are a renowned AI engineer and programmer. You receive world knowledge, a task, an error-code and a 
code solution. The code solution was created by another LLM Agent like you to the given task and world knowledge. The 
code was already executed resulting in the provided error message. Your task is to develop a plan to geather 
resources and correct given PyCramPlanCode. PyCramPlanCode is a plan instruction for a robot that should enable the 
robot to perform the provided high level task. For each plan, indicate which external tool, along with the input for 
the tool, is used to gather evidence. You can store the evidence in a variable #E, which can be called upon by other 
tools later. (Plan, #E1, Plan, #E2, Plan, ...). 
Use a format that can be recognized by the following regex pattern: 
Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*([^]+)]

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


Here is an PyCramPlanCode with its corresponding correction plan (use them just as examples to learn the plan structure):
Failed PyCramPlanCode: 
<code>
from pycram.bullet_world import BulletWorld, Object
from pycram.process_module import simulated_robot
# The import statements for designators are incomplete.
from pycram.designators.motion_designator import *
from pycram.designators.location_designator import *
from pycram.designators.action_designator import *
from pycram.designators.object_designator import *
from pycram.enums import ObjectType, Arms
from pycram.pose import Pose

# Initialize the BulletWorld
world = BulletWorld()

# Add objects to the world
kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf')
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95]))

# Define designators
cereal_desig = ObjectDesignatorDescription(names=['cereal'])
robot_desig = ObjectDesignatorDescription(names=['pr2']).resolve()

# Activate the simulated robot
with simulated_robot:
    # Robot actions
    ParkArmsAction([Arms.BOTH]).resolve().perform()
    MoveTorsoAction([0.25]).resolve().perform()

    # Determine the pick-up position for the cereal
    pickup_pose = CostmapLocation(target=cereal.resolve(), reachable_for=robot_desig).resolve()
    pickup_arm = pickup_pose.reachable_arms[0]

    # Navigate to the cereal and pick it up
    NavigateAction(target_locations=[pickup_pose.pose]).resolve().perform()
    PickUpAction(object_designator_description=cereal_desig, arms=[pickup_arm], grasps=['front']).resolve().perform()

    # Determine the target position and move there
    target_pose = Pose([pickup_pose.pose.position.x + 3, pickup_pose.pose.position.y, pickup_pose.pose.position.z], pickup_pose.pose.orientation)
    move_motion = MoveMotion(target=target_pose)
    move_motion.perform()

    # Place the cereal on the kitchen island
    place_island = SemanticCostmapLocation('kitchen_island_surface', kitchen.resolve(), cereal.resolve()).resolve()
    PlaceAction(object_to_place=cereal, target_locations=[place_island.pose], arms=[pickup_arm]).resolve().perform()

# Exit the simulation
world.exit()
</code>
---
Corresponding error: 
AttributeError                            Traceback (most recent call last)
Cell In[1], line 43
     40     move_motion.perform()
     42     # Cerealien auf der Kücheninsel ablegen
---> 43     place_island = SemanticCostmapLocation('kitchen_island_surface', kitchen.resolve(), cereal.resolve()).resolve()
     44     PlaceAction(object_designator_description=cereal_desig, target_locations=[place_island.pose], arms=[pickup_arm]).resolve().perform()
     46 # Simulation beenden

AttributeError: 'Object' object has no attribute 'resolve'
---
World knowledge: 
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
---
Task: 
Kannst du das Müsli aufnehmen und 3 Schritte rechts wieder abstellen?
---
Corresponding output plan: 
Plan 1: Verify the correct usage of the resolve method on Object instances in PyCram. This 
will help us understand whether kitchen.resolve() and cereal.resolve() in the line that caused the error are being 
used appropriately. #E1 = Retrieve[How to correctly use the 'resolve' method with Object instances in PyCram] 
Plan 2: Confirm the proper way to reference PyCram objects when setting up locations or actions that involve these objects. 
This ensures that we correctly interact with kitchen and cereal objects in our plan, especially in context to 
SemanticCostmapLocation. #E2 = Retrieve[How to reference objects for actions and locations in PyCram without using 
the 'resolve' method.]
Plan 3: Acquire knowledge on the proper instantiation and usage of SemanticCostmapLocation. Understanding its 
parameters and usage will help us correctly position the cereal on the kitchen island. #E3 = Retrieve[Correct 
instantiation and usage of SemanticCostmapLocation in PyCram.]
Plan 4: Ensure we have a clear understanding of how to use the PlaceAction correctly, especially how to specify the 
object_to_place and target_locations. This will correct the final action where the cereal is to be placed 3 steps to 
the right. #E4 = Retrieve[How to use PlaceAction correctly in PyCram, including specifying object_to_place and 
target_locations.]
Plan 5: Given the task to move the cereal 3 steps to the right, we need to understand how to calculate the new 
position based on the current position of the cereal. This will involve modifying the target pose for the MoveMotion 
or directly in the PlaceAction to achieve the desired placement. #E5 = LLM[Given an object's current position, 
calculate a new position that is 3 steps to the right in a coordinate system.] 
--- end of example ---

Begin!
Describe your plans with rich details. Each plan should follow only one #E and it should be exactly in the given structure. Don't use any highlighting with markdown and co. You do not need to consider how PyCram is installed and set up in the plans, as this is already given.

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
retriever_haiku = get_retriever(2, 8)

# More complex template for tutorial writing, generating comprehensive documentation
prompt_retriever_chain = ChatPromptTemplate.from_template(
    """You are an professional tutorial writer and coding educator. Given the search query, write a detailed, 
    structured, and comprehensive coding documentation for this question and the topic based on all the important 
    information from the following context: {context}
    
    Search query: {task} 
    Use at least 4000 tokens for the output and adhere to the provided information. Incorporate 
    important code examples in their entirety. The installation and configuration of pycram is not important, because it 
    is already given."""
)
chain_docs_haiku = (
    {"context": retriever_haiku | format_docs, "task": RunnablePassthrough()}
    | prompt_retriever_chain
    | llm_AH
    | StrOutputParser()
)

# GPT
# Chain to retrieve documents using a vector store retriever and formatting them
retriever_gpt = get_retriever(2, 6)

chain_docs_gpt = (
        {"context": retriever_gpt | format_docs, "task": RunnablePassthrough()}
        | prompt_retriever_chain
        | llm3
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
        try:
            result = chain_docs_haiku.invoke(tool_input)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                result = chain_docs_gpt.invoke(tool_input)
            else:
                raise e
    elif tool == "Statement":
        result = tool_input
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


# Solve function to generate PyCramPlanCode based on the plan and its steps...
solve_prompt = """You are a professional programmer, specialized on correcting PycramRoboterPlanCode. To repair the 
code, we have made step-by-step Plan and retrieved corresponding evidence to each Plan. The evidence are examples 
and information to write PyCramCode so use them with caution because long evidence might contain irrelevant 
information and only use the world knowledge for specific world information. Also be conscious about you 
hallucinating and therefore use evidence and example code as strong inspiration.

Plan with evidence and examples:
<Plan>
{plan}
</Plan>

Now create the new properly functioning PyCramPlanCode Version for the task according to provided evidence above and 
the world knowledge. Respond with nothing other than the generated PyCramPlan python code. 
PyCramPlanCode follow the following structure:
<PyCramPlan structure>
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
</PyCramPlan structure>


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
    result_chain = llm_with_tool | parser_tool
    result = result_chain.invoke(prompt)
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
task_test = """ Kannst du das Müsli aufnehmen und 3 Schritte rechts wieder abstellen?"""
world_test = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', 
ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', 
pose=Pose([1.4, 1, 0.95])))]"""


def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]


# Function to stream the execution of the application
def stream_rewoo_check(task, world, code_input, error):
    for s in app.stream({"task": task, "world": world, "code": code_input, "error": error}):
        print(s)
        print("---")
    result = s[END]["result"]
    result_print = result[0].imports + "\n\n #Code start: \n" + result[0].code
    print(result_print)
    return result


##result = chain_docs.invoke("PyCram Grundlagen")
# result = chain_docs.invoke("PyCram Grundlagen")
# print(result)
# stream_rewoo(task, world)
