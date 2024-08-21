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
from langchain_core.pydantic_v1 import BaseModel, Field
from helper_func import format_docs, format_code, format_examples, format_example, llm, llm_GP, llm_mini
from run_llm_local import run_llama3_remote

# Load environment variables for secure access to configuration settings
load_dotenv()


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


###pyDanticToolParser
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Import statements of the code")
    code: str = Field(description="Just the Code block without including import statements")
    description = "Schema for code solutions for robot tasks."


llm_with_tool = llm.with_structured_output(code)

# Define a long and complex prompt template for generating plans...
prompt = r"""You are a renowned AI engineer and programmer. You receive world knowledge, a task, an error-code and a 
code solution. The code solution was created by another LLM Agent like you to the given task and world knowledge. The 
code was already executed resulting in the provided error message. Your task is to develop a sequenz of plans to geather 
resources and correct the given PyCramPlanCode. PyCramPlanCode is a plan instruction for a robot that should enable the 
robot to perform the provided high level task. For each plan, indicate which external tool, along with the input for 
the tool, is used to gather evidence. You can store the evidence in a variable #E, which can be called upon by other 
tools later. (Plan, #E1, Plan, #E2, Plan, ...). Don't use **...** to highlight something.

The tools can be one of the following: 
(1) Retrieve[input]: A vector database retrieval system containing the documentation of PyCram. Use this tool when you need information about PyCram functionality. The input should be a specific search query as a detailed question. 
(2) LLM[input]: A pre-trained LLM like yourself. Useful when you need to act with general information and common sense. Prefer it if you are confident you can solve the problem yourself. The input can be any statement.
(3) CodeRetrieve[input]: A vector database retriever to search and look directly into the PyCram package code. As input give the exact Function and a little description.
(4) URDF[input]: A database retriver which returns the URDF file text. Use this tool when you need information about the URDF files used in the world. Provide the URDF file name as input.

PyCramPlanCode follow the following structure:
Imports
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot'-Block (defines the Actions and moves of the Robot)
    Start with this two Code lines in this block:
        'ParkArmsAction([Arms.BOTH]).resolve().perform()
        MoveTorsoAction([0.25]).resolve().perform()'
    ActionDesignators
    SematicCostmapLocation
BulletWorld Close


Here is an PyCramPlanCode with its corresponding correction plan (use them just as examples to learn the plan structure):
Failed PyCramPlanCode: 
<failed_code>
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
</failed_code>
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
Plan 1: Research the Code of the function SemanticCostmapLocation. #E1 = CodeRetrieve[SemanticCostmapLocation]
Plan 2: Verify the correct usage of the resolve method on Object instances in PyCram. This will help us understand whether kitchen.resolve() and cereal.resolve() in the line that caused the error are being used appropriately. #E2 = Retrieve[How to correctly use the 'resolve' method with Object instances in PyCram] 
Plan 3: Confirm the proper way to reference PyCram objects when setting up locations or actions that involve these objects. This ensures that we correctly interact with kitchen and cereal objects in our plan, especially in context to SemanticCostmapLocation. #E3 = Retrieve[How to reference objects for actions and locations in PyCram without using the 'resolve' method.]
Plan 4: Acquire knowledge on the proper instantiation and usage of SemanticCostmapLocation. Understanding its parameters and usage will help us correctly position the cereal on the kitchen island. #E4 = Retrieve[Correct instantiation and usage of SemanticCostmapLocation in PyCram.]
Plan 5: Ensure we have a clear understanding of how to use the PlaceAction correctly, especially how to specify the object_to_place and target_locations. This will correct the final action where the cereal is to be placed 3 steps to the right. #E5 = Retrieve[How to use PlaceAction correctly in PyCram, including specifying object_to_place and target_locations.]
Plan 6: Given the task to move the cereal 3 steps to the right, we need to understand how to calculate the new position based on the current position of the cereal. This will involve modifying the target pose for the MoveMotion or directly in the PlaceAction to achieve the desired placement. #E6 = LLM[Given an object's current position, calculate a new position that is 3 steps to the right in a coordinate system.] 
--- end of example ---

Below you find all the Infos for your current task.
Describe your plans with rich details. Each plan should follow only one #E and it should be exactly in the given structure. Do not include other characters for highlighting because this can break the Regex Pattern.
Don't use any highlighting with markdown and co. You do not need to consider how PyCram is installed and set up in the plans, as this is already given.
Your task is to make a plan to correct the error but also include a general check up for unseen errors in the plan.

Failed PyCramPlanCode: {code}
---
Corresponding error: {error}
---
World knowledge: {world}
---
Task: {task}
---
Plan:"""

# Regex to match expressions of the form E#... = ...[...]
# Regex pattern to extract information from the plan format specified in the prompt
regex_pattern = (
    r"\**Plan\s*\d*:\**\s*(.+?)\s*\**(#E\d+)\**\s*=\s*(\w+)\s*\[([^\]]+)\].*"
)
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | llm_GP

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


# Function to determine the current task based on state
def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


# More complex template for tutorial writing, generating comprehensive documentation
prompt_retriever_chain = ChatPromptTemplate.from_template(
    """You are an experienced technical writer and coding educator, specializing in creating comprehensive guides for implementing specific tasks and workflows using the PyCram framework. Your task is to thoroughly explain how to accomplish the given task within PyCram, based on the provided context. You should research and extract all relevant information, summarizing and organizing it in a way that enables another LLM agent to efficiently implement the workflow in code.

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

Use up to 4000 tokens for your explanation.
Ensure that all necessary code examples are complete and well-explained.
Organize the information in a clear, logical order to facilitate implementation by another LLM agent."""
)

# GPT
# Chain to retrieve documents using a vector store retriever and formatting them
retriever_gpt = get_retriever(2, 8)

chain_docs_gpt = (
    {"context": retriever_gpt | format_docs, "task": RunnablePassthrough()}
    | prompt_retriever_chain
    | llm_mini
    | StrOutputParser()
)

# PyCram Code Retriever
prompt_retriever_code = ChatPromptTemplate.from_template(
    """You are an experienced technical writer and coding educator, specializing in creating detailed and precise tutorials. 
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
)
retriever_code = get_retriever(1, 6)

chain_code = (
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
    elif tool == "CodeRetrieve":
        result = chain_code.invoke(tool_input)
    elif tool == "Retrieve":
        result = chain_docs_gpt.invoke(tool_input)
    elif tool == "URDF":
        urdf_retriever = get_retriever(4, 1)
        result = urdf_retriever.invoke(tool_input)
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
Imports #Import Designators with *
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

{code_example}


Failed PyCramPlanCode: {code}
---
Corresponding error: {error}
---
World knowledge: {world}
---
Task: {task}
---
Corrected Code Response (Response structure: prefix("Description of the problem and approach"); imports()"Import statements of the code"); code(Code block not including import statements)):
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
    # code_example_filler = """Here is also an example of a similar PyCRAM plan code (use this as a semantic and syntactic example for the code structure and not for the world knowledge):
    # <Code example>""" + code_example + "\n</Code example>"
    code_example_filler = ""
    prompt = solve_prompt.format(
        plan=plan,
        error=state["error"],
        code_example=code_example_filler,
        code=state["code"],
        task=task,
        world=state["world"],
    )
    result_chain = llm_with_tool
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


def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]


# Function to stream the execution of the application
def stream_rewoo_check(task, world, code_input, error):
    for s in app.stream(
        {"task": task, "world": world, "code": code_input, "error": error}
    ):
        print(s)
        print("---")
    if "result" in s:
        final_result = s["result"]
    else:
        final_result = s["solve"]["result"]
    result_print = final_result.imports + "\n" + final_result.code
    print(result_print)
    return final_result


# Example task and world knowledge strings...
task_test = """ Kannst du das Müsli aufnehmen und 3 Schritte rechts wieder abstellen?"""
world_test = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', 
ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', 
pose=Pose([1.4, 1, 0.95])))]"""
##result = chain_docs.invoke("PyCram Grundlagen")
# result = chain_docs.invoke("PyCram Grundlagen")
# print(result)
# stream_rewoo(task, world)

# result = ((retriever_code | format_code).invoke("""SemanticCostmapLocation"""))
# print(result)
