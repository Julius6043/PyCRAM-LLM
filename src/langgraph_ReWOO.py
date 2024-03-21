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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()


def format_docs(docs):
    text = "\n\n---\n\n".join([d.page_content for d in docs])

    pattern = r"Next \n\n.*?\nBuilds"
    pattern2 = r"pycram\n          \n\n                latest\n.*?Edit on GitHub"
    # Ersetzen des gefundenen Textabschnitts durch einen leeren String
    filtered_text = re.sub(pattern, "", text, flags=re.DOTALL)
    filtered_text2 = re.sub(pattern2, "", filtered_text, flags=re.DOTALL)
    bereinigter_text = re.sub(r"\n{3,}", "\n\n", filtered_text2)
    return bereinigter_text


class ReWOO(TypedDict):
    task: str
    world: str
    plan_string: str
    steps: List
    results: dict
    result: str


llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
llm3 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = """You are a renowned AI engineer and programmer. You receive world knowledge and a task, command, or question and are to develop a plan for creating PyCramPlanCode for a robot that enables the robot to perform the task. For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You can store the evidence in a variable #E, which can be called upon by other tools later. (Plan, #E1, Plan, #E2, Plan, ...).
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

This is an example of a PyCramPlanCode:
'#Imports
from pycram.bullet_world import BulletWorld, Object
from pycram.process_module import simulated_robot
from pycram.designators.motion_designator import *
from pycram.designators.location_designator import *
from pycram.designators.action_designator import *
from pycram.designators.object_designator import *
from pycram.enums import ObjectType

#Bulletword Definition
world = BulletWorld()

#Objects
kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf')
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', position=[1.4, 1, 0.95])

#Object Designators
cereal_desig = ObjectDesignatorDescription(names=['cereal'])
kitchen_desig = ObjectDesignatorDescription(names=['kitchen'])
robot_desig = ObjectDesignatorDescription(names=['pr2']).resolve()

#The 'with simulated_robot:'-Block
with simulated_robot:
    #AktionDesignators
    ParkArmsAction([Arms.BOTH]).resolve().perform()

    MoveTorsoAction([0.3]).resolve().perform()
    
    #SemanticCostmapLocation
    pickup_pose = CostmapLocation(target=cereal_desig.resolve(), reachable_for=robot_desig).resolve()
    pickup_arm = pickup_pose.reachable_arms[0]

    #AktionDesignators
    NavigateAction(target_locations=[pickup_pose.pose]).resolve().perform()

    PickUpAction(object_designator_description=cereal_desig, arms=[pickup_arm], grasps=['front']).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()
        
    #SemanticCostmapLocation
    place_island = SemanticCostmapLocation('kitchen_island_surface', kitchen_desig.resolve(), cereal_desig.resolve()).resolve()

    place_stand = CostmapLocation(place_island.pose, reachable_for=robot_desig, reachable_arm=pickup_arm).resolve()
    
    #AktionDesignators
    NavigateAction(target_locations=[place_stand.pose]).resolve().perform()

    PlaceAction(cereal_desig, target_locations=[place_island.pose], arms=[pickup_arm]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

#BulletWorld Close
world.exit()'


The corresponding plan leading to this PyCramPlanCode above might look like this:
World knowledge: [kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', position=[1.4, 1, 0.95]))]
Task: Can you place the cereal on the kitchen island?
Plan 1: Determine the basics of PyCram, including creating the simulated world and initializing objects. #E1 = Retrieve[PyCram basics]
Plan 2: Research how object and robot designators are used in PyCram. #E2 = Retrieve[Use of designators in PyCram]
Plan 3: Investigate how movement and action modules in PyCram are used for navigation, grasping, and placing objects. #E3 = Retrieve[Movement and action modules in PyCram]
Plan 4: Research best practices for implementing action sequences in PyCram, including error handling and state checking. #E4 = Retrieve[Implementation of action sequences in PyCram]
Plan 5: Determine the best approach to properly close the simulated environment and release resources in PyCram. #E5 = Retrieve[Ending the simulation and resource management in PyCram]

Begin!
Describe your plans with rich details. Each plan should follow only one #E. You do not need to consider how PyCram is installed and set up in the plans, as this is already given.

World knowledge: {world}
Task: {task}"""


# Regex to match expressions of the form E#... = ...[...]
regex_pattern = r"Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | llm


def get_plan(state: ReWOO):
    task = state["task"]
    world = state["world"]

    result = planner.invoke({"task": task, "world": world})
    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    print(matches)
    return {"steps": matches, "plan_string": result.content}


search = DuckDuckGoSearchResults()


def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


# retriver chain


###
retriever = get_retriever(2, 5)
re_chain = retriever | format_docs

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
    | llm3
    | StrOutputParser()
)


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


solve_prompt = """You are a professional programmer, specialized on writing PycramRoboterPlans. To write the code, we have made step-by-step Plan and \
retrieved corresponding evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information. The evidence are examples and information to write PyCramCode so use them with caution and only use the world knowledge for specific world information. But also be conscious about you hallucinating and therefore use evidence and example code as strong inspiration.

Plan with evidence and examples:
<Plan>
{plan}
</Plan>

Now create the PyCramPlanCode for the quenstion or task according to provided evidence above and the world knowledge. Respond with nothing other than the generated PyCramPlan python code.
PyCramPlanCode follow the following structure:
<Plan structure>
Imports
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot:'-Block (defines the Aktions and moves of the Robot)
    AktionDesignators
    SematicCostmapLocation
BulletWorld Close
</Plan structure>


Task: {task}
World knowledge: {world}
Response:
"""


def solve(state: ReWOO):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = solve_prompt.format(plan=plan, task=state["task"], world=state["world"])
    result = llm.invoke(prompt)
    return {"result": result.content}


def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"


graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.set_entry_point("plan")

app = graph.compile()

task = """ Kannst du das Müsli aufnehmen und 3 Schritte rechts wieder abstellen?
"""
world = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', position=[1.4, 1, 0.95]))]
"""


def stream_rewoo(task, world):
    for s in app.stream({"task": task, "world": world}):
        print(s)
        print("---")

    print(s[END]["result"])
    return s[END]["result"]


##result = chain_docs.invoke("PyCram Grundlagen")
# result = chain_docs.invoke("PyCram Grundlagen")
# print(result)
