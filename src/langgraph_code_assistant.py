from typing import Dict, TypedDict
from langgraph.graph import END, StateGraph
from operator import itemgetter
from vector_store_SB import get_retriever
from langgraph_ReWOO import stream_rewoo
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re

load_dotenv()

test_code = """
from pycram.bullet_world import BulletWorld, Object
from pycram.process_module import simulated_robot
from pycram.designators.motion_designator import *
from pycram.designators.location_designator import *
from pycram.designators.action_designator import *
from pycram.designators.object_designator import *
from pycram.pose import Pose
from pycram.enums import ObjectType

world = BulletWorld()

#Objects
kitchen = Object("kitchen", ObjectType.ENVIRONMENT, "kitchen.urdf")
robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf")
cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, "breakfast_cereal.stl", pose=Pose([1.4, 1, 0.95]))

#Object Designators
cereal_desig = ObjectDesignatorDescription(names=["cereal"])

with simulated_robot:
    # Initial actions
    ParkArmsAction([Arms.BOTH]).resolve().perform()
    MoveTorsoAction([0.25]).resolve().perform()

    # Pick up the cereal
    pickup_pose = CostmapLocation(target=cereal_desig.resolve(), reachable_for=robot).resolve()
    pickup_arm = pickup_pose.reachable_arms[0]
    NavigateAction(target_locations=[pickup_pose.pose]).resolve().perform()
    PickUpAction(object_designator_description=cereal_desig, arms=[pickup_arm], grasps=["front"]).resolve().perform()

    # Move 3 steps to the right
    motion_designator = MotionDesignator(direction="right", distance=3)
    motion_designator.resolve().perform()

    # Place the cereal
    place_pose = CostmapLocation(target=cereal_desig.resolve(), reachable_for=robot).resolve()
    NavigateAction(target_locations=[place_pose.pose]).resolve().perform()
    PlaceAction(cereal_desig, target_locations=[place_pose.pose], arms=[pickup_arm]).resolve().perform()

    # Park the robot's arms again
    ParkArmsAction([Arms.BOTH]).resolve().perform()

world.exit()
"""


def format_docs(docs):
    text = "\n\n---\n\n".join([d.page_content for d in docs])
    pattern = r"Next \n\n.*?\nBuilds"
    pattern2 = r"pycram\n          \n\n                latest\n.*?Edit on GitHub"
    # Ersetzen des gefundenen Textabschnitts durch einen leeren String
    filtered_text = re.sub(pattern, "", text, flags=re.DOTALL)
    filtered_text2 = re.sub(pattern2, "", filtered_text, flags=re.DOTALL)
    bereinigter_text = re.sub(r"\n{3,}", "\n\n", filtered_text2)
    return bereinigter_text


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


def generate(state: GraphState):
    """
    Generate a code solution based on LCEL docs and the input question
    with optional feedback from code execution tests

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    ## State
    state_dict = state["keys"]
    question = state_dict["question"]
    world = state_dict["world"]
    iter = state_dict["iterations"]

    ## Data model
    class code(BaseModel):
        """Code output"""

        prefix: str = Field(description="Description of the problem and approach")
        imports: str = Field(description="Code block import statements")
        code: str = Field(description="Code block not including import statements")

    ## LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # Content
    retriever = get_retriever(2, 10)
    re_chain = retriever | format_docs
    # Error information
    retriever_error = get_retriever(2, 2)
    re_chain_error = retriever_error | format_docs

    # Tool
    code_tool_oai = convert_to_openai_tool(code)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[code_tool_oai],
        tool_choice={"type": "function", "function": {"name": "code"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[code])

    ## Prompt
    template = """You are a coding assistant with expertise in PyCram, a python Roboter Language. \n Here is the most 
    important part of the PyCram regarding your coming task documentation: \n ------- \n {context} \n ------- \n Here 
    is a code written by an specially trained LLM Network: \n --- \n {code_rewoo}. \n --- \n Check the Code based on 
    the documentation and edit it only when you are 100 percent sure based on the information that there is a 
    mistake. Also ensure that the code can be executed with all required imports and variables defined. \n Be aware 
    the documentation are only examples, use world knowledge for specific world information. \n Structure the final 
    answer with a description of the code solution. \n Then list the imports. And finally list the functioning code 
    block. \n Here is the user question: {question} Here is the world knowledge: {world}"""

    ## Generation
    if "error" in state_dict:
        print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

        error = state_dict["error"]
        code_solution = state_dict["generation"]

        # Udpate prompt
        template_1 = """You are a coding assistant with expertise in PyCram, a python Roboter Language. \n Here is 
        the most important part of the PyCram regarding your coming task documentation: \n ------- \n {context} \n 
        ------- \n \n Be aware the documentation are only examples, use world knowledge for specific world 
        information. \n Structure the final answer with a description of the code solution. \n Then list the imports. 
        And finally list the functioning code block. \n Here is the user question: {question} Here is the world 
        knowledge: {world}"""
        addendum = """\n --- --- --- \n You previously tried to solve this problem. \n Here is your solution: \n --- 
        --- --- \n {generation}  \n --- --- --- \n  Here is the resulting error from code execution:  \n --- --- --- 
        \n {error}\n --- \n Additional information regarding the error: \n {information_error}  \n --- --- --- \n 
        Please re-try to answer this. Structure your answer with a description of the code solution. \n Then list the 
        imports. And finally list the functioning code block. Structure your answer with a description of the code 
        solution. \n Then list the imports. And finally list the functioning code block. \n Here is the user 
        question: \n {question}"""
        template = template_1 + addendum

        # Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "context",
                "question",
                "world",
                "generation",
                "error",
                "information_error",
            ],
        )

        # Chain
        chain = (
            {
                "context": itemgetter("question") | re_chain,
                "question": itemgetter("question"),
                "world": itemgetter("world"),
                "generation": itemgetter("generation"),
                "error": itemgetter("error"),
                "information_error": itemgetter("error") | re_chain_error,
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )

        code_solution = chain.invoke(
            {
                "question": question,
                "world": world,
                "generation": str(code_solution[0]),
                "error": error,
            }
        )

    else:

        print("---GENERATE SOLUTION---")
        # code_rewoo = stream_rewoo(question, world)
        code_rewoo = test_code
        # Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "code_rewoo", "question", "world"],
        )

        # Chain
        chain = (
            {
                "context": itemgetter("question") | re_chain,
                "code_rewoo": itemgetter("code_rewoo"),
                "question": itemgetter("question"),
                "world": itemgetter("world"),
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )

        code_solution = chain.invoke(
            {"code_rewoo": code_rewoo, "question": question, "world": world}
        )
        print(code_solution)

    iter = iter + 1
    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "world": world,
            "iterations": iter,
        }
    }


def check_code_imports(state: GraphState):
    """
    Check imports

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    ## State
    print("---CHECKING CODE IMPORTS---")
    state_dict = state["keys"]
    question = state_dict["question"]
    world = state_dict["world"]
    code_solution = state_dict["generation"]
    imports = code_solution[0].imports
    iter = state_dict["iterations"]

    try:
        # Attempt to execute the imports
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error
    else:
        print("---CODE IMPORT CHECK: SUCCESS---")
        # No errors occurred
        error = "None"

    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "world": world,
            "error": error,
            "iterations": iter,
        }
    }


def check_code_execution(state: GraphState):
    """
    Check code block execution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    ## State
    print("---CHECKING CODE EXECUTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    world = state_dict["world"]
    prefix = code_solution[0].prefix
    imports = code_solution[0].imports
    code = code_solution[0].code
    code_block = imports + "\n" + code
    iter = state_dict["iterations"]

    try:
        # Attempt to execute the code block
        exec(code_block)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error
    else:
        print("---CODE BLOCK CHECK: SUCCESS---")
        # No errors occurred
        error = "None"

    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "error": error,
            "prefix": prefix,
            "imports": imports,
            "iterations": iter,
            "code": code,
            "world": world,
        }
    }


### Edges


def decide_to_check_code_exec(state: GraphState):
    """
    Determines whether to test code execution, or re-try answer generation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]

    if error == "None":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "check_code_execution"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish (re-try code 3 times.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]
    iter = state_dict["iterations"]

    if error == "None" or iter == 3:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "end"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code_imports", check_code_imports)  # check imports
workflow.add_node("check_code_execution", check_code_execution)  # check execution

# Build graph
workflow.set_entry_point("generate")
workflow.add_edge("generate", "check_code_imports")
workflow.add_conditional_edges(
    "check_code_imports",
    decide_to_check_code_exec,
    {
        "check_code_execution": "check_code_execution",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "check_code_execution",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

# Compile
app = workflow.compile()

config = {"recursion_limit": 50}


def model(input: dict):
    return app.invoke({"keys": {**input, "iterations": 0}}, config=config)


world_test = """[robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf", pose=Pose([1, 2, 0])), apartment = Object(
"apartment", ObjectType.ENVIRONMENT, "apartment.urdf"), milk = Object("milk", ObjectType.MILK, "milk.stl", 
pose=Pose([2.5, 2, 1.02]), color=[1, 0, 0, 1]), cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, 
"breakfast_cereal.stl", pose=Pose([2.5, 2.3, 1.05]), color=[0, 1, 0, 1]), spoon = Object("spoon", ObjectType.SPOON, 
"spoon.stl", pose=Pose([2.4, 2.2, 0.85]), color=[0, 0, 1, 1]), bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", 
pose=Pose([2.5, 2.2, 1.02]), color=[1, 1, 0, 1]), apartment.attach(spoon, 'cabinet10_drawer_top')]"""
result = model(
    {
        "question": "Can you set the table for breakfast? I want to eat a bowl of cereals with milk.",
        "world": world_test,
    }
)

print(result)
