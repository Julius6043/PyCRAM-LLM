import asyncio
from typing import Dict, TypedDict
from langgraph.graph import END, StateGraph

from vector_store_SB import get_retriever, load_in_vector_store
from ReWOO_parallel import stream_rewoo
from ReWOO_codeCheck_parallel import stream_rewoo_check
from dotenv import load_dotenv
from code_exec import execute_code_in_process
from prompts import preprocessing_chain

load_dotenv()


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

    # State
    state_dict = state["keys"]
    question = state_dict["question"]
    world = state_dict["world"]
    iter = state_dict["iterations"]
    max_iter = state_dict["max_iter"]
    if "filled_plan" in state_dict:
        filled_plan = state_dict["filled_plan"]
    else:
        filled_plan = ""
    if "plan" in state_dict:
        plan = state_dict["plan"]
    else:
        plan = ""
    if "plans_code_check" in state_dict:
        plans_code_check = state_dict["plans_code_check"]
    else:
        plans_code_check = []
    iter = iter + 1
    # Generation
    if "error" in state_dict:
        print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

        error = state_dict["error"]
        code_solution = state_dict["generation"]
        code_solution, plan_check, filled_plan_check = asyncio.run(
            stream_rewoo_check(question, world, str(code_solution), error)
        )
        plans_code_check.append((plan_check, filled_plan_check, code_solution))
        print(f"----CodePlan Versuch {iter}----")
        print(code_solution)

    else:
        # Question und worldknowledge können hier nochmal vorverarbeitet werden
        should_prethink = state_dict["should_prethink"]
        if should_prethink:
            print("---Prethinking---")
            pre_thinking = preprocessing_chain.invoke(
                {"prompt": question, "world": world}
            )
            question = f"User Instruction: {question}\n\nThe following is a pre thinking process for the user instruction. It is not necessarily right especially the Positions. But use it as a foundation for your task:\n<thinking>{pre_thinking}</thinking>\n\n"
            print(question)
        print("---GENERATE SOLUTION---")
        code_solution, plan, filled_plan = asyncio.run(stream_rewoo(question, world))
        print("----CodePlan Versuch 1----")
        print(code_solution)
    if iter == 1:
        first_code_solution = code_solution
    else:
        first_code_solution = state_dict["first_solution"]
    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "world": world,
            "plan": plan,
            "filled_plan": filled_plan,
            "iterations": iter,
            "max_iter": max_iter,
            "plans_code_check": plans_code_check,
            "first_solution": first_code_solution,
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

    # State
    print("---CHECKING CODE IMPORTS---")
    state_dict = state["keys"]
    question = state_dict["question"]
    world = state_dict["world"]
    code_solution = state_dict["generation"]
    imports = code_solution.imports
    plan = state_dict["plan"]
    filled_plan = state_dict["filled_plan"]
    plans_code_check = state_dict["plans_code_check"]
    iter = state_dict["iterations"]
    max_iter = state_dict["max_iter"]
    first_solution = state_dict["first_solution"]
    # Code Execution
    exec_import_result = execute_code_in_process(imports)
    print(exec_import_result)
    if exec_import_result == "SUCCESS":
        print("---CODE IMPORT CHECK: SUCCESS---")
        # No errors occurred
        error = "None"
    else:
        print("---CODE IMPORT CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = exec_import_result
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error

    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "world": world,
            "error": error,
            "plan": plan,
            "filled_plan": filled_plan,
            "iterations": iter,
            "max_iter": max_iter,
            "plans_code_check": plans_code_check,
            "first_solution": first_solution,
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
    imports = code_solution.imports
    code = code_solution.code
    code_block = imports + "\n" + code
    plan = state_dict["plan"]
    filled_plan = state_dict["filled_plan"]
    plans_code_check = state_dict["plans_code_check"]
    first_solution = state_dict["first_solution"]
    iter = state_dict["iterations"]
    max_iter = state_dict["max_iter"]
    full_result = (
        ""
        + question
        + "\nWorld Knowledge:\n"
        + "<world_knowledge>\n"
        + world
        + "\n</world_knowledge>\n\n"
        + "PyCramPlanCode:\n"
        + "<code>\n"
        + code_block
        + "\n</code>\n\n"
        + "This is the corresponding plan:\n"
        + plan
    )

    exec_result = execute_code_in_process(code_block)
    print(exec_result)
    if exec_result == "SUCCESS":
        print("---CODE BLOCK CHECK: SUCCESS---")
        # No errors occurred
        error = "None"
        # Load the result as a new example in the database
        load_in_vector_store([full_result], 3)
        success = True
    else:
        print("---CODE BLOCK CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = exec_result
        success = False
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error

    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "error": error,
            "imports": imports,
            "iterations": iter,
            "plan": plan,
            "filled_plan": filled_plan,
            "plans_code_check": plans_code_check,
            "code": code,
            "world": world,
            "max_iter": max_iter,
            "full_result": full_result,
            "success": success,
            "first_solution": first_solution,
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
    max_iter = state_dict["max_iter"]

    if error == "None" or iter == max_iter:
        print("---DECISION: TEST CODE EXECUTION---")
        return "end"
    else:
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


def generate_plan_parallel(question, world, max_iterations=3, should_prethink=True):
    result_dic = model(
        {
            "question": question,
            "world": world,
            "max_iter": max_iterations,
            "should_prethink": should_prethink,
        }
    )
    result_code = result_dic["keys"]["generation"]
    result_code_plan = result_code.imports + "\n\n" + result_code.code
    full_result = result_dic["keys"]["full_result"]
    filled_plan = result_dic["keys"]["filled_plan"]
    final_iter = result_dic["keys"]["iterations"]
    plans_code_check = result_dic["keys"]["plans_code_check"]
    success = result_dic["keys"]["success"]
    first_solution = result_dic["keys"]["first_solution"]
    return (
        result_code_plan,
        full_result,
        filled_plan,
        final_iter,
        success,
        plans_code_check,
        first_solution,
    )


task_test = """Kannst du das Müsli aufnehmen und 3 Schritte rechts wieder abstellen?"""
world_test = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
"""

task_test2 = """Can you set the table for breakfast? I want to eat a bowl of cereals with milk."""
world_test2 = """[robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf", pose=Pose([1, 2, 0])), apartment = Object(
"apartment", ObjectType.ENVIRONMENT, "apartment.urdf"), milk = Object("milk", ObjectType.MILK, "milk.stl", 
pose=Pose([2.5, 2, 1.02]), color=[1, 0, 0, 1]), cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, 
"breakfast_cereal.stl", pose=Pose([2.5, 2.3, 1.05]), color=[0, 1, 0, 1]), spoon = Object("spoon", ObjectType.SPOON, 
"spoon.stl", pose=Pose([2.4, 2.2, 0.85]), color=[0, 0, 1, 1]), bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", 
pose=Pose([2.5, 2.2, 1.02]), color=[1, 1, 0, 1]), apartment.attach(spoon, 'cabinet10_drawer_top')]"""
# result = generate_plan(task_test, world_test)

# print(result)
