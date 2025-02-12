from openai import fine_tuning

from langgraph_code_assistant import generate_plan
from langgraph_code_assistant_parallel import generate_plan_parallel
from vector_store_SB import get_retriever
from helper_func import extract_urdf_files
from prompts import chain_docs_docu, preprocessing_chain, chain_docs_code, urdf_tool
import sys

## test case documentation
task_test1 = """Place the cereal box on the kitchen island."""
world_test1 = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'),
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL,'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95]))]
"""

## test case single easy task
task_test2 = """Move to position (-2.5,1,0)."""
world_test2 = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')]"""


## test case simple
task_test3 = """Pick up the bowl from the table."""
world_test3 = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')
bowl = Object('bowl', ObjectType.BOWL, 'bowl.stl', pose=Pose([1.4, 1, 0.89]), 
color=[1, 1, 0, 1])]"""


## test case middle
task_test4 = """Place the cereal and a bowl side by side on the kitchen island"""
world_test4 = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')
bowl = Object('bowl', ObjectType.BOWL, 'bowl.stl', pose=Pose([1.4, 0.50, 0.89]), 
color=[1, 1, 0, 1]),
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', 
pose=Pose([1.4, 1, 0.95]))]"""


## test case complex
task_test5 = """Can you set the table for breakfast? I want to eat a bowl of cereals with milk."""
world_test5 = """[robot = Object("pr2", ObjectType.ROBOT, 'pr2.urdf', pose=Pose([1, 2, 0])), 
apartment = Object('apartment', ObjectType.ENVIRONMENT, 'apartment.urdf'), 
milk = Object('milk', ObjectType.MILK, 'milk.stl', pose=Pose([2.5, 2, 1.02]), 
color=[1, 0, 0, 1]), 
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', 
pose=Pose([2.5, 2.3, 1.05]), color=[0, 1, 0, 1]), 
spoon = Object('spoon', ObjectType.SPOON, 'spoon.stl', pose=Pose([2.4, 2.2, 0.85]), 
color=[0, 0, 1, 1]), 
bowl = Object('bowl', ObjectType.BOWL, 'bowl.stl', pose=Pose([2.5, 2.2, 1.02]), 
color=[1, 1, 0, 1]), 
apartment.attach(spoon, 'cabinet10_drawer_top')]"""


## test case room perception
task_test6 = """Place the cereal box directly next to the refrigerator."""
world_test6 = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), 
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', 
pose=Pose([1.4, 1, 0.95]))]"""


## test case unknown object
task_test7 = """Pick up the 'thing' from the kitchen counter."""
world_test7 = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), 
spoon = Object('spoon', ObjectType.SPOON, 'spoon.stl', pose=Pose([1.4, 1, 0.87]), 
color=[0, 0, 1, 1])]"""


## test case negative
task_test8 = """Pick up a spoon, but not one that's already on a table."""
world_test8 = """[robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf', pose=Pose([1, 2, 0])), 
apartment = Object('apartment', ObjectType.ENVIRONMENT, 'apartment.urdf'),
milk = Object('milk', ObjectType.MILK, 'milk.stl', pose=Pose([2.5, 2, 1.02]), color=[1, 0, 0, 1]), 
spoon1 = Object('spoon1', ObjectType.SPOON, 'spoon.stl', pose=Pose([2.4, 2.2, 0.85]), color=[0, 0, 1, 1]),
spoon2 = Object('spoon2', ObjectType.SPOON, 'spoon.stl', pose=Pose([2.5, 2.3, 1.00]), color=[0, 0, 1, 1]), 
spoon3 = Object('spoon3', ObjectType.SPOON, 'spoon.stl', pose=Pose([0.4, 3.1, 0.96]), color=[0, 0, 1, 1]), 
bowl = Object('bowl', ObjectType.BOWL, 'bowl.stl', pose=Pose([2.5, 2.2, 1.00]), color=[1, 1, 0, 1]),
apartment.attach(spoon1, 'cabinet10_drawer_top')]"""


# retriever test
test_code_retriever = get_retriever(1, 2)
test_docu_retriever = get_retriever(2, 2)


def test_tool(tool_num, test_num, prompt=""):
    task = f"task_test{test_num}"
    world = f"world_test{test_num}"

    if task in globals() and world in globals():
        task = globals()[task]
        world = globals()[world]
    else:
        return "Der Test existiert nicht"
    if tool_num == 1:
        result = chain_docs_docu.invoke(
            {"task": prompt, "instruction": task, "world": world}
        )
    elif tool_num == 2:
        result = chain_docs_code.invoke(
            {"task": prompt, "instruction": task, "world": world}
        )
    elif tool_num == 3:
        result = preprocessing_chain.invoke({"prompt": task, "world": world})
    elif tool_num == 4:
        urdf_retriever = get_retriever(4, 1, {"source": prompt})
        files = urdf_retriever.invoke(prompt)
        if len(files) >= 1:
            file = files[0].page_content
            result = urdf_tool.invoke({"prompt": task, "world": world, "urdf": file})
        else:
            result = f"The URDF {prompt} is not in the database."
    else:
        raise Exception("Invalid tool number")
    return result


def make_test(test_num, run_sufix=None):
    task = f"task_test{test_num}"
    world = f"world_test{test_num}"

    if run_sufix is None:
        run_num = 1

    if task in globals() and world in globals():
        task = globals()[task]
        world = globals()[world]
    else:
        return "Der Test existiert nicht"

    (
        result,
        full_result_plan,
        first_filled_plan,
        final_iter,
        success,
        plans_code_check,
        first_solution,
    ) = generate_plan_parallel(task, world)
    first_plan = full_result_plan.split("This is the corresponding plan:")[-1]
    first_solution = first_solution.imports + "\n" + first_solution.code
    if final_iter > 1:
        plans_code_check_string = f"All Iterations with Plan and Solution:\nRun 1:\nPlan: \n{first_plan}\n-\n\nCode Solution:\n{first_solution}\n-\n\nFilled Plan:\n{first_filled_plan}\n\n---\n"
        i = 2
        for plan in plans_code_check:
            plan_string, filled_plan_string, code_solution = plan
            code_solution = code_solution.imports + "\n" + code_solution.code
            plans_code_check_string += f"Code Check Run {i}:\nPlan: \n{plan_string}\n-\n\nCode Solution:\n{code_solution}\n-\n\nFilled Plan:\n{filled_plan_string}\n\n---Next Run---\n"
            i += 1
    else:
        plans_code_check_string = ""
    if success:
        success = "Yes"
    else:
        success = "No"
    with open(f"../test_files/test{test_num}v{run_sufix}.txt", "w") as file:
        test_string = f"Iterations:\n{final_iter}\n\nSuccess: {success}\n----\n\nPlan:\n{full_result_plan}\n\n----\nResult:\n{result}\n\n----\nFilled Plan:\n{first_filled_plan}\n\n------\n\n{plans_code_check_string}"
        file.write(test_string)
    return result


### tests


"""pre_thinking = preprocessing_chain.invoke({"prompt": task_test4, "world": world_test4})
print(pre_thinking)"""
# print(result_retriever_code)

print(make_test(8, "finetuning"))
"""code_retrieve = test_code_retriever.invoke("How is CostmapLocation defined?")
docu_retieve = test_docu_retriever.invoke(
    "What are Action Designators and how do i use them?"
)"""
"""for text in code_retrieve:
    print(text.page_content)
    print("\n\n-----\n\n")
"""
"""print("\n------------------------Docu test:\n")
for text in docu_retieve:
    print(text.page_content)
    print("\n\n-----\n\n")"""
# Doku Tool
# print(test_tool(1, 4, "What are Action Designators and how are they used?"))
# code Tool
# print(test_tool(2, 4, "How is CostmapLocation defined?"))
# prethink
# print(test_tool(3, 4))
# urdf Tool
# print(test_tool(4, 4, "kitchen.urdf"))
