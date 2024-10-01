from langgraph_code_assistant import generate_plan
from langgraph_code_assistant_parallel import generate_plan_parallel
from vector_store_SB import get_retriever
from helper_func import extract_urdf_files
from prompts import chain_docs_docu, preprocessing_chain, chain_docs_code
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


def test_tool(tool_num, test_num, prompt):
    task = f"task_test{test_num}"
    world = f"world_test{test_num}"

    if task in globals() and world in globals():
        task = globals()[task]
        world = globals()[world]
    else:
        return "Der Test existiert nicht"
    if tool_num == 1:
        result = chain_docs_docu.invoke({"task": prompt, "instruction": task, "world": world})
    elif tool_num == 2:
        result = chain_docs_code.invoke({"task": prompt, "instruction": task, "world": world})
    elif tool_num == 3:
        result = preprocessing_chain.invoke({"prompt": task, "world": world})
    else:
        raise Exception("Invalid tool number")
    return result


def make_test(test_num, run_num=1):
    task = f"task_test{test_num}"
    world = f"world_test{test_num}"

    if task in globals() and world in globals():
        task = globals()[task]
        world = globals()[world]
    else:
        return "Der Test existiert nicht"

    result, first_plan, first_filled_plan, final_iter, success, plans_code_check, first_solution = generate_plan_parallel(task, world)
    if final_iter > 1:
        plans_code_check_string = f"All Iterations with Plan and Solution:\nRun 1:\nPlan: \n{first_plan}\n\n Filled Plan:\n{first_filled_plan}\n\nCode Solution:\n{first_solution}\n\n---\n"
        i = 2
        for plan in plans_code_check:
            plan_string, filled_plan_string, code_solution = plan
            plans_code_check_string += f"Code Check Run {i}:\nPlan: \n{plan_string}\n\n Filled Plan:\n{filled_plan_string}\n\nCode Solution:\n{code_solution}\n\n---\n"
            i += 1
    else:
        plans_code_check_string = ""
    if success:
        success = "Yes"
    else:
        success = "No"
    with open(f"../test_files/test{test_num}v{run_num}.txt", "w") as file:
        test_string = f"Iterations:\n{final_iter}\n\nSuccess: {success}\n---\n\nPlan:\n{first_plan}\n\n----\nResult:\n{result}\n\n----\nFilled Plan:\n{first_filled_plan}\n\n------\n\n{plans_code_check_string}"
        file.write(test_string)
    return result


### tests


"""pre_thinking = preprocessing_chain.invoke({"prompt": task_test4, "world": world_test4})
print(pre_thinking)"""
# print(result_retriever_code)

print(make_test(8, 3))
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
# print(test_tool(1, "What are Action Designators and how do i use them?"))
# code Tool
# print(test_tool(2, "How is CostmapLocation defined?"))
