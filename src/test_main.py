from langgraph_code_assistant import generate_plan
from langgraph_ReWOO import stream_rewoo
from ReWOO_codeCheck import stream_rewoo_check
from vector_store_SB import get_retriever
import sys

## test case documentation
task_test1 = """Place the cereal box on the kitchen island."""
world_test1 = """
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), 
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
"""

## test case single easy task
task_test2 = """Move to position (0,1,1)."""
world_test2 = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')]"""


## test case simple
task_test3 = """Pick up the bowl from the table."""
world_test3 = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')
bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", pose=Pose([1.4, 1, 0.95]), 
color=[1, 1, 0, 1])]"""


## test case middle
task_test4 = """Place the cereal and a bowl side by side on the kitchen island"""
world_test4 = """[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')
bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", pose=Pose([1.4, 1, 0.95]), 
color=[1, 1, 0, 1]),
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', 
pose=Pose([1.4, 1, 0.95]))]]"""


## test case complex
task_test5 = """Can you set the table for breakfast? I want to eat a bowl of cereals with milk."""
world_test5 = """[robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf", pose=Pose([1, 2, 0])), 
apartment = Object("apartment", ObjectType.ENVIRONMENT, "apartment.urdf"), 
milk = Object("milk", ObjectType.MILK, "milk.stl", pose=Pose([2.5, 2, 1.02]), 
color=[1, 0, 0, 1]), 
cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, "breakfast_cereal.stl", 
pose=Pose([2.5, 2.3, 1.05]), color=[0, 1, 0, 1]), 
spoon = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), 
color=[0, 0, 1, 1]), 
bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", pose=Pose([2.5, 2.2, 1.02]), 
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
spoon = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), 
color=[0, 0, 1, 1])]"""


## test case negative
task_test8 = """Pick up a spoon, but not one that's already on a table."""
world_test8 = """[robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf", pose=Pose([1, 2, 0])), 
apartment = Object("apartment", ObjectType.ENVIRONMENT, "apartment.urdf"), 
milk = Object("milk", ObjectType.MILK, "milk.stl", pose=Pose([2.5, 2, 1.02]), 
color=[1, 0, 0, 1]), 
cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, "breakfast_cereal.stl", 
pose=Pose([2.5, 2.3, 1.05]), color=[0, 1, 0, 1]), 
spoon = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), 
color=[0, 0, 1, 1]),
spoon_table1 = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), 
color=[0, 0, 1, 1]), 
spoon_table2 = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), 
color=[0, 0, 1, 1]), 
bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", pose=Pose([2.5, 2.2, 1.02]), 
color=[1, 1, 0, 1]), 
apartment.attach(spoon, 'cabinet10_drawer_top')]"""


# retriever test
test_code_retriever = get_retriever(1, 3)
test_docu_retriever = get_retriever(2, 3)


def make_test(test_num, run_num=1):
    task = f"task_test{test_num}"
    world = f"world_test{test_num}"

    if task in globals() and world in globals():
        task = globals()[task]
        world = globals()[world]
    else:
        return "Der Test existiert nicht"

    result, plan, filled_plan = generate_plan(task, world)
    with open(f"test{test_num}v{run_num}.txt", "w") as file:
        test_string = f"Plan:\n{plan}\n\n----\nFilled Plan:\n{filled_plan}\n\n----\nResult:\n{result}"
        file.write(test_string)
    return result
