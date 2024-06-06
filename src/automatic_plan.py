# Import necessary libraries and modules
from vector_store_SB import get_retriever, load_in_vector_store
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from dotenv import load_dotenv

# Load environment variables, if any
load_dotenv()


# Define a function to format documents for better readability
def format_docs(docs):
    # Join documents using a specified delimiter for separation
    return "\n\n<next example>\n".join([d.page_content for d in docs])


# Initialize the LLM with a specific model and temperature setting
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Set up the retriever with specific parameters and chain it with the document formatter
retriever = get_retriever(3, 1)  # Assuming '3, 3' are retriever-specific parameters
re_chain = retriever | format_docs

# Configure a chat prompt template for the retriever chain
prompt_retriever_chain = ChatPromptTemplate.from_template(
    """You are a renowned AI engineer and programmer. You receive PyCramPlanCode, world knowledge and a task. A task could be a question or a command. Use these information to generate a plan which can be used by a LLM to reverse enginere the PyCramPlanCode. 
PyCramPlanCode is a code that enables a robot to perform the task. For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You can store the evidence in a variable #E, which can be called upon by other tools later. (Plan, #E1, Plan, #E2, Plan, ...).
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


Here are some examples of PyCramPlanCode with its corresponding building plan (use them just as examples to learn the code format and the plan structure):
{examples}

--- end of examples ---

Begin!
Describe your plans with rich details. Each plan should follow only one #E. You do not need to consider how PyCram is installed and set up in the plans, as this is already given. Include the PyCramCode, the WorldKnowledge and the task in your output.

(Include the following also in your Output and just answer with the following texts and the Plan structure)
PyCramPlanCode: {pycram_plan_code}
World Knowledge: {world_knowledge}
Task: {task}
Plan 1:
"""
)

# Chain components together for document processing, prompt generation, and LLM execution
chain_docs = (
    {
        "examples": itemgetter("task") | re_chain,
        "world_knowledge": itemgetter("world"),
        "pycram_plan_code": itemgetter("pycram_plan_code"),
        "task": itemgetter("task"),
    }
    | prompt_retriever_chain
    | llm
    | StrOutputParser()
)


# Main function to generate a plan based on task, pycram plan code, and world knowledge
def generate_plan(
    task, pycram_plan_code, world_knowledge, load_vector_store=False, plan=""
):
    # Invoke the chain with the provided inputs
    if load_vector_store and plan != "":
        result = plan
    else:
        # Invoke the chain with the provided inputs and return the result
        result = chain_docs.invoke(
            {
                "task": task,
                "pycram_plan_code": pycram_plan_code,
                "world": world_knowledge,
            }
        )
    # Option to load the result into a vector store, if needed
    if load_vector_store:
        full_result = (
            "Task:"
            + task
            + "\nPyCramPlanCode:\n"
            + "<code>\n"
            + pycram_plan_code
            + "\n</code>\n"
            + "World Knowledge:\n"
            + "<world_knowledge>\n"
            + world_knowledge
            + "\n</world_knowledge>\n"
            + "\n This is the corresponding plan:\n"
            + result
        )
        load_in_vector_store([full_result], 3)
    # Return the generated plan or result
    return result


# Function to load a case plan into the vector store
def load_case(case_plan):
    # Load the provided case plan into the vector store with a specific parameter
    load_in_vector_store([case_plan], 3)


case_plan1 = """This is an example of a PyCramPlanCode:
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
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95]))

#Object Designators
cereal_desig = ObjectDesignatorDescription(names=['cereal'])
kitchen_desig = ObjectDesignatorDescription(names=['kitchen'])
robot_desig = ObjectDesignatorDescription(names=['pr2']).resolve()

#The 'with simulated_robot:'-Block
with simulated_robot:
    #AktionDesignators
    ParkArmsAction([Arms.BOTH]).resolve().perform()

    MoveTorsoAction([0.25]).resolve().perform()
    
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
World knowledge: [kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95]))]
Task: Can you place the cereal on the kitchen island?
Plan 1: Retrieve the information on how to create and use ObjectDesignatorDescription for the cereal and the kitchen island within PyCram. #E1 = Retrieve[How to create and use ObjectDesignatorDescription for specific objects in PyCram?]
Plan 2: Retrieve the procedure for finding the location of a specific object using costmaps. #E2 = Retrieve[How to use costmaps to locate objects?]
Plan 3: Retrieve the method for navigating a robot to a specific object using PyCram. #E3 = Retrieve[How to navigate a robot to a specific object's location using PyCram?]
Plan 4: Retrieve the procedure for picking up an object with a specific arm and grasp in PyCram. #E4 = Retrieve[How to pick up an object with a specific arm and grasp using PyCram?]
Plan 5: Retrieve the method for navigating to a semantic location, like the kitchen island, within PyCram. #E5 = Retrieve[How to navigate a robot to a semantic location like the kitchen island in PyCram?]
Plan 6: Retrieve the procedure for placing an object on a target location using PyCram. #E6 = Retrieve[How to place an object on a target location using PyCram?]"""
case_plan2 = """Plan 1: Retrieve"""
case_plan3 = """Plan 1: Retrieve"""


### test
task_test = (
    "Can you set the table for breakfast? I want to eat a bowl of cereals with milk."
)
pycram_plan_code_test = """
from pycram.designators.action_designator import *
from pycram.designators.location_designator import *
from pycram.designators.object_designator import *
from pycram.pose import Pose
from pycram.bullet_world import BulletWorld, Object
from pycram.process_module import simulated_robot, with_simulated_robot
from pycram.enums import ObjectType

world = BulletWorld()
robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf", pose=Pose([1, 2, 0]))
apartment = Object("apartment", ObjectType.ENVIRONMENT, "apartment.urdf")

milk = Object("milk", ObjectType.MILK, "milk.stl", pose=Pose([2.5, 2, 1.02]), color=[1, 0, 0, 1])
cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, "breakfast_cereal.stl", pose=Pose([2.5, 2.3, 1.05]), color=[0, 1, 0, 1])
spoon = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), color=[0, 0, 1, 1])
bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", pose=Pose([2.5, 2.2, 1.02]), color=[1, 1, 0, 1])
apartment.attach(spoon, 'cabinet10_drawer_top')

pick_pose = Pose([2.7, 2.15, 1])

robot_desig = BelieveObject(names=["pr2"])
apartment_desig = BelieveObject(names=["apartment"])

@with_simulated_robot
def move_and_detect(obj_type):
    NavigateAction(target_locations=[Pose([1.7, 2, 0])]).resolve().perform()
    LookAtAction(targets=[pick_pose]).resolve().perform()
    object_desig = DetectAction(BelieveObject(types=[obj_type])).resolve().perform()
    return object_desig


with simulated_robot:
    ParkArmsAction([Arms.BOTH]).resolve().perform()

    MoveTorsoAction([0.25]).resolve().perform()

    milk_desig = move_and_detect(ObjectType.MILK)

    TransportAction(milk_desig, ["left"], [Pose([4.8, 3.55, 0.8])]).resolve().perform()

    cereal_desig = move_and_detect(ObjectType.BREAKFAST_CEREAL)

    TransportAction(cereal_desig, ["right"], [Pose([5.2, 3.4, 0.8], [0, 0, 1, 1])]).resolve().perform()

    bowl_desig = move_and_detect(ObjectType.BOWL)

    TransportAction(bowl_desig, ["left"], [Pose([5, 3.3, 0.8], [0, 0, 1, 1])]).resolve().perform()

    # Finding and navigating to the drawer holding the spoon
    handle_desig = ObjectPart(names=["handle_cab10_t"], part_of=apartment_desig.resolve())
    drawer_open_loc = AccessingLocation(handle_desig=handle_desig.resolve(), robot_desig=robot_desig.resolve()).resolve()

    NavigateAction([drawer_open_loc.pose]).resolve().perform()

    OpenAction(object_designator_description=handle_desig, arms=[drawer_open_loc.arms[0]]).resolve().perform()
    spoon.detach(apartment)

    # Detect and pickup the spoon
    LookAtAction([apartment.get_link_pose("handle_cab10_t")]).resolve().perform()

    spoon_desig = DetectAction(BelieveObject(types=[ObjectType.SPOON])).resolve().perform()

    pickup_arm = "left" if drawer_open_loc.arms[0] == "right" else "right"
    PickUpAction(spoon_desig, [pickup_arm], ["top"]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

    close_loc = drawer_open_loc.pose
    close_loc.position.y += 0.1
    NavigateAction([close_loc]).resolve().perform()

    CloseAction(object_designator_description=handle_desig, arms=[drawer_open_loc.arms[0]]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

    MoveTorsoAction([0.15]).resolve().perform()

    # Find a pose to place the spoon, move and then place it
    spoon_target_pose = Pose([4.85, 3.3, 0.8], [0, 0, 1, 1])
    placing_loc = CostmapLocation(target=spoon_target_pose, reachable_for=robot_desig.resolve()).resolve()

    NavigateAction([placing_loc.pose]).resolve().perform()

    PlaceAction(spoon_desig, [spoon_target_pose], [pickup_arm]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()
world.exit()
"""
world_knowledge_test = """[robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf", pose=Pose([1, 2, 0])), apartment = Object("apartment", ObjectType.ENVIRONMENT, "apartment.urdf"), milk = Object("milk", ObjectType.MILK, "milk.stl", pose=Pose([2.5, 2, 1.02]), color=[1, 0, 0, 1]), cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, "breakfast_cereal.stl", pose=Pose([2.5, 2.3, 1.05]), color=[0, 1, 0, 1]), spoon = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), color=[0, 0, 1, 1]), bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", pose=Pose([2.5, 2.2, 1.02]), color=[1, 1, 0, 1]), apartment.attach(spoon, 'cabinet10_drawer_top')]"""
# result = generate_plan(task_test, pycram_plan_code_test, world_knowledge_test, True)
# print(result)
