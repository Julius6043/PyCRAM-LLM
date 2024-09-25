from code_scraper import (
    scrape_python_files_to_text,
    scrape_docu_files_to_text,
    scrape_udrf_files_to_text,
)
from src.vector_store_SB import load_in_vector_store
from dotenv import load_dotenv
import os


# Load environment variables for secure access to configuration settings
load_dotenv()

example1 = """User Instruction: Can you set the table for breakfast? I want to eat a bowl of cereals with milk.
---


World Knowledge:
<knowledge>
[robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf", pose=Pose([1, 2, 0])), apartment = Object("apartment", ObjectType.ENVIRONMENT, "apartment.urdf"), milk = Object("milk", ObjectType.MILK, "milk.stl", pose=Pose([2.5, 2, 1.02]), color=[1, 0, 0, 1]), cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, "breakfast_cereal.stl", pose=Pose([2.5, 2.3, 1.05]), color=[0, 1, 0, 1]), spoon = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), color=[0, 0, 1, 1]), bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", pose=Pose([2.5, 2.2, 1.02]), color=[1, 1, 0, 1]), apartment.attach(spoon, 'cabinet10_drawer_top')]
</knowledge>

PyCramPlanCode:
<code>
from pycram.worlds.bullet_world import BulletWorld
from pycram.designators.action_designator import *
from pycram.designators.location_designator import *
from pycram.designators.object_designator import *
from pycram.datastructures.enums import ObjectType, WorldMode
from pycram.datastructures.pose import Pose
from pycram.process_module import simulated_robot, with_simulated_robot
from pycram.object_descriptors.urdf import ObjectDescription
from pycram.world_concepts.world_object import Object
from pycram.datastructures.dataclasses import Color

world = BulletWorld(WorldMode.GUI)
robot = Object("pr2", ObjectType.ROBOT, f"pr2.urdf", pose=Pose([1, 2, 0]))
apartment = Object("apartment", ObjectType.ENVIRONMENT, f"apartment.urdf")

milk = Object("milk", ObjectType.MILK, "milk.stl", pose=Pose([2.5, 2, 1.02]), color=Color(1, 0, 0, 1))
cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, "breakfast_cereal.stl", pose=Pose([2.5, 2.3, 1.05]), color=Color(0, 1, 0, 1))
spoon = Object("spoon", ObjectType.SPOON, "spoon.stl", pose=Pose([2.4, 2.2, 0.85]), color=Color(0, 0, 1, 1))
bowl = Object("bowl", ObjectType.BOWL, "bowl.stl", pose=Pose([2.5, 2.2, 1.02]), color=Color(1, 1, 0, 1))
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

    TransportAction(milk_desig, [Arms.LEFT], [Pose([4.8, 3.55, 0.8])]).resolve().perform()

    cereal_desig = move_and_detect(ObjectType.BREAKFAST_CEREAL)

    TransportAction(cereal_desig, [Arms.RIGHT], [Pose([5.2, 3.4, 0.8], [0, 0, 1, 1])]).resolve().perform()

    bowl_desig = move_and_detect(ObjectType.BOWL)

    TransportAction(bowl_desig, [Arms.LEFT], [Pose([5, 3.3, 0.8], [0, 0, 1, 1])]).resolve().perform()

    # Finding and navigating to the drawer holding the spoon
    handle_desig = ObjectPart(names=["handle_cab10_t"], part_of=apartment_desig.resolve())
    drawer_open_loc = AccessingLocation(handle_desig=handle_desig.resolve(), robot_desig=robot_desig.resolve()).resolve()

    NavigateAction([drawer_open_loc.pose]).resolve().perform()

    OpenAction(object_designator_description=handle_desig, arms=[drawer_open_loc.arms[0]]).resolve().perform()
    spoon.detach(apartment)

    # Detect and pickup the spoon
    LookAtAction([apartment.get_link_pose("handle_cab10_t")]).resolve().perform()

    spoon_desig = DetectAction(BelieveObject(types=[ObjectType.SPOON])).resolve().perform()

    pickup_arm = Arms.LEFT if drawer_open_loc.arms[0] == Arms.RIGHT else Arms.RIGHT
    PickUpAction(spoon_desig, [pickup_arm], [Grasp.TOP]).resolve().perform()

    ParkArmsAction([Arms.LEFT if pickup_arm == Arms.LEFT else Arms.RIGHT]).resolve().perform()

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
</code>

This is the corresponding plan:

Plan 1: Get the URDF file of the apartment. #E1 = URDF[apartment.urdf]
Plan 2: Get the URDF file of the pr2 robot. #E2 = URDF[pr2.urdf]
Plan 3: Initialize the BulletWorld and load the robot and environment. #E3 = Retrieve[How do I initialize a BulletWorld and load objects?]
Plan 4: Create the objects milk, cereal, spoon, and bowl. #E4 = Retrieve[How do I create objects in PyCram?]
Plan 5: Create designators for the robot, the object and environment. #E5 = Retrieve[How do I create object designators in PyCram?]
Plan 6: Park both arms of the robot and raise the torso. #E6 = Retrieve[How do I park the arms and move the torso in PyCram?]
Plan 7: Detect the milk and transport it. #E7 = Retrieve[How do I detect and transport objects in PyCram?]
Plan 8: Detect the cereal and transport it. #E8 = Retrieve[How do I detect and transport objects in PyCram?]
Plan 9: Detect the bowl and transport it. #E9 = Retrieve[How do I detect and transport objects in PyCram?]
Plan 10: Open the drawer and detect the spoon. #E10 = Retrieve[How do I open drawers and detect objects in PyCram?]
Plan 11: Pick up the spoon and close the drawer. #E11 = Retrieve[How do I pick up objects and close drawers in PyCram?]
Plan 12: Place the spoon and park the arms. #E12 = Retrieve[How do I place objects and park the arms in PyCram?]
Plan 13: Close the BulletWorld. #E13 = Retrieve[How do I close the BulletWorld in PyCram?]
"""

example2 = """User Instruction: Can you place the cereal on the kitchen island?
---


World Knowledge:
<knowledge>
[kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95]))]
</knowledge>

PyCramPlanCode:
<code>
from pycram.worlds.bullet_world import BulletWorld
from pycram.world_concepts.world_object import Object
from pycram.process_module import simulated_robot
from pycram.designators.motion_designator import *
from pycram.designators.location_designator import *
from pycram.designators.action_designator import *
from pycram.designators.object_designator import *
from pycram.datastructures.enums import ObjectType, Arms, Grasp, WorldMode

world = BulletWorld(WorldMode.GUI)
kitchen = Object("kitchen", ObjectType.ENVIRONMENT, "kitchen.urdf")
robot = Object("pr2", ObjectType.ROBOT, "pr2.urdf")
cereal = Object("cereal", ObjectType.BREAKFAST_CEREAL, "breakfast_cereal.stl", pose=Pose([1.4, 1, 0.95]))

cereal_desig = ObjectDesignatorDescription(names=["cereal"])
kitchen_desig = ObjectDesignatorDescription(names=["kitchen"])
robot_desig = ObjectDesignatorDescription(names=["pr2"]).resolve()

with simulated_robot:
    ParkArmsAction([Arms.BOTH]).resolve().perform()

    MoveTorsoAction([0.3]).resolve().perform()

    pickup_pose = CostmapLocation(target=cereal_desig.resolve(), reachable_for=robot_desig).resolve()
    pickup_arm = pickup_pose.reachable_arms[0]

    NavigateAction(target_locations=[pickup_pose.pose]).resolve().perform()

    PickUpAction(object_designator_description=cereal_desig, arms=[pickup_arm], grasps=[Grasp.FRONT]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

    place_island = SemanticCostmapLocation("kitchen_island_surface", kitchen_desig.resolve(), cereal_desig.resolve()).resolve()

    place_stand = CostmapLocation(place_island.pose, reachable_for=robot_desig, reachable_arm=pickup_arm).resolve()

    NavigateAction(target_locations=[place_stand.pose]).resolve().perform()

    PlaceAction(cereal_desig, target_locations=[place_island.pose], arms=[pickup_arm]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

world.exit()
</code>

This is the corresponding plan:

Plan 1: Get the URDF file of the kitchen. #E1 = URDF[kitchen.urdf]
Plan 2: Get the URDF file of the pr2 robot. #E2 = URDF[pr2.urdf]
Plan 3: Create a object designator for the cereal, kitchen and the roboter object. #E3 = LLM[How to create an object designator in PyCram?]
Plan 4: Retrieve the method for navigating the robot to the kitchen. #E4 = Retrieve[How to navigate the robot to a specific position in PyCram?]
Plan 5:  Retrieve the method to locate and pick up an object. #E5 = Retrieve[How to locate an object and pick it up?]
Plan 6: Retrieve the procedure for finding a suitable position to place the cereal on the kitchen island. #E6 = Retrieve[How to find a suitable position for placing an object in PyCram?]
Plan 7: Retrieve the method for placing the cereal on the kitchen island. #E7 = Retrieve[How to place an object at a specific position in PyCram?]
Plan 8: Close the BulletWorld. #E8 = Retrieve[How do I close the BulletWorld in PyCram?]
"""


def load_data(pycram_path):
    scrape_docu_files_to_text(pycram_path + "/examples", "scraped_docu.txt")
    scrape_python_files_to_text(pycram_path + "/src/pycram", "scraped_code.txt")
    scrape_udrf_files_to_text(pycram_path + "/resources", "scraped_urdf.txt")


def setup_vector_store(pycram_path, just_upload=True):
    if just_upload:
        load_data(pycram_path)
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    # load_in_vector_store(folder_path + "/scraped_docu")
    load_in_vector_store(folder_path + "/scraped_code.txt", 1)
    load_in_vector_store(folder_path + "/scraped_urdf.txt", 4)
    load_in_vector_store([example1], 3)


#setup_vector_store("d:\Git\Repository\pycram")
load_in_vector_store([example1, example2], 3)
