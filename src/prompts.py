from langchain_core.prompts import ChatPromptTemplate
from vector_store_SB import get_retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from helper_func import (
    format_docs,
    format_code,
    llm,
    llm_GP,
    llm_mini,
    extract_urdf_files,
    llm_tools,
)

llm_planer = llm_GP
llm_mini_main = llm_tools
## rewoo planer ----------------------------------
rewoo_planer_prompt = """You are a renowned AI engineer and programmer. You receive world knowledge and a task. Your task is to develop a sequenz of plans to geather 
resources and break down the given task for a other LLM Agent to generate PyCramPlanCode. PyCramPlanCode is a plan instruction for a robot that should enable it to perform the provided high level task. For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You 
can store the evidence in a variable #E, which can be called upon by other tools later. (Plan, #E1, Plan, #E2, Plan, 
...). Don't use **...** to highlight anything.

The tools can be one of the following: 
(1) Retrieve[input]: A vector database retrieval system containing the documentation of PyCram. Use this tool when you need information about PyCram functionality. The input should be a specific search query as a detailed question. 
(2) Code[input]: A LLM Agent with a database retriever for the PyCRAM code. Returns a function from the code base and provides a tutorial for it. Provide a function as input.
(3) URDF[input]: A database retriver which returns the URDF file text. Use this tool when you need information about the URDF files used in the world. Provide the URDF file name as input.

PyCramPlanCode follow the following structure:
Imports
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot:'-Block (defines the Actions and moves of the Robot)
    Start with this two Code lines in this block:
        'ParkArmsAction([Arms.BOTH]).resolve().perform()
        MoveTorsoAction([0.25]).resolve().perform()'
    ActionDesignators (Concentrate on using Action Designators over MotionDesignators)
    SematicCostmapLocation
BulletWorld Close


Here are some examples of PyCramPlanCode with its corresponding building plan (use them just as examples to learn the 
code format and the plan structure): 
{examples}

--- end of examples ---

Below you find all the Infos for your current task.
Describe your plans with rich details. Each plan should follow only one #E and it should be exactly in the given structure. Don't use any highlighting with markdown and co. You do not need to consider how 
PyCram is installed and set up in the plans, as this is already given.

World knowledge: {world}
Task: {task}"""

prompt_template = ChatPromptTemplate.from_template(rewoo_planer_prompt)
rewoo_planner = prompt_template | llm_planer

## Rewoo solve prompt ------------------------------------------------------------------------
# Solve function to generate PyCramPlanCode based on the plan and its steps...
rewoo_solve_prompt = """You are a professional programmer, specialized on writing PyCram-Roboter-Plans. To write the code, 
we have made a step-by-step Plan and retrieved corresponding evidence to each Plan. The evidences are examples and information to write PyCramPlanCode so 
use them with caution because long evidence might contain irrelevant information and only use the world knowledge 
for specific world information. Also be conscious about you hallucinating and therefore use evidence and example code 
as strong inspiration.

Example of PyCRAM Plan Code with the corresponding example plan (use this only as a example how the PyCRAM Code to a plan should look like):
<Code_example>
{code_example}
</Code_example>

Plan with evidence and examples:
<Plan>
{plan}
</Plan>

Now create the PyCramPlanCode for the task according to provided evidence above and the world knowledge. 
Respond with nothing other than the generated PyCramPlan python code.
PyCramPlanCode follow the following structure:
<Plan structure>
Imports #Import Designators with *
#clear separation between code block
BulletWorld Definition
Objects
Object Designators
The 'with simulated_robot:'-Block (defines the Actions and moves of the Robot)
    Start with this two Code lines in this block:
        'ParkArmsAction([Arms.BOTH]).resolve().perform()
        MoveTorsoAction([0.25]).resolve().perform()'
    ActionDesignators
    SemanticCostmapLocation
BulletWorld Close
</Plan structure>


Task: {task}
World knowledge: {world}
Code Response (Response structure: prefix("Description of the problem and approach"); imports()"Import statements of the code"); code(Code block not including import statements)):
"""

## rewoo codecheck planer ------------------------------------------------------------------
# Define a long and complex prompt template for generating plans...
rewoo_codecheck_planer_prompt = """You are a renowned AI engineer and programmer. You receive world knowledge, a task, an error-code and a 
code solution. The code solution was created by another LLM Agent like you to the given task and world knowledge. The 
code was already executed resulting in the provided error message. Your task is to develop a sequenz of plans to geather 
resources and correct the given PyCramPlanCode. PyCramPlanCode is a plan instruction for a robot that should enable the 
robot to perform the provided high level task. For each plan, indicate which external tool, along with the input for 
the tool, is used to gather evidence. You can store the evidence in a variable #E, which can be called upon by other 
tools later. (Plan, #E1, Plan, #E2, Plan, ...). Don't use **...** to highlight something.

The tools can be one of the following: 
(1) Retrieve[input]: A vector database retrieval system containing the documentation of PyCram. Use this tool when you need information about PyCram functionality. The input should be a specific search query as a detailed question. 
(2) Code[input]: A vector database retriever to search and look directly into the PyCram package code. As input give the exact Function and a little description.
(3) URDF[input]: A database retriver which returns the URDF file text. Use this tool when you need information about the URDF files used in the world. Provide the URDF file name as input.

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

prompt_template_codecheck_planer = ChatPromptTemplate.from_template(
    rewoo_codecheck_planer_prompt
)
codecheck_planner = prompt_template_codecheck_planer | llm_planer

## rewoo codecheck solve prompt-------------------------------------------------------------------------------------------------
# Solve function to generate PyCramPlanCode based on the plan and its steps...
codecheck_solve_prompt = """You are a professional programmer, specialized on correcting PycramRoboterPlanCode. To repair the 
code, we have made step-by-step Plan and retrieved corresponding evidence to each Plan. The evidence are examples 
and information to write PyCramCode so use them with caution because long evidence might contain irrelevant 
information and only use the world knowledge for specific world information. Also be conscious about you 
hallucinating and therefore use evidence and example code as strong inspiration.

Example of PyCRAM Plan Code with the corresponding example plan (use this only as a example how the PyCRAM Code to a plan should look like):
<Code_example>
{code_example}
</Code_example>

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
    ActionDesignators (Concentrate on using Action Designators over MotionDesignators)
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
Corrected Code Response (Response structure: prefix("Description of the problem and approach"); imports()"Import statements of the code"); code(Code block not including import statements)):
"""


## Docu Tool Chain -----------------------------------------------------------------------------------------------
# More complex template for tutorial writing, generating comprehensive documentation
prompt_docs = """**You are an experienced technical writer and coding educator specializing in creating comprehensive guides for implementing specific tasks and workflows using the PyCram framework. You are a Tool in an LLM Agent structure and you get a subtask for which you gather information.**

**Your task is to thoroughly explain how to accomplish the given task within PyCram, based on the provided context. Research and extract all relevant information, summarizing and organizing it in a way that enables another LLM agent to efficiently implement the workflow in code. Focus on using ActionDesignators over MotionDesignators. Do not provide information how to install and setup PyCRAM because it is already done.**

---

### Context:
Here is the high level task and world knowledge. Use it just as context and only work on you subtask:
<high_level_task_context>
{instruction}

World knowledge:
{world}
</high_level_task_context>
Here is the retrieved context for your subtask:
{context}

---


### Your Subtask:
{task}

---

### Instructions:

1. **Task Overview and Objectives**

   - **Define the Task:** Clearly articulate the task or workflow to be accomplished.
   - **Explain the Goal:** Describe the objective of the task and its significance within the PyCram framework.
   - **Prerequisites and Setup:** Detail any necessary setup steps or prerequisites required before starting the task.

2. **Detailed Workflow Explanation**

   - **Step-by-Step Guide:** Break down the process into logical, sequential steps.
   - **Key Concepts:** Explain important concepts and how they relate to each step.
   - **Relevant Functions:** Highlight relevant PyCram functions and explain their roles in the workflow.
   - **Integration:** Discuss how these steps and functions integrate within the PyCram framework.

3. **Code Examples and Implementation Guidance**

   - **Code Snippets:** Provide clear, complete code examples for each significant step.
   - **Explanation:** Accompany code with detailed explanations to ensure understanding.
   - **Adaptability:** Ensure code examples can be easily adapted to similar tasks or scenarios.

4. **Framework Integration and Concepts**

   - **Broader Context:** Explain how the task fits into the larger PyCram framework.
   - **Essential Components:** Discuss crucial components, tools, or modules within PyCram relevant to the task.
   - **Conceptual Understanding:** Enhance understanding of the framework's architecture as it relates to the task.

5. **Best Practices and Considerations**

   - **Implementation Tips:** Offer best practices for effectively implementing the task.
   - **Potential Challenges:** Identify common pitfalls or challenges that may arise.
   - **Solutions:** Provide recommendations or strategies to overcome these challenges.

---

### Important Notes:

- **Length:** Write as short as possible but ensure that everything important is included.
- **Completeness:** Ensure that all code examples are **complete and well-explained**.
- **Clarity and Organization:** Present information in a **clear, logical order** to facilitate implementation by another LLM agent.
- **Style Guidelines:**
    - Use clear and professional language appropriate for a technical audience.
    - Structure your guide with headings, subheadings, and bullet points for readability.
    - Avoid ambiguity and ensure precision in explanations and code.
"""
prompt_retriever_chain = ChatPromptTemplate.from_template(prompt_docs)
# Chain to retrieve documents using a vector store retriever and formatting them
retriever_gpt = get_retriever(2, 4)
docu_retrieve_chain = retriever_gpt | format_docs
# More complex template for tutorial writing, generating comprehensive documentation
chain_docs_docu = (
    {"context": itemgetter("task") | docu_retrieve_chain, "task": itemgetter("task"), "instruction": itemgetter("instruction"), "world": itemgetter("world")}
    | prompt_retriever_chain
    | llm_mini_main
    | StrOutputParser()
)


## Code Tool Chain -----------------------------
# PyCram Code Retriever
prompt_code = """**You are an experienced technical writer and coding educator specializing in creating detailed and precise tutorials. You are a Tool in an LLM Agent structure and you get a subtask for which you gather information.**

**Your task is to craft a comprehensive guide on how to use the provided function within the PyCram framework, based on the given documentation and code context. You should not only explain the function itself but also describe its relationship with other relevant functions and components within the context. Do not provide information how to install and setup PyCRAM because it is already done.**

---

### Context:
Here is the high level task and world knowledge. Use it just as context and only work on you subtask:
<high_level_task_context>
{instruction}

World knowledge:
{world}
</high_level_task_context>
Here is the retrieved context for your task:
{context}

---

### Your Subtask:
**Function:** {task}

---

### Instructions:

1. **Function Explanation and Contextualization**

   - **Detailed Description:** Begin with a comprehensive description of the function, including its purpose and how it fits within the PyCram framework.
   - **Syntax and Parameters:** Explain the function's syntax, input parameters, and return values.
   - **Integration:** Describe how this function integrates into the framework and its role within the overall context.
   - **Relationship with Other Components:** Discuss how the function interacts with other relevant functions and components in the context.

2. **Code Examples and Implementation**

   - **Full Function Code:** Provide the complete code of the function.
   - **Demonstration Snippets:** Include relevant code snippets from the context that demonstrate the function in action.
   - **Step-by-Step Explanation:** Explain how the code works step by step.
   - **Adaptation:** Show how the code can be adapted to solve similar tasks.

3. **General Framework Functionality**

   - **Fundamental Concepts:** Explain the fundamental functionality of the PyCram framework as it relates to the given function.
   - **Key Principles:** Discuss key concepts and principles necessary to understand the function and its application.
   - **Importance:** Highlight why these concepts are essential for effectively using the function.

4. **Best Practices and Recommendations**

   - **Effective Usage:** Provide guidance and best practices for effectively using the function and the framework.
   - **Common Pitfalls:** Mention potential pitfalls and how to avoid them.
   - **Optimization Tips:** Offer suggestions on optimizing the function's performance and reliability.

5. **Planning and Implementation for Developers**

   - **Implementation Plan:** Design a clear plan outlining how developers can implement the function in their own projects.
   - **Integration Steps:** Outline the necessary steps to correctly integrate and customize the function.
   - **Customization Guidance:** Provide advice on tailoring the function to meet specific project requirements.


### Important Notes:

- **Length:** Write as short as possible but ensure that everything important is included.
- **Code Examples:** Incorporate all essential code examples in their entirety.
- **Systematic Thinking:** Think systematically to ensure that another LLM agent can produce correct code based on your output.
- **Clarity and Organization:**
    - Use clear and professional language appropriate for a technical audience.
    - Structure your tutorial with headings, subheadings, and bullet points for readability.
    - Ensure explanations are precise and unambiguous.
"""
prompt_retriever_code = ChatPromptTemplate.from_template(prompt_code)

retriever_code = get_retriever(1, 4)
code_retrieve_chain = retriever_code | format_code
chain_docs_code = (
    {"context": itemgetter("task") | code_retrieve_chain, "task": itemgetter("task"), "instruction": itemgetter("instruction"), "world": itemgetter("world")}
    | prompt_retriever_code
    | llm_mini_main
    | StrOutputParser()
)


## Instruction preprocessing chain ------------------------------------------
preprocessing_prompt = """You are an intelligent planning agent for robots. Your task is to analyze a given instruction, world knowledge, and a URDF file, and break them down into the following components:

1. **Initial stage:** Describe the current state before the task begins.
2. **Goal stage:** Describe the desired end state after completing the task.
3. **Step-by-step action plan:** Create a detailed but concise sequence of actions that the robot must perform to move from the initial stage to the goal stage.

The output should be brief and concise, formatted as a clear instruction, similar to the provided instruction, but with more details. When you are not sure you can provide the right answer just try your best.

Example input:
- **Instruction:** Place the cereal box on the kitchen island.
- **World knowledge:** [kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), 
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
- **URDF file:** (The content of the file is very long, so no example here. The file would contain information about the components of the objects, such as 'kitchen.urdf' and 'pr2.urdf'.)


**Your Expected output:**
- **Initial stage:**  
    - **Cereal box:** Located on the ground approximately at position **[1.4, 1, 0.95]**.
    - **PR2 Robot:** Positioned near the cereal box, approximately at position **[1.4, 1, 0]**.

- **Goal stage:**  
    - **Cereal box:** Located on the kitchen island surface approximately at position **[-1.07, 1.72, 0.84]**.

- **Step-by-step plan:**

    1. **Robot positioning:**
    - **Action:** The PR2 robot moves to position **[1.4, 1, 0]**, near the cereal box.
    
    2. **Grabbing the cereal box:**
    - **Action:** The robot navigates to the cereal box at **[1.4, 1, 0.95]**.
    - **Action:** Securely grab the cereal box.

    3. **Movement to the kitchen island:**
    - **Action:** The robot transports the cereal box from **[1.4, 1, 0.95]** to the kitchen island approximately at **[-1.07, 1.72, 0.84]**.

    4. **Placing the cereal box:**
    - **Action:** Carefully place the cereal box on the kitchen island approximately at position **[-1.07, 1.72, 0.84]**.

    5. **Task completion:**
    - **Action:** Finish the task and return to the initial position or prepare for the next instruction.

Now do the Task for the Input:
**Input**
- **Instruction:** {prompt}
- **World knowledge:** {world}
- **URDF file:** 
<urdf>
{urdf}
</urdf>
"""

preprocessing_template = ChatPromptTemplate.from_template(preprocessing_prompt)
urdf_tool_template = ChatPromptTemplate.from_template("""You are a file summariser tool. You get a urdf file for an environment, a robot or an object. You also get a instruction and world knowledge. 
Your task is to summarize and compress the urdf-file and list the important data in it. Be sure that you list all the Data in the file which is important for the world understanding to accomplish the instruction.
Combine the summary and the listing with the world knowledge and create a world model in natural language with it. Be sure that the used information is correct.

Here is the input. Remember that the instruction is just to understand the context and not your task:
- **Instruction:** {prompt}
---
- **World knowledge:** {world}
---
- **URDF file:** 
<urdf>
{urdf}
</urdf>
""")

urdf_clean = (
    {"urdf": itemgetter("world") | RunnableLambda(extract_urdf_files), "prompt": itemgetter("prompt"), "world": itemgetter("world")}
    | urdf_tool_template
    | llm_mini_main
    | StrOutputParser()
)
urdf_chain = RunnableLambda(extract_urdf_files)


# More complex template for tutorial writing, generating comprehensive documentation
preprocessing_chain = (
    {
        "prompt": itemgetter("prompt"),
        "world": itemgetter("world"),
        "urdf": {"world": itemgetter("world"), "prompt": itemgetter("prompt")} | urdf_clean,
    }
    | preprocessing_template
    | llm_GP
    | StrOutputParser()
)

# Urdf Tool

urdf_tool = (
    {
        "prompt": itemgetter("prompt"),
        "world": itemgetter("world"),
        "urdf": itemgetter("urdf")
    }
    | urdf_tool_template
    | llm_mini_main
    | StrOutputParser()
)