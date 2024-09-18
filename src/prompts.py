from langchain_core.prompts import ChatPromptTemplate
from vector_store_SB import get_retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from helper_func import (
    format_docs,
    format_code,
    format_examples,
    format_example,
    llm,
    llm_GP,
    llm_mini,
)


## rewoo planer ----------------------------------
rewoo_planer_prompt = """You are a renowned AI engineer and programmer. You receive world knowledge and a task. You use them to develop a detailed sequence of plans to creat PyCramPlanCode for a robot that enables the robot to perform the 
task step by step. Concentrate on using Action Designators over MotionDesignators. For each plan, indicate which external tool, along with the input for the tool, is used to gather evidence. You 
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
    ActionDesignators
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
rewoo_planner = prompt_template | llm_GP

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
codecheck_planner = prompt_template_codecheck_planer | llm_GP

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
    ActionDesignators
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
prompt_docs = """You are an experienced technical writer and coding educator, specializing in creating comprehensive guides for implementing specific tasks and workflows using the PyCram framework. 
Your task is to thoroughly explain how to accomplish the given task within PyCram, based on the provided context. 
You should research and extract all relevant information, summarizing and organizing it in a way that enables another LLM agent to efficiently implement the workflow in code.

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

Use 4000 tokens for your explanation.
Ensure that all necessary code examples are complete and well-explained.
Organize the information in a clear, logical order to facilitate implementation by another LLM agent.
"""
prompt_retriever_chain = ChatPromptTemplate.from_template(prompt_docs)
# Chain to retrieve documents using a vector store retriever and formatting them
retriever_gpt = get_retriever(2, 4)

# More complex template for tutorial writing, generating comprehensive documentation
chain_docs_docu = (
    {"context": retriever_gpt | format_docs, "task": RunnablePassthrough()}
    | prompt_retriever_chain
    | llm_mini
    | StrOutputParser()
)


## Code Tool Chain -----------------------------
# PyCram Code Retriever
prompt_code = """You are an experienced technical writer and coding educator, specializing in creating detailed and precise tutorials.
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
prompt_retriever_code = ChatPromptTemplate.from_template(prompt_code)

retriever_code = get_retriever(1, 4)

chain_docs_code = (
    {"context": retriever_code | format_code, "task": RunnablePassthrough()}
    | prompt_retriever_code
    | llm_mini
    | StrOutputParser()
)


## Instruction preprocessing chain ------------------------------------------
preprocessing_prompt = """Du bist ein intelligenter Planungsagent für Roboter. Deine Aufgabe ist es, eine gegebene Anweisung, das Weltwissen und eine URDF-Datei zu analysieren und diese in die folgenden Komponenten zu zerlegen:

1. **Ausgangsstadium:** Beschreibe den aktuellen Zustand vor Beginn der Aufgabe.
2. **Zielstadium:** Beschreibe den gewünschten Endzustand nach Ausführung der Aufgabe.
3. **Schritt-für-Schritt-Handlungsanweisung:** Erstelle eine detaillierte, aber prägnante Abfolge von Aktionen, die der Roboter ausführen muss, um vom Ausgangsstadium zum Zielstadium zu gelangen.

Die Ausgabe soll kurz und prägnant sein und in Form einer klaren Anweisung verfasst werden, wie die übergebene Anweisung, jedoch mit mehr Details. Gebe dabei die übergebene Anweisung mit aus.

**Eingabeformat:**

- **Anweisung:** {prompt}
- **Weltwissen:** {worldknowledge}
- **URDF-Datei:** 
<urdf>
{urdf}
</urdf>


**Beispiel:**

- **Anweisung:** Place the cereal box on the kitchen island.
- **Weltwissen:** [kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), 
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), 
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', pose=Pose([1.4, 1, 0.95])))]
- **URDF-Datei:** (Inhalt der Datein sind sehr lang, also kommt hier kein Beispiel. Hier würde der Inhalt von 'kitchen.urdf' und 'pr2.urdf' stehen. Diese enthalten Informationen über die Komponenten der Objekte.)

**Erwartete Ausgabe:**
- **Grundanweisung**: Place the cereal box on the kitchen island.
- **Ausgangsstadium:**  
    - **Cerealschachtel:** Befindet sich auf dem Boden bei Position **[1.4, 1, 0.95]**.
    - **Roboter PR2:** Positioniert sich in der Nähe der Cerealschachtel, beispielsweise bei Position **[1.4, 1, 0]**.

- **Zielstadium:**  
    - **Cerealschachtel:** Befindet sich auf der Kücheninsel an der Oberfläche bei Position **[-1.07, 1.72, 0.84]**.

- **Schritt-für-Schritt-Anleitung:**

    1. **Positionierung des Roboters:**
    - **Aktion:** Der Roboter PR2 bewegt sich zur Position **[1.4, 1, 0]**, nahe der Cerealschachtel.
    
    2. **Greifen der Cerealschachtel:**
    - **Aktion:** Der Roboter navigiert zur Cerealschachtel bei **[1.4, 1, 0.95]**.
    - **Aktion:** Greife die Cerealschachtel sicher.

    3. **Bewegung zur Kücheninsel:**
    - **Aktion:** Der Roboter transportiert die Cerealschachtel von **[1.4, 1, 0.95]** zur Kücheninsel bei **[-1.07, 1.72, 0.84]**.

    4. **Platzieren der Cerealschachtel:**
    - **Aktion:** Positioniere die Cerealschachtel vorsichtig auf der Kücheninsel an der genauen Position **[-1.07, 1.72, 0.84]**.

    5. **Abschluss der Aufgabe:**
    - **Aktion:** Überprüfe, ob die Cerealschachtel stabil auf der Kücheninsel platziert ist.
    - **Aktion:** Beende die Aufgabe und kehre zur Ausgangsposition zurück oder bereite dich auf die nächste Anweisung vor.
"""

preprocessing_template = ChatPromptTemplate.from_template(preprocessing_prompt)


# More complex template for tutorial writing, generating comprehensive documentation
preprocessing_chain = (
    {
        "prompt": itemgetter("prompt"),
        "worldknowledge": itemgetter("world"),
        "urdf": itemgetter("urdf"),
    }
    | preprocessing_template
    | llm
    | StrOutputParser()
)
