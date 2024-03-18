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
    return "\n\n---\n\n".join([d.page_content for d in docs])


class ReWOO(TypedDict):
    task: str
    world: str
    plan_string: str
    steps: List
    results: dict
    result: str


llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
llm3 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = """Du bist ein renommierter AI Enginer und Programmierer. Du bekommst Weltwissen und eine Aufgabe, Aufforderung oder Frage übergeben und sollst damit einen Ablaufplan zur entwicklung eines PyCramPlanCode für einen Roboter entwickeln, welcher den Roboter die Aufgabe ausführen lässt. Für jeden Plan, gib an, welches externe Werkzeug zusammen mit der Eingabe für das Werkzeug verwendet wird, um Beweise zu sammeln. Du kannst die Beweise in einer Variablen #E speichern, die später von anderen Werkzeugen aufgerufen werden kann. (Plan, #E1, Plan, #E2, Plan, ...).
Verwende ein Format, das durch folgendes Regex-Muster erkannt werden kann:
Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*([^]+)]

Die Werkzeuge können eines der folgenden sein:
(1) Retrieve[input]: Ein Vektor-Datenbank-Abrufsystem, welches die Dokumentation von PyCram enthält. Verwende dieses Werkzeug, wenn du Informationen über Funktionen von PyCram benötigst.
Die Eingabe sollte eine Suchanfrage sein.
(2) LLM[input]: Ein vortrainiertes LLM wie du selbst. Nützlich, wenn du mit allgemeinem Weltwissen und gesundem Menschenverstand handeln musst. Bevorzuge es, wenn du zuversichtlich bist, das Problem selbst lösen zu können. Die Eingabe kann jede Anweisung sein.
(3) Statement[state]: Es wird kein Tool benutzt sondern einfach ein Statement direkt verfasst. Nützlich, wenn du einfach nur Informationen aus dem Input extrahieren möchtest.

PyCramPlanCode folgt der folgenden Struktur:
'from pycram.bullet_world import BulletWorld, Object
from pycram.process_module import simulated_robot
from pycram.designators.motion_designator import *
from pycram.designators.location_designator import *
from pycram.designators.action_designator import *
from pycram.designators.object_designator import *
from pycram.enums import ObjectType

world = BulletWorld()
kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf')
robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf')
cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', position=[1.4, 1, 0.95])

cereal_desig = ObjectDesignatorDescription(names=['cereal'])
kitchen_desig = ObjectDesignatorDescription(names=['kitchen'])
robot_desig = ObjectDesignatorDescription(names=['pr2']).resolve()

with simulated_robot:
    ParkArmsAction([Arms.BOTH]).resolve().perform()

    MoveTorsoAction([0.3]).resolve().perform()

    pickup_pose = CostmapLocation(target=cereal_desig.resolve(), reachable_for=robot_desig).resolve()
    pickup_arm = pickup_pose.reachable_arms[0]

    NavigateAction(target_locations=[pickup_pose.pose]).resolve().perform()

    PickUpAction(object_designator_description=cereal_desig, arms=[pickup_arm], grasps=['front']).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

    place_island = SemanticCostmapLocation('kitchen_island_surface', kitchen_desig.resolve(), cereal_desig.resolve()).resolve()

    place_stand = CostmapLocation(place_island.pose, reachable_for=robot_desig, reachable_arm=pickup_arm).resolve()

    NavigateAction(target_locations=[place_stand.pose]).resolve().perform()

    PlaceAction(cereal_desig, target_locations=[place_island.pose], arms=[pickup_arm]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

world.exit()'


Der dazugehörige Ablaufplan der zu diesen PyCramPlanCode führen soll, sieht beispielsweise so aus:
Weltwissen: [kitchen = Object('kitchen', ObjectType.ENVIRONMENT, 'kitchen.urdf'), robot = Object('pr2', ObjectType.ROBOT, 'pr2.urdf'), cereal = Object('cereal', ObjectType.BREAKFAST_CEREAL, 'breakfast_cereal.stl', position=[1.4, 1, 0.95]))]
Aufgabe: Kannst du das Müsli auf die Kücheninsel stellen?
Plan 1: Ermittle die Grundlagen von PyCram, einschließlich der Erstellung der simulierten Welt und der Initialisierung von Objekten. #E1 = Retrieve[PyCram Grundlagen]
Plan 2: Erforsche, wie in PyCram Objekt- und Roboterdesignatoren verwendet werden. #E2 = Retrieve[Verwendung von Designatoren in PyCram]
Plan 3: Untersuche, wie Bewegungs- und Aktionsmodule in PyCram für Navigation, Greifen und Platzieren von Objekten genutzt werden. #E3 = Retrieve[Bewegungs- und Aktionsmodule in PyCram]
Plan 4: Erforsche Best Practices zur Implementierung von Aktionssequenzen in PyCram, einschließlich Fehlerbehandlung und Zustandsüberprüfung. #E4 = Retrieve[Implementierung von Aktionssequenzen in PyCram]
Plan 5: Ermittle die beste Vorgehensweise zum ordnungsgemäßen Schließen der simulierten Umgebung und zum Freigeben von Ressourcen in PyCram. #E5 = Retrieve[Beenden der Simulation und Ressourcenmanagement in PyCram]

Beginne!
Beschreibe deine Pläne mit reichen Details. Jeder Plan sollte nur einer #E folgen.

Weltwissen: {world}
Aufgabe: {task}"""


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
retriever = get_retriever(2, 10)
re_chain = retriever | format_docs
prompt_retriever_chain = ChatPromptTemplate.from_template(
    """Given the search query, write a detailed, structured, and comprehensive article on this topic based on all the important information from the following context:
{context}

Search query: {task}
Use at least 4000 tokens for the output and adhere to the provided information. Incorporate important code examples in their entirety.
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
contain irrelevant information.

{plan}

Now solve the question or task according to provided evidence above. Respond with nothing other than the generated python code

Task: {task}
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
    prompt = solve_prompt.format(plan=plan, task=state["task"])
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

# app = graph.compile()

task = """A, B, C und O sind Arbeitskollegen und gehen praktisch jeden Samstag ins Stadion. Während A, B und 
C aber auch befreundet sind, können sich A und O nicht leiden. 
Im März 2013, C ist dieses eine Mal ausnahmsweise nicht dabei, verabreden sich A und B, nach dem 
Stadionbesuch den O „aufzumischen und ihm sein Handy zu klauen“. Sie wollen das vor ihrer 
Stammkneipe tun, in der sie sich wöchentlich treffen und in der auch O regelmäßig Gast ist. Wie 
immer nimmt B den A in seinem Wagen mit. Auf der Fahrt dorthin überlegt es sich B doch anders und 
sagt A, dass er nicht mitmache. B denkt sich dabei, dass der A die geplante Tat alleine nicht begehen 
werde und fährt weiter Richtung Stammkneipe. Dort angekommen steigen A und B aus, A geht 
alleine auf O los, schlägt ihn mehrfach und sagt ihm, dass er O „das Handy wegnehmen wird, weil er 
es eh nicht braucht“. O denkt sich, dass er sein Mobiltelefon in jedem Fall verloren hat und reicht es 
A hin, um weitere Schläge zu vermeiden. A und B gehen ohne O, der wenig später A und B bei der 
Polizei anzeigt, in die Kneipe. 
Nach mehreren Monaten kommt es zur Anklage. A will diese ganze Angelegenheit möglichst 
unbeschadet überstehen und den C hierzu einspannen. A berichtet ihm von der polizeilichen 
Untersuchung. Er sagt zu C: „Aber hier, das kann doch gar nicht sein, wir fahren doch immer 
gemeinsam und das hättest du doch gesehen, wenn das passiert wäre.“ C erinnert sich nicht genau, 
hält das aber für zutreffend. In der Hauptverhandlung sagt C als Zeuge aus, dass er mit A und B 
immer und so auch an dem fraglichen Tag zusammen zu ihrer Kneipe fahre, und dass er keinen 
Vorfall zwischen A und O gesehen hat. 
"""

# for s in app.stream({"task": task}):
# print(s)
# print("---")

# print(s[END]["result"])


result = chain_docs.invoke("PyCram Grundlagen")
print(result)
