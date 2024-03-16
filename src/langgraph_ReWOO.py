import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from src.vector_store_SB import get_retriever

load_dotenv()


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str


llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
llm3 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = """Für den folgenden Sachverhalt, erstelle Pläne nach dem Gutachtenstil, die das Problem Schritt für Schritt lösen können. Für jeden Plan, gib an, welches externe Werkzeug zusammen mit der Eingabe für das Werkzeug verwendet wird, um Beweise zu sammeln. Du kannst die Beweise in einer Variablen #E speichern, die später von anderen Werkzeugen aufgerufen werden kann. (Plan, #E1, Plan, #E2, Plan, ...).
Verwende ein Format, das durch folgendes Regex-Muster erkannt werden kann:
Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*([^]+)]

Die Werkzeuge können eines der folgenden sein:
(1) Retrive[input]: Ein Vektor-Datenbank-Abrufsystem, das juristische Texte enthält. Verwende dieses Werkzeug, wenn du Informationen über Recht benötigst.
Die Eingabe sollte eine Suchanfrage sein.
(2) Google[input]: Ein Werkzeug, das Ergebnisse von Google sucht. Nützlich, wenn du kurze und prägnante Antworten zu einem spezifischen Thema finden musst. Die Eingabe sollte eine Suchanfrage sein.
(3) LLM[input]: Ein vortrainiertes LLM wie du selbst. Nützlich, wenn du mit allgemeinem Weltwissen und gesundem Menschenverstand handeln musst. Bevorzuge es, wenn du zuversichtlich bist, das Problem selbst lösen zu können. Die Eingabe kann jede Anweisung sein.

Zum Beispiel,
Aufgabe: A lässt absichtlich die Vase des B fallen, um ihn zu schädigen. Die Vase zerspringt dabei in 1000 Einzelteile.
Plan: Angesichts dessen, dass Thomas x Stunden gearbeitet hat, übersetze das Problem in algebraische Ausdrücke und löse es mit Wolfram Alpha. #E1 = WolframAlpha[Löse x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Finde heraus, wie viele Stunden Thomas gearbeitet hat. #E2 = LLM[Was ist x, gegeben #E1]
Plan: Berechne, wie viele Stunden Rebecca gearbeitet hat. #E3 = Rechner[(2 ∗ #E2 − 10) − 8]

Beginne!
Beschreibe deine Pläne mit reichen Details. Jeder Plan sollte nur einer #E folgen.

Aufgabe: {task}"""


# Regex to match expressions of the form E#... = ...[...]
regex_pattern = r"Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | llm


def get_plan(state: ReWOO):
    task = state["task"]
    result = planner.invoke({"task": task})
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
        retriver = get_retriever(5, 10)
        result = retriver.invoke(tool_input)
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""


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

app = graph.compile()
app.config["recursion_limit"] = 50

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
Aufgabe 1: Prüfen Sie in einem Gutachten die Strafbarkeit der Beteiligten nach dem StGB. 
Aufgabe 2: Im Zusammenhang mit Sitzblockaden und anderen expressiven Demonstrations
varianten „wütender Bürger“ fällt oft der Begriff des „zivilen Ungehorsams“. Wie verstehen 
Sie diesen Begriff? Nehmen Sie Stellung zu der These, „nicht gewalttätige“ Straftaten (z.B. 
§ 240 StGB oder Straftaten nach dem Versammlungsgesetz) seien wegen ihrer Eigenschaft als 
ziviler Ungehorsam gerechtfertigt, also nicht strafbar."""

for s in app.stream({"task": task}):
    print(s)
    print("---")

print(s[END]["result"])
