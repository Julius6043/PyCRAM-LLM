import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from vector_store_SB import retrieve_large

load_dotenv()


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str


llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
llm3 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)
Use a format which can be recognised by this regex pattern:
Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]


Tools can be one of the following:
(1) Google[input]: Worker that searches results from Google. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin! 
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}"""


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
        result = retrieve_large(tool_input, 10)
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

task = """A fährt nach reichlichem Alkoholkonsum, der ihn gleichwohl nicht an seiner Fahr
tüchtigkeit zweifeln lässt, am späten Abend mit seinem Pkw nach Hause. Da wenig
Verkehr herrscht und er es eilig hat, fährt A nicht mit der zulässigen Höchstgeschwin
digkeit von 70 km/h, sondern etwas über 90 km/h. Als A ein Waldgebiet durchfährt,
tritt plötzlich die F, die sich im Wald verlaufen hat, hinter einem Baum hervor auf
die Straße, um das herannahende Fahrzeug auf sich aufmerksam zu machen. Obgleich
A sofort bremst, kann er nicht mehr verhindern, dass F von seinem Fahrzeug erfasst
und schwer verletzt wird. Da A sein Mobiltelefon zu Hause vergessen hat und er
auch nicht über eine andere Möglichkeit verfügt, einen Rettungswagen herbeizurufen,
entschließt er sich, die bewusstlose F mit seinem Pkw selbst in das nächstgelegene
Krankenhaus zu bringen. Er hält sich dabei weiterhin für fahrtüchtig, da er die Kollisi
on mit F nicht auf seine Alkoholisierung zurückführt.
Während der Fahrt zum Krankenhaus überschreitet A die zulässige Höchstgeschwin
digkeit erneut um 20 km/h. Infolge seiner Alkoholisierung - die Blutalkoholkonzentra
tion von A beträgt 1,2 Promille - bemerkt er einen anderen Pkw, der von rechts auf
einer Vorfahrtstraße seine Fahrbahn kreuzen will, zu spät. Ein Unfall kann jedoch
durch eine geschickte Reaktion des anderen Fahrers vermieden werden. A erreicht das
Krankenhaus 20 Minuten später ohne weitere Zwischenfälle. Dort kann das Leben der
F durch eine umgehend eingeleitete Notoperation gerettet werden. Wäre F nur zehn
Minuten später eingeliefert worden, wäre ihre Rettung nicht mehr möglich gewesen.
Vom Krankenhaus fährt A mit seinem Pkw nach Hause, ohne dort seine Personalien
zu hinterlassen oder das Eintreffen der Polizei abzuwarten; er wird jedoch kurze Zeit
später als Unfallverursacher ermittelt.
In dem anschließenden Strafverfahren macht A geltend, dass der Unfall sicher auch
dann eingetreten wäre, wenn er die zulässige Geschwindigkeit von 70 km/h eingehal
ten und nicht unter Alkoholeinfluss gestanden hätte. Der Staatsanwalt hält ihm entge
gen, dass er zum einen aufgrund seiner Alkoholisierung von vornherein kein Fahrzeug
mehr hätte führen dürfen; jedenfalls hätte er aber aufgrund seiner alkoholbedingten
Fahruntüchtigkeit nur mit einer Geschwindigkeit von 50 km/h fahren dürfen. Bei
dieser Geschwindigkeit hätte der Zusammenstoß mit F durch sofortiges Bremsen aber
noch vermieden werden können.
1. Wie hat sich A nach dem StGB strafbar gemacht? Gegebenenfalls erforderliche
Strafanträge sind gestellt.
2. Wie wäre zu entscheiden, wenn ein Sachverständiger feststellte, dass bei einer
Geschwindigkeit von nur 70 km/h der Unfall von einem nüchternen Fahrer mit
30 prozentiger Wahrscheinlichkeit vermieden worden wäre?"""

for s in app.stream({"task": task}):
    print(s)
    print("---")

print(s[END]["result"])
