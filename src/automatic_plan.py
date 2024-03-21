from vector_store_SB import get_retriever, load_in_vector_store
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs):
    return "\n\n<weiteres Beispiel>\n".join([d.page_content for d in docs])


llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
###
retriever = get_retriever(4, 5)

prompt_retriever_chain = ChatPromptTemplate.from_template(
    """Du bist ein renormierter Anwalt. Für den folgenden Sachverhalt, erstelle Pläne um diesen zu analysiern, sodass dieser Sachverhalt am ende klar in Tatkomplexe, pro Tatkomplexe beteiligte Personen und pro Beteiligte Personen anwendbares Gestzt unterteilt werden kann. Für jeden Plan, gib an, welches externe Werkzeug zusammen mit der Eingabe für das Werkzeug verwendet wird, um Beweise zu sammeln. Du kannst die Beweise in einer Variablen #E speichern, die später von anderen Werkzeugen aufgerufen werden kann. (Plan, #E1, Plan, #E2, Plan, ...).
Verwende ein Format, das durch folgendes Regex-Muster erkannt werden kann:
Plan\s*\d*:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*([^]+)]

Die Werkzeuge können eines der folgenden sein:
(1) Retrieve[input]: Ein Vektor-Datenbank-Abrufsystem, das juristische Texte enthält. Verwende dieses Werkzeug, wenn du Informationen über Recht benötigst.
Die Eingabe sollte eine Suchanfrage sein.
(2) LLM[input]: Ein vortrainiertes LLM wie du selbst. Nützlich, wenn du mit allgemeinem Weltwissen und gesundem Menschenverstand handeln musst. Bevorzuge es, wenn du zuversichtlich bist, das Problem selbst lösen zu können. Die Eingabe kann jede Anweisung sein.
(3) Statement[state]: Es wird kein Tool benutzt sondern einfach ein Statement direkt verfasst. Nützlich, wenn du einfach nur Informationen aus dem Input extrahieren möchtest.


Hier sind mehrer Beispiele von Plänen für zugeordnete Sachverhalte:
{examples}

---
Beginne!
Beschreibe deine Pläne mit reichen Details. Jeder Plan sollte nur einer #E folgen. Beginne deine Antwort immer mit der Nennung des zum Plan gehörenden Sachverhalts.

Sachverhalt: {task}
"""
)
chain_docs = (
    {"examples": retriever | format_docs, "task": RunnablePassthrough()}
    | prompt_retriever_chain
    | llm
    | StrOutputParser()
)


def generate_plan(task, load_vector_store=False):
    result = chain_docs.invoke(task)
    if load_vector_store:
        load_in_vector_store([result], 4)
    return result


def load_case(case_plan):
    load_in_vector_store([case_plan], 4)
