# PyCRAM-LLM

Dieses Repository dient als Arbeitsbasis, um mithilfe von Large Language Models (LLMs) automatisiert PyCRAM-Pläne für Roboter zu erstellen und zu testen. Hierbei kommen verschiedene Module für das Planen, Abrufen von Code-Snippets und Dokumentationstexten, Ausführen und Überprüfen von Code zum Einsatz.

## Übersicht

- **PyCRAM**: Ein Framework, das die Planung und Ausführung von Roboteraktionen abstrahiert (z.B. Objekte greifen, navigieren, Objekte platzieren).
- **LangChain-Module**: Nutzen LLM-Funktionen zum Durchsuchen von Code, Dokumentationen und zum Generieren von PyCRAM-Code.
- **Supabase**: Dient als Vektordatenbank, in der Schnipsel von Code- oder Dokumentationstexten für das Retrieval abgelegt werden.

Das Ziel ist es, automatisiert mit einem Textkommando („Stelle das Müsli auf den Tisch“) mittels LLM einen ausführbaren PyCRAM-Plan zu erzeugen und dessen Korrektheit durch Code-Execution-Checks sicherzustellen.

## Hauptfunktionen und Ablauf

1. **Eingabe eines Befehls**: Der Nutzer gibt einen Befehl wie „Bringe die Milch von A nach B“ zusammen mit Weltwissen zu vorhandenen Objekten (z.B. `milk`, `bowl`, `kitchen`, `pr2` etc.) ein.
2. **Vorverarbeitung**: Ein LLM versucht, den Befehl in eine präzisere Reihe von Schritten zu übersetzen (z.B. „Fahre zur Milch“, „Greife Milch“, „Fahre zum Tisch“, „Stelle Milch ab“).
3. **Plan-Generierung**: Auf Basis von Tool-Anfragen (z.B. Abruf von Code-Dokumentation oder URDF-Dateien im Vektorspeicher) wird ein PyCRAMPlanCode erstellt.
4. **Code-Ausführung und Fehlerprüfung**: In einem separaten Prozess wird der generierte Code ausgeführt und geprüft. Tritt ein Fehler auf, so wird automatisch ein neuer Plan- und Code-Verbesserungszyklus gestartet.
5. **Speicherung**: Anschließend wird der erfolgreich ausgeführte Code als Beispiel bzw. „Erfolgsfall“ in der Vektordatenbank gespeichert.

## Installation

1. **Repository klonen**
   ```bash
   git clone https://github.com/Julius6043/PyCRAM-LLM.git
   cd PyCRAM-LLM
   ```
2. **Python-Umgebung vorbereiten**
   - Mindestens Python 3.9 wird empfohlen.
   - Optional: virtuelles Environment anlegen und aktivieren.
3. **Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```
4. **Umgebungsvariablen setzen**
   - `.env` anhand der Vorlage `env_example` erstellen und alle Schlüssel (z.B. OpenAI-, Supabase-Keys) eintragen.

## Repository-Struktur (Auswahl)

    PyCRAM-LLM/
    ├── code_scraper.py
    ├── env_example
    ├── requirements.txt
    ├── setup.py
    ├── src/
    │   ├── ReWOO_codeCheck.py
    │   ├── ReWOO_codeCheck_parallel.py
    │   ├── ReWOO_parallel.py
    │   ├── automatic_plan.py
    │   ├── code_exec.py
    │   ├── helper_func.py
    │   ├── langgraph_ReWOO.py
    │   ├── langgraph_code_assistant.py
    │   ├── langgraph_code_assistant_parallel.py
    │   ├── prompts.py
    │   ├── run_llm_local.py
    │   └── vector_store_SB.py
    └── README.md (dieses Dokument)

### Wichtige Bestandteile

- **`code_scraper.py`**
  Skript zum Einsammeln von Python-, Markdown- und URDF-Dateien sowie deren Aufteilung in Chunks, um sie in der Vektordatenbank speichern zu können.

- **`setup.py`**
  Zeigt ein mögliches Setup an, um Daten aus PyCRAM-Code (z.B. `src/pycram`) ins Vektorsystem zu laden.

- **`src/`**
  Enthält die wesentlichen Module und die Logik, die den LLM-Ablauf, das Retrieval sowie die Ausführung und Überprüfung von Code steuern.
  - **`langgraph_*`**: Verwenden StateMachine/Graph-Logik (ReWOO) für Planerstellung, Codekorrektur und Ausführung.
  - **`helper_func.py`**: Sammelfunktionen wie Token-Zählung, Formatierungshelfer, definierte LLM-Instanzen.
  - **`prompts.py`**: Enthält Prompt-Texte, die im LLM-Ablauf genutzt werden (z.B. für das generische Tooling, Code-Abfragen etc.).
  - **`vector_store_SB.py`**: Verknüpfung zur Supabase als Vektordatenbank; Funktionen zum Laden/Abfragen von Dokumenten-Snippets.
  - **`code_exec.py`**: Führt den in Textform generierten Code in einem separaten Python-Prozess aus und fängt Fehler ab, um die nächste Iteration der Code-Verbesserung anzustoßen.

## Nutzung

1. **Start**: Ein typischer Workflow beginnt mit dem Skript `test_main.py` oder den Funktionen in `langgraph_code_assistant(_parallel).py`. Dort wird eine Funktion aufgerufen, die auf einen Nutzereingabebefehl reagiert.
2. **Befehl formulieren**: Etwa: “Kannst du bitte das Müsli vom Tisch nehmen und in den Schrank stellen?”
3. **Automatisierter Prozess**:
   - Der Befehl wird vorverarbeitet (zerlegt in einzelne Aktionen).
   - Benötigte Code-Snippets oder Hilfstexte aus dem Vektorspeicher werden abgerufen.
   - Die LLM-Pipeline generiert daraufhin PyCRAM-Code.
   - Der Code wird in einem separaten Prozess ausgeführt. Bei Fehlern startet eine Korrektrumrunde.
4. **Ergebnis**: Nach erfolgreicher Ausführung liegt ein funktionsfähiger PyCRAMPlanCode vor, der auch direkt in der Datenbank gespeichert wird.

## Häufige Fehlerquellen

- **Falscher Pfad oder fehlende URDF-Dateien**: Prüfe, ob alle URDFs (z.B. kitchen.urdf, pr2.urdf) korrekt in `vector_store_SB.py` eingebunden sind.
- **Fehlende Abhängigkeiten**: Achte auf installierte Pakete aus der `requirements.txt`.
- **Falsche API Keys**: In der `.env` sollten alle Keys (OpenAI, Supabase etc.) korrekt sein.

## Weiterentwicklung

- **Erweiterte Abdeckung**: Weitere Tools (z.B. Web-Suche) könnten integriert werden, um Code- oder URDF-Dateien dynamisch zu finden.
- **Mehr Beispielwelten**: Zusätzliche URDF-Dateien oder Umgebungsmodelle steigern die Bandbreite der erkennbaren Aktionen.
- **Verbesserte Zuverlässigkeit**: Verfeinerung der Code-Korrektur-Logik bei komplexen Fehlermeldungen.

---

Bei Fragen oder Problemen einfach ein Issue eröffnen oder einen Pull Request stellen. Viel Spaß beim Experimentieren mit dem PyCRAM-LLM-Projekt!
