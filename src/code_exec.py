import multiprocessing
import traceback


def execute_code_in_process(code):
    """
    Diese Funktion erstellt einen separaten Prozess, um den übergebenen Code auszuführen,
    und gibt das Ergebnis der Ausführung zurück.
    """
    def code_executor(code, result_queue):
        """
        Diese Funktion wird im separaten Prozess ausgeführt.
        Sie führt den übergebenen Code aus und sendet das Ergebnis über eine Queue zurück.
        """
        try:
            exec(code)
            result_queue.put("SUCCESS")  # Signalisieren, dass der Code erfolgreich ausgeführt wurde
        except Exception as e:
            result_queue.put(f"Execution Error: {traceback.format_exc()}")  # Senden der Fehlermeldung zurück
            try:
                exec("world.exit()")
            except Exception as ee:
                print("No World has been exited")


    # Erstellen einer Queue für die Kommunikation zwischen den Prozessen
    result_queue = multiprocessing.Queue()

    # Erstellen und Starten eines separaten Prozesses, der code_executor ausführt
    process = multiprocessing.Process(target=code_executor, args=(code, result_queue))
    process.start()

    # Warten auf die Beendigung des Prozesses
    process.join()

    # Ergebnis aus der Queue abrufen
    result = result_queue.get()
    return result


"""
from pycram.bullet_world import BulletWorld, Object
from pycram.designators.action_designator import *
from pycram.designators.motion_designator import *
from pycram.designators.location_designator import *
from pycram.designators.object_designator import *
from pycram.process_module import simulated_robot
from pycram.pose import Pose
from pycram.enums import ObjectType
code = Miau
"""
#result = execute_code_in_process(code)
#print(result)
