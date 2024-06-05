import paramiko
import time
import os

ssh_key = os.getenv("SSH_KEY_PASS")
# First run "ollama serve" on the server
def run_llama3_remote(input_text):
    # SSH-Verbindung einrichten
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('denver.informatik.uni-bremen.de', username='juliusha', passphrase=ssh_key, port=22)

    # Remote-Befehl mit Eingabetext ausführen
    command = f'source myenv/bin/activate && python3 ~/run_llama3_ollama.py "{input_text}"'
    stdin, stdout, stderr = client.exec_command(command)

    # Ergebnis auslesen
    result = stdout.read().decode('utf-8')
    print(stderr.read().decode('utf-8'))
    stdin.close()
    stdout.close()
    stderr.close()
    client.close()
    return result
