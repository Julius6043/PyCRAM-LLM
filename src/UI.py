import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import threading
from speech_to_text import speech_to_text


class VoiceRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Recorder")
        self.recording = False
        self.frames = []

        self.label = tk.Label(root, text="Press the button to start recording")
        self.label.pack(pady=10)

        self.record_button = tk.Button(root, text="Record", command=self.toggle_recording)
        self.record_button.pack(pady=5)

        self.save_button = tk.Button(root, text="Save", command=self.save_recording, state=tk.DISABLED)
        self.save_button.pack(pady=5)

    def toggle_recording(self):
        if self.recording:
            self.recording = False
            self.record_button.config(text="Record")
        else:
            self.recording = True
            self.record_button.config(text="Stop")
            self.save_button.config(state=tk.DISABLED)
            self.frames = []
            self.record_thread = threading.Thread(target=self.record)
            self.record_thread.start()

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)

        while self.recording:
            data = stream.read(1024)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        self.save_button.config(state=tk.NORMAL)
        messagebox.showinfo("Info", "Recording stopped. You can now save the file.")

    def save_recording(self):
        filename = "recording.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        messagebox.showinfo("Info", f"Recording saved as {filename}")
        self.save_button.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecorderApp(root)
    root.mainloop()