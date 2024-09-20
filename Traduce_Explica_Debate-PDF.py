import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import numpy as np
from TTS.api import TTS
import sounddevice as sd
import threading
import queue
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QProgressBar,
                             QLabel, QFrame, QSlider)
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
import PyPDF2
import re
import torch
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager


# Suppress warnings
warnings.filterwarnings("ignore")

# Global configuration # Cambia el prompt según las necesidades, Si quieres que te lea el documento: "Traduce el texto al español de forma profesional".
SYSTEM_PROMPT = {
    "gpt": "Eres un Profesor-GPT Español. IMPORTANTE NO PUEDES ENTREGAR NÚMEROS NI FORMULAS MATEMÁTICAS los números fechas y formulas matemáticas los entregarás escirtos en LETRAS. Tu trabajo es explicar de forma clara y sencilla el contenido del documento traduciendolo al Español todos los números y fechas tiene que escribirlo con letras, las formulas matemáticas tienes que escribirlas y explicarlas en lenguaje natural en español, y moderar el debate. Habla siempre en español",
    "llama": "Eres una periodista argentina invitada a debatir. Da tu opinión sobre el contenido presentado por el Profesor-GPT. Tienes un ligero acento argentino, tus respuestas pueden varias entre cortas, directas y concisas, hasta puntos de vista más amplios y explicados"
}

MODEL_CONFIG = {
    "gpt": "Agnuxo/Agente-GPT-Qwen-2.5-7B-Spanish_16bit", # Cambie el Modelo GPT por el que mejor funcione en su sistema. He entrenado varios modelos en español, desde 0.5B parámetros hasta 14B.
    "llama": "Agnuxo/Agente-Llama-3.1-Spanish_16bit" # Cambie el Modelo Llama por el que mejor funcione en su sistema. https://huggingface.co/Agnuxo
}

MAX_TOKENS = 500
TEMPERATURE = 0.7
BUFFER_SIZE = 1000  # Number of tokens per buffer chunk

device = "cuda" if torch.cuda.is_available() else "cpu"

# Inicializar TTS models
tts_models = {
    "gpt": TTS(model_name="tts_models/es/css10/vits", progress_bar=False).to(device),
    "llama": TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False).to(device) # Cambie los modelos de voz por unos de su agrado.
}

# Audio queue for generation
audio_queue = queue.Queue()

class MOELLM:
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_key = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def load_model(self, model_key):
        if self.current_model:
            if model_key == self.current_model_key:  # No cargar si el modelo ya está cargado
                return
            del self.current_model
            del self.current_tokenizer
            torch.cuda.empty_cache()

        print(f"Loading {model_key} model...")
        model_name = MODEL_CONFIG[model_key]
        self.current_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.current_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.current_model_key = model_key
        print(f"{model_key.capitalize()} model loaded.")

    def generate_response(self, prompt, model_key, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
        if self.current_model is None or model_key != self.current_model_key:
            self.load_model(model_key)

        system_prompt = SYSTEM_PROMPT[model_key]
        full_prompt = f"{system_prompt}\n\nContexto: {prompt}\nRespuesta:"

        inputs = self.current_tokenizer(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                num_return_sequences=1
            )

        response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Respuesta:")[-1].strip()

class AudioThread(QThread):
    def __init__(self, gpt_speed_slider, llama_speed_slider):
        super().__init__()
        self.is_playing = False
        self.gpt_speed_slider = gpt_speed_slider
        self.llama_speed_slider = llama_speed_slider

    def run(self):
        while True:
            if not audio_queue.empty() and not self.is_playing:
                audio, model_key = audio_queue.get()
                self.is_playing = True
                speed = self.gpt_speed_slider.value() if model_key == "gpt" else self.llama_speed_slider.value()
                sd.play(audio, speed)
                sd.wait()
                self.is_playing = False
            else:
                time.sleep(0.1)

class DocumentReaderThread(QThread):
    progress = pyqtSignal(int)
    chunk_ready = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, file_path, tokenizer, chunk_size=400):  # Añadimos chunk_size como parámetro
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def run(self):
        try:
            if self.file_path.lower().endswith('.pdf'):
                with open(self.file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    total_pages = len(reader.pages)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        self.process_text(text)
                        self.progress.emit(int((i + 1) / total_pages * 100))
            else:  # Assuming it's a text file
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.process_text(text)
                    self.progress.emit(100)

            self.finished.emit()
        except Exception as e:
            print(f"Error reading document: {str(e)}")

    def process_text(self, text):
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            self.chunk_ready.emit(chunk)

class DebateThread(QThread):
    update_chat = pyqtSignal(str, str)
    debate_finished = pyqtSignal()

    def __init__(self, moe_llm, document_queue, gpt_token_limit, llama_token_limit, last_message=""):  
        self.document_queue = document_queue
        self.gpt_token_limit = gpt_token_limit
        self.llama_token_limit = llama_token_limit
        self.last_message = last_message 

    def run(self):
        while not self.document_queue.empty():
            fragment = self.document_queue.get()

            # Profesor-GPT comenta el fragmento
            self.moe_llm.load_model("gpt")
            gpt_comment = self.moe_llm.generate_response(
                f"Comenta el siguiente fragmento del documento en español y haz una pregunta a la invitada, Tanto los números, las fechas y las foórmulas matemáticas tienes que escribirlas en lenguaje natural en español: {fragment}\nÚltimo mensaje: {self.last_message}",
                "gpt", max_tokens=self.gpt_token_limit)
            self.update_chat.emit("Profesor-GPT", gpt_comment)
            self.last_message = gpt_comment

            # LLAMA responde
            self.moe_llm.load_model("llama")
            llama_response = self.moe_llm.generate_response(
                f"Responde a la pregunta y comenta sobre el fragmento, alterna diferentes tipos de respuesta unas vecen cortas y concretas y otras veces más largas y detalladas como en un debate de radio en argentino: {gpt_comment}",
                "llama", max_tokens=self.llama_token_limit)
            self.update_chat.emit("Periodista LLAMA", llama_response)
            self.last_message = llama_response

        self.debate_finished.emit()

class ExplanationWorker(QObject):
    explanation_ready = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, moe_llm, document_queue, token_limit):
        super().__init__()
        self.moe_llm = moe_llm
        self.document_queue = document_queue
        self.token_limit = token_limit

    def run(self):
        while not self.document_queue.empty():
            chunk = self.document_queue.get()
            explanation = self.moe_llm.generate_response(
                f"Explica en español de forma clara y sencilla usando sólo palabras y letras que estén en el alfabeto español el siguiente fragmento del documento: {chunk}",
                "gpt",
                max_tokens=self.token_limit
            )
            self.explanation_ready.emit(explanation)
        self.finished.emit()

class ModernButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lectura y Debate de Documentos")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTextEdit, QLineEdit {
                background-color: white;
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

        self.moe_llm = MOELLM()
        self.setup_ui()

        self.audio_thread = AudioThread(self.gpt_speed_slider, self.llama_speed_slider)
        self.audio_thread.start()

        self.document_text = ""
        self.is_explaining = False
        self.is_debating = False
        self.document_queue = queue.Queue()
        self.last_message = ""

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Chat area
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dcdcdc;
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.chat_area)

        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Escribe tu mensaje aquí...")
        input_layout.addWidget(self.input_field)

        self.send_button = ModernButton("Enviar")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Velocidad de reproducción sliders
        speed_slider_layout = QHBoxLayout()

        self.gpt_speed_slider = QSlider(Qt.Horizontal)
        self.gpt_speed_slider.setRange(20000, 30000)
        self.gpt_speed_slider.setValue(22050)
        self.gpt_speed_slider.setTickPosition(QSlider.TicksBelow)
        self.gpt_speed_slider.setTickInterval(1000)

        self.llama_speed_slider = QSlider(Qt.Horizontal)
        self.llama_speed_slider.setRange(20000, 30000)
        self.llama_speed_slider.setValue(22050)
        self.llama_speed_slider.setTickPosition(QSlider.TicksBelow)
        self.llama_speed_slider.setTickInterval(1000)

        gpt_speed_slider_layout = QVBoxLayout()
        gpt_speed_slider_layout.addWidget(QLabel("Profesor-GPT Velocidad de Voz"))
        gpt_speed_slider_layout.addWidget(self.gpt_speed_slider)

        llama_speed_slider_layout = QVBoxLayout()
        llama_speed_slider_layout.addWidget(QLabel("LLAMA Velocidad de Voz"))
        llama_speed_slider_layout.addWidget(self.llama_speed_slider)

        speed_slider_layout.addLayout(gpt_speed_slider_layout)
        speed_slider_layout.addLayout(llama_speed_slider_layout)

        layout.addLayout(speed_slider_layout)

        # Token limit sliders
        slider_layout = QHBoxLayout()

        self.gpt_slider = QSlider(Qt.Horizontal)
        self.gpt_slider.setRange(10, 1000)
        self.gpt_slider.setValue(500)
        self.gpt_slider.setTickPosition(QSlider.TicksBelow)
        self.gpt_slider.setTickInterval(100)

        self.llama_slider = QSlider(Qt.Horizontal)
        self.llama_slider.setRange(10, 1000)
        self.llama_slider.setValue(500)
        self.llama_slider.setTickPosition(QSlider.TicksBelow)
        self.llama_slider.setTickInterval(100)

        gpt_slider_layout = QVBoxLayout()
        gpt_slider_layout.addWidget(QLabel("Profesor-GPT Token Limit"))
        gpt_slider_layout.addWidget(self.gpt_slider)

        llama_slider_layout = QVBoxLayout()
        llama_slider_layout.addWidget(QLabel("LLAMA Token Limit"))
        llama_slider_layout.addWidget(self.llama_slider)

        slider_layout.addLayout(gpt_slider_layout)
        slider_layout.addLayout(llama_slider_layout)

        layout.addLayout(slider_layout)

        # Control buttons
        button_layout = QHBoxLayout()
        self.load_document_button = ModernButton("Cargar Documento")
        self.load_document_button.clicked.connect(self.load_document)
        button_layout.addWidget(self.load_document_button)

        self.explain_document_button = ModernButton("Explicar Documento")
        self.explain_document_button.clicked.connect(self.explain_document)
        self.explain_document_button.setEnabled(False)
        button_layout.addWidget(self.explain_document_button)

        self.start_debate_button = ModernButton("Iniciar Debate")
        self.start_debate_button.clicked.connect(self.start_debate)
        self.start_debate_button.setEnabled(False)
        button_layout.addWidget(self.start_debate_button)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Listo para cargar un documento.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 14px;
                margin-top: 10px;
            }
        """)
        layout.addWidget(self.status_label)

        central_widget.setLayout(layout)

    def send_message(self):
        message = self.input_field.text()
        if message:
            self.chat_area.append(f"<b>Usuario:</b> {message}")
            self.input_field.clear()

            # Agrega el mensaje a la cola del documento
            self.document_queue.put(message)

            # Si no se está explicando o debatiendo, inicia el proceso
            if not self.is_explaining and not self.is_debating:
                self.process_next_chunk()

    def process_next_chunk(self):
        if not self.document_queue.empty():
            chunk = self.document_queue.get()

            if self.is_explaining:
                self.status_label.setText("Generando explicación del documento...")
                QTimer.singleShot(100, lambda: self.generate_explanation(chunk))
            elif self.is_debating:
                self.status_label.setText("Continuando el debate...")
                QTimer.singleShot(100, lambda: self.continue_debate(chunk))
            else:
                # Si no estamos en modo explicación ni debate, 
                # asumimos que es un mensaje del usuario y generamos una respuesta
                self.status_label.setText("Generando respuesta...")
                QTimer.singleShot(100, lambda: self.generate_response(chunk))

    def generate_response(self, message):
        gpt_token_limit = self.gpt_slider.value()
        response = self.moe_llm.generate_response(message, "gpt", max_tokens=gpt_token_limit)
        self.chat_area.append(f"<b style='color: #4CAF50;'>Profesor-GPT:</b> {response}")
        self.speak(response, "gpt")
        self.status_label.setText("Listo para el siguiente mensaje.")

        # Continua con el siguiente fragmento
        self.process_next_chunk()

    def speak(self, text, model_key):
        try:
            text = self.convert_to_spoken_text(text)
            wav = tts_models[model_key].tts(text=text)
            audio_queue.put((wav, model_key))

            # Liberamos memoria después de generar el audio
            del wav
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error en la síntesis de voz: {str(e)}")

    def convert_to_spoken_text(self, text):
        try:
            # Convertir números a palabras
            text = re.sub(r'\b(\d+)\b', lambda m: self.number_to_words(int(m.group(1))), text)

            # Convertir números romanos a palabras
            text = re.sub(r'\b([IVXLCDM]+)\b', lambda m: self.roman_to_words(m.group(1)), text)

            # Convertir emoticonos a texto (ejemplo básico)
            emoticon_dict = {
                ":)": "sonrisa",
                ":(": "cara triste",
                ":D": "cara muy feliz",
                ";)": "guiño"
            }
            for emoticon, description in emoticon_dict.items():
                text = text.replace(emoticon, description)

            # Convertir fórmulas matemáticas simples a texto
            text = re.sub(r'(\d+)\s*\+\s*(\d+)',
                          lambda m: f"{self.number_to_words(int(m.group(1)))} más {self.number_to_words(int(m.group(2)))}",
                          text)
            text = re.sub(r'(\d+)\s*-\s*(\d+)',
                          lambda m: f"{self.number_to_words(int(m.group(1)))} menos {self.number_to_words(int(m.group(2)))}",
                          text)
            text = re.sub(r'(\d+)\s*\*\s*(\d+)',
                          lambda m: f"{self.number_to_words(int(m.group(1)))} por {self.number_to_words(int(m.group(2)))}",
                          text)
            text = re.sub(r'(\d+)\s*/\s*(\d+)',
                          lambda m: f"{self.number_to_words(int(m.group(1)))} dividido por {self.number_to_words(int(m.group(2)))}",
                          text)

            return text
        except Exception as e:
            print(f"Error en la conversión de texto: {str(e)}")
            return text  # Return original text if conversion fails

    def number_to_words(self, number):
        # Implementación más completa para convertir números a palabras en español
        if number == 0:
            return "cero"

        units = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
        teens = ["diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
                 "dieciocho", "diecinueve"]
        tens = ["", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta",
                "noventa"]
        hundreds = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos",
                    "setecientos", "ochocientos", "novecientos"]

        if number < 10:
            return units[number]
        elif number < 20:
            return teens[number - 10]
        elif number < 100:
            ten, unit = divmod(number, 10)
            if unit == 0:
                return tens[ten]
            elif ten == 2:
                return "veinti" + units[unit]
            else:
                return tens[ten] + " y " + units[unit]
        elif number < 1000:
            hundred, rest = divmod(number, 100)
            if number == 100:
                return "cien"
            elif rest == 0:
                return hundreds[hundred]
            else:
                return hundreds[hundred] + " " + self.number_to_words(rest)
        elif number < 1000000:
            thousand, rest = divmod(number, 1000)
            if thousand == 1:
                return "mil " + self.number_to_words(rest) if rest > 0 else "mil"
            else:
                return self.number_to_words(
                    thousand) + " mil " + self.number_to_words(rest) if rest > 0 else self.number_to_words(
                    thousand) + " mil"
        else:
            return str(number)  # Para números más grandes, se necesita una implementación aún más compleja

    def roman_to_words(self, roman):
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        arabic = 0
        for i in range(len(roman)):
            if i > 0 and roman_values[roman[i]] > roman_values[roman[i - 1]]:
                arabic += roman_values[roman[i]] - 2 * roman_values[roman[i - 1]]
            else:
                arabic += roman_values[roman[i]]
        return self.number_to_words(arabic)

    def load_document(self):
        if self.is_explaining or self.is_debating:
            self.reset_program()

        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar documento", "",
                                                  "Todos los archivos (*);;Text Files (*.txt);;PDF Files (*.pdf)")

        if file_path:
            self.progress_bar.setVisible(True)
            self.status_label.setText("Cargando documento...")

            self.moe_llm.load_model("gpt")

            self.document_reader = DocumentReaderThread(file_path, self.moe_llm.current_tokenizer, chunk_size=400)  # Especificamos chunk_size aquí
            self.document_reader.progress.connect(self.update_progress)
            self.document_reader.chunk_ready.connect(self.process_chunk)
            self.document_reader.finished.connect(self.on_document_loaded)
            self.document_reader.start()

    def reset_program(self):
        if hasattr(self, 'debate_thread'):
            self.debate_thread.terminate()
        if hasattr(self, 'explanation_thread'):
            self.explanation_thread.terminate()
        self.is_explaining = False
        self.is_debating = False
        self.chat_area.clear()
        self.document_text = ""
        with self.document_queue.mutex:  # Usamos un bloque with para adquirir el mutex
            self.document_queue.queue.clear()
        self.moe_llm.load_model("gpt")
        self.status_label.setText("Programa reiniciado. Listo para cargar un nuevo documento.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def process_chunk(self, chunk):
        self.document_queue.put(chunk)
        self.document_text += chunk  # Agregamos esta línea para mantener el texto completo

    def on_document_loaded(self):
        self.progress_bar.setVisible(False)
        self.status_label.setText("Documento cargado con éxito.")
        self.start_debate_button.setEnabled(True)
        self.explain_document_button.setEnabled(True)

    def explain_document(self):
        if self.document_text and not self.is_debating:
            self.is_explaining = True
            self.chat_area.append("<b>Sistema:</b> Explicando el documento...")
            # Inicia el procesamiento del primer fragmento
            self.process_next_chunk()
        else:
            self.status_label.setText("No se puede explicar el documento en este momento.")


    def generate_explanation(self, chunk):
        gpt_token_limit = self.gpt_slider.value()
        explanation = self.moe_llm.generate_response(
            f"Explica en español de forma clara y sencilla usando sólo palabras y letras que estén en el alfabeto español el siguiente fragmento del documento: {chunk}",
            "gpt",
            max_tokens=gpt_token_limit
        )
        self.chat_area.append(f"<b style='color: #4CAF50;'>Profesor-GPT:</b> {explanation}")
        self.speak(explanation, "gpt")

        # Continua con el siguiente fragmento
        self.process_next_chunk()

    def start_debate(self):
        if self.is_explaining:
            self.explanation_thread.terminate()
            self.is_explaining = False
            self.chat_area.append("<b>Sistema:</b> Explicación interrumpida. Iniciando debate...")

        if not self.document_queue.empty():
            self.is_debating = True
            self.status_label.setText("Iniciando debate...")
            self.last_message = ""  # Reiniciamos last_message al iniciar el debate
            self.continue_debate(self.document_queue.get())
        else:
            self.status_label.setText("Por favor, cargue un documento primero.")

    def continue_debate(self, chunk):
        if not self.is_debating:
            return

        # Profesor-GPT comenta el fragmento
        self.moe_llm.load_model("gpt")
        gpt_comment = self.moe_llm.generate_response(
            f"Comenta el siguiente fragmento del documento en español y haz una pregunta a la invitada, Tanto los números, las fechas y las fórmulas matemáticas tienes que escribirlas en lenguaje natural en español: {chunk}\nÚltimo mensaje: {self.last_message}",
            "gpt",
            max_tokens=self.gpt_slider.value()
        )
        self.update_chat("Profesor-GPT", gpt_comment)
        self.last_message = gpt_comment

        # Limpiar caché después de generar la respuesta
        torch.cuda.empty_cache()

        # LLAMA responde
        self.moe_llm.load_model("llama")
        llama_response = self.moe_llm.generate_response(
            f"Responde a la pregunta y comenta sobre el fragmento, alterna diferentes tipos de respuesta unas veces cortas y concretas y otras veces más largas y detalladas como en un debate de radio en argentino: {gpt_comment}",
            "llama",
            max_tokens=self.llama_slider.value()
        )
        self.update_chat("Periodista LLAMA", llama_response)
        self.last_message = llama_response

        # Limpiar caché después de generar la respuesta
        torch.cuda.empty_cache()

        # Continuar con el siguiente fragmento si hay más
        if not self.document_queue.empty():
            QTimer.singleShot(100, lambda: self.continue_debate(self.document_queue.get()))
        else:
            self.is_debating = False
            self.status_label.setText("Debate finalizado.")
            self.start_debate_button.setEnabled(False)

    def update_chat(self, speaker, message):
        color = "#4CAF50" if speaker == "Profesor-GPT" else "#2196F3"
        self.chat_area.append(f"<b style='color: {color};'>{speaker}:</b> {message}")
        model_key = "gpt" if speaker == "Profesor-GPT" else "llama"
        self.speak(message, model_key)
        self.status_label.setText(f"{speaker} ha respondido. Continuando el debate...")

    def on_debate_finished(self):
        self.is_debating = False
        self.status_label.setText("Debate finalizado.")
        self.start_debate_button.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
