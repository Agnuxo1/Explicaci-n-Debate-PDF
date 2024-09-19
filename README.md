# Explicación-Debate-PDF-VOZ
Programa en local para subir documentos PDF o TXT en cualquier idioma, con Opción de consuta, explicación y Debate sobre el documento entre dos LLM GPT y Llama, con salida de voz en Español.

## 🖥️ Interfaz de Usuario

La interfaz gráfica incluye:

- 💬 Área de chat para mostrar el debate
- 🎛️ Controles de velocidad de voz
- 🔢 Ajustes de límite de tokens
- 📂 Botón para cargar documentos
- ▶️ Botones para iniciar explicación o debate


## 🧠 Modelos de IA

El programa utiliza dos modelos de lenguaje:

- 🧑‍🏫 **Profesor-GPT**: Basado en "GPT"
- 👩‍🎤 **Periodista LLAMA**: Basado en "Meta-Llama-3.1-8B"


## 🎙️ Síntesis de Voz

Se utilizan dos modelos de TTS:

- 🇪🇸 Modelo español para el Profesor-GPT
- 🇦🇷 Modelo con acento argentino para la Periodista LLAMA


## 📝 Notas Adicionales

- El programa convierte números y fórmulas a texto para mejorar la síntesis de voz.
- Se incluyen controles para ajustar la velocidad de reproducción de voz.
- La interfaz permite una fácil navegación y control del proceso de lectura y debate.


---

Desarrollado por [Francisco Angulo de Lafuente]

# 📚 Lector y Debatidor de Documentos

![Banner](https://hebbkx1anhila5yf.public.blob.vercel-storage.com/placeholder.svg?height=200&width=800)

## 🌟 Características Principales

- 📖 Lectura de documentos PDF y TXT
- 🗣️ Explicación del contenido en español
- 🎭 Debate simulado entre dos IA
- 🔊 Síntesis de voz para las respuestas
- 🖥️ Interfaz gráfica moderna y fácil de usar

## 🛠️ Tecnologías Utilizadas

- Python
- PyQt5 para la interfaz gráfica
- Transformers para modelos de lenguaje
- TTS para síntesis de voz
- PyPDF2 para lectura de PDFs

## 🚀 Cómo Funciona

El programa ofrece una experiencia interactiva de lectura y debate de documentos:

1️⃣ **Carga de Documentos**: Sube fácilmente archivos PDF o TXT.

2️⃣ **Explicación del Contenido**: Una IA explica el documento de forma clara y concisa.

3️⃣ **Debate Simulado**: Dos IA, un "Profesor-GPT" y una "Periodista LLAMA", debaten sobre el contenido.

4️⃣ **Síntesis de Voz**: Las respuestas se convierten en audio para una experiencia más inmersiva.

5️⃣ **Interacción del Usuario**: Participa en el debate añadiendo tus propios comentarios o preguntas.

## 📊 Diagrama de Flujo

```mermaid title="Flujo del Programa" type="diagram"
graph TD
    A[Inicio] --> B[Cargar Documento]
    B --> C{Tipo de Acción}
    C -->|Explicar| D[Generar Explicación]
    C -->|Debatir| E[Iniciar Debate]
    D --> F[Sintetizar Voz]
    E --> F
    F --> G[Mostrar en Interfaz]
    G --> H{Continuar?}
    H -->|Sí| C
    H -->|No| I[Fin]
