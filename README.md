# 🎙️ Lab 3 — ASR + LLM + TTS Pipeline

> **Curso:** Deep Learning  
> **Laboratorio:** 3 — Aplicación con ASR, LLM y Voice Cloning

## 📋 Descripción

Pipeline end-to-end que convierte una pregunta hablada en una respuesta de audio sintetizada:

1. **ASR** (Automatic Speech Recognition) — Transcribe audio a texto
2. **LLM** (Large Language Model) — Genera una respuesta inteligente
3. **TTS** (Text-to-Speech) — Sintetiza la respuesta en audio

---

## 🏗️ Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ASR → LLM → TTS                         │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │  🎤 ASR  │         │  🤖 LLM  │         │  🔊 TTS  │
    │ (Whisper)│────────▶│(FLAN-T5) │────────▶│(xTTS v2) │
    └──────────┘         └──────────┘         └──────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │  Audio   │         │  Texto   │         │  Audio   │
    │ Entrada  │         │Respuesta │         │  Salida  │
    │(pregunta)│         │  (1-3    │         │(respuesta│
    │          │         │oraciones)│         │   voz)   │
    └──────────┘         └──────────┘         └──────────┘
```

### Flujo de Datos

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   ENTRADA   │    │    ASR      │    │    LLM      │    │    TTS      │
│             │    │             │    │             │    │             │
│ 🎙️ Audio    │───▶│ 📝 Texto    │───▶│ 💬 Respuesta│───▶│ 🔊 Audio    │
│ (pregunta)  │    │ transcrito  │    │   generada  │    │ (respuesta) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                  │
                    ⏱️ asr_time        ⏱️ llm_time        ⏱️ tts_time
```

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Descripción |
|------------|------------|-------------|
| **ASR** | [OpenAI Whisper](https://github.com/openai/whisper) | Modelo `turbo` para transcripción |
| **LLM** | [FLAN-T5-Base](https://huggingface.co/google/flan-t5-base) | Generación de texto local |
| **TTS** | [Coqui TTS xTTS v2](https://github.com/coqui-ai/TTS) | Síntesis de voz con clonación |

---

## 📁 Estructura del Proyecto

```
DL_Laborario03/
├── 📓 Lab3_ASR_LLM_TTS.ipynb   # Notebook principal (Google Colab)
├── 📄 Laboratorio 3.pdf         # Enunciado del laboratorio
├── 📖 README.md                 # Este archivo
└── 🚫 .gitignore                # Archivos ignorados por Git
```

---

## 🚀 Uso Rápido

### 1. Abrir en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulrojasj/DL_Laborario03/blob/main/Lab3_ASR_LLM_TTS.ipynb)

### 2. Ejecutar las celdas en orden

```
Celda 1: Instalación de dependencias
Celda 2: Función de grabación de audio
Celda 3: Grabar pregunta
Celda 4: ASR — Transcripción con Whisper
Celda 5: LLM — Generación de respuesta
Celda 6: TTS — Síntesis de audio
```

### 3. Dependencias

```bash
pip install openai-whisper jiwer transformers accelerate sentencepiece coqui-tts
```

---

## 📊 Métricas de Rendimiento

El pipeline mide y reporta:

- ⏱️ **ASR Time**: Tiempo de transcripción
- ⏱️ **LLM Time**: Tiempo de generación de respuesta
- ⏱️ **TTS Time**: Tiempo de síntesis de audio
- ⏱️ **Total Time**: Tiempo end-to-end

---

## 📝 Notas

- Ejecutar en **Google Colab** con GPU habilitada para mejor rendimiento
- El modelo Whisper `turbo` requiere ~6GB de VRAM; usar `small` si hay limitaciones
- Audio de referencia para voice cloning debe ser de 5-15 segundos de duración

---

## 👥 Autor

Desarrollado como parte del curso de Deep Learning.
