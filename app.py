import os
import google.generativeai as genai
import gradio as gr

# API-Key aus Umgebungsvariablen laden oder direkt im Code festlegen
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Konfiguration der API
genai.configure(api_key=GOOGLE_API_KEY)

# Funktion zur Textgenerierung mit der Google GenAI API
def generate_text(prompt, model_name, temperature, stop_sequences, top_k, top_p):
    # Modell initialisieren
    model = genai.GenerativeModel(model_name)
    
    # Erstellen der GenerationConfig ohne max_output_tokens
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        stop_sequences=[seq.strip() for seq in stop_sequences.split(",") if seq.strip()],
        top_k=top_k,
        top_p=top_p
    )
    
    # Generierung der Antwort
    response = model.generate_content(
        contents=[prompt],  # Hier wird das 'contents'-Argument korrekt übergeben
        generation_config=generation_config
    )
    
    # Kombinieren aller Textblöcke in der Antwort
    generated_text = ''.join([chunk.text for chunk in response])
    
    return generated_text

# Titel und Beschreibung für die Gradio-App
title = "Text Generation with Google GenAI"
description = "Generate text using Google GenAI API with customizable models and parameters."

# Gradio-Interface definieren
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt here", placeholder="Type something..."),
        gr.Dropdown(label="Select Model", choices=["gemini-1.5-flash", "gemini-pro", "gemini-pro-vision"], value="gemini-1.5-flash"),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.4),
        gr.Textbox(label="Stop Sequences (comma-separated)", placeholder="Enter stop sequences..."),
        gr.Slider(label="Top-K", minimum=1, maximum=40, step=1, value=32),
        gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, value=0.9),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title=title,
    description=description,
)

# Starten der App
iface.launch()
