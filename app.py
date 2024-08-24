import os
import google.generativeai as genai
import gradio as gr

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model_name = "gemini-1.5-flash"
temperature = 1.0
stop_sequences = ""
top_k = 64
top_p = 0.95

def predict(text):
    model = genai.GenerativeModel(model_name)
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        stop_sequences=[seq.strip() for seq in stop_sequences.split(",") if seq.strip()],
        top_k=top_k,
        top_p=top_p
    )
    try:
        response = model.generate_content(
            contents=[text],
            generation_config=generation_config
        )
        generated_text = ''.join([chunk.text for chunk in response if hasattr(chunk, 'text')])
    except Exception as e:
        return f"An error occurred: {str(e)}"
    return generated_text

title = "Text Generation with Google GenAI"
description = "Generate text using Google GenAI API with customizable models and parameters."

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter your prompt here", placeholder="Type something..."),
    outputs='text',
    title=title,
    description=description,
)

iface.launch()