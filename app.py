from typing import List, Tuple, Optional

import google.generativeai as genai
import gradio as gr
from PIL import Image

TITLE = """<h1 align="center">Gemini Pro and Pro Vision via API 🚀</h1>"""
DUPLICATE = """
<div style="text-align: center; display: flex; justify-content: center; align-items: center;">
    <a href="https://huggingface.co/spaces/SkalskiP/ChatGemini?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space" style="margin-right: 10px;">
    </a>
    <span>Duplicate the Space and run securely with your GOOGLE API KEY.</span>
</div>
"""


def predict(
    google_key: str,
    text_prompt: str,
    image_prompt: Optional[Image.Image],
    chatbot: List[Tuple[str, str]]
) -> Tuple[str, List[Tuple[str, str]]]:
    if not google_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Please follow the instructions in the README to set it up.")

    genai.configure(api_key=google_key)

    if image_prompt is None:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(text_prompt, stream=True)
        response.resolve()
    else:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([text_prompt, image_prompt], stream=True)
        response.resolve()

    chatbot.append((text_prompt, response.text))
    return "", chatbot


google_key_component = gr.Textbox(
    label="GOOGLE API KEY",
    value="",
    type="password",
    placeholder="...",
    info="You have to provide your own GOOGLE_API_KEY for this app to function properly",
)

image_prompt_component = gr.Image(type="pil", label="Image", scale=1)
chatbot_component = gr.Chatbot(label='Gemini', scale=2)
text_prompt_component = gr.Textbox(
    placeholder="Hi there!",
    label="Ask me anything and press Enter"
)
run_button_component = gr.Button()

inputs = [
    google_key_component,
    text_prompt_component,
    image_prompt_component,
    chatbot_component
]

with gr.Blocks() as demo:
    gr.HTML(TITLE)
    gr.HTML(DUPLICATE)
    with gr.Column():
        google_key_component.render()
        with gr.Row():
            image_prompt_component.render()
            chatbot_component.render()
        text_prompt_component.render()
        run_button_component.render()

    run_button_component.click(
        fn=predict,
        inputs=inputs,
        outputs=[text_prompt_component, chatbot_component],
    )

    text_prompt_component.submit(
        fn=predict,
        inputs=inputs,
        outputs=[text_prompt_component, chatbot_component],
    )

demo.queue(max_size=99).launch(debug=True)
