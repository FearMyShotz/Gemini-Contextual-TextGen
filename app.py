from typing import List, Tuple, Optional

import google.generativeai as genai
import gradio as gr
from PIL import Image

TITLE = """<h1 align="center">Gemini Playground 💬</h1>"""
SUBTITLE = """<h2 align="center">Play with Gemini Pro and Gemini Pro Vision API</h2>"""
DUPLICATE = """
<div style="text-align: center; display: flex; justify-content: center; align-items: center;">
    <a href="https://huggingface.co/spaces/SkalskiP/ChatGemini?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space" style="margin-right: 10px;">
    </a>
    <span>Duplicate the Space and run securely with your 
        <a href="https://makersuite.google.com/app/apikey">GOOGLE API KEY</a>.
    </span>
</div>
"""

print("google-generativeai:", genai.__version__)


def preprocess_stop_sequences(stop_sequences: str) -> Optional[List[str]]:
    if not stop_sequences:
        return None
    return [sequence.strip() for sequence in stop_sequences.split(",")]


def predict(
    google_key: str,
    text_prompt: str,
    image_prompt: Optional[Image.Image],
    temperature: float,
    max_output_tokens: int,
    stop_sequences: str,
    top_k: int,
    top_p: float,
    chatbot: List[Tuple[str, str]]
) -> Tuple[str, List[Tuple[str, str]]]:
    if not google_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Please follow the instructions in the README to set it up.")

    genai.configure(api_key=google_key)
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        stop_sequences=preprocess_stop_sequences(stop_sequences=stop_sequences),
        top_k=top_k,
        top_p=top_p)

    if image_prompt is None:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            text_prompt,
            stream=True,
            generation_config=generation_config)
        response.resolve()
    else:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(
            [text_prompt, image_prompt],
            stream=True,
            generation_config=generation_config)
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
temperature_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.4,
    step=0.05,
    label="Temperature",
    info=(
        "Temperature controls the degree of randomness in token selection. Lower "
        "temperatures are good for prompts that expect a true or correct response, "
        "while higher temperatures can lead to more diverse or unexpected results. "
    ))
max_output_tokens_component = gr.Slider(
    minimum=1,
    maximum=2048,
    value=1024,
    step=1,
    label="Token limit",
    info=(
        "Token limit determines the maximum amount of text output from one prompt. A "
        "token is approximately four characters. The default value is 2048."
    ))
stop_sequences_component = gr.Textbox(
    label="Add stop sequence",
    value="",
    type="text",
    placeholder="STOP, END",
    info=(
        "A stop sequence is a series of characters (including spaces) that stops "
        "response generation if the model encounters it. The sequence is not included "
        "as part of the response. You can add up to five stop sequences."
    ))
top_k_component = gr.Slider(
    minimum=1,
    maximum=40,
    value=32,
    step=1,
    label="Top-K",
    info=(
        "Top-k changes how the model selects tokens for output. A top-k of 1 means the "
        "selected token is the most probable among all tokens in the model’s "
        "vocabulary (also called greedy decoding), while a top-k of 3 means that the "
        "next token is selected from among the 3 most probable tokens (using "
        "temperature)."
    ))
top_p_component = gr.Slider(
    minimum=0,
    maximum=1,
    value=1,
    step=0.01,
    label="Top-P",
    info=(
        "Top-p changes how the model selects tokens for output. Tokens are selected "
        "from most probable to least until the sum of their probabilities equals the "
        "top-p value. For example, if tokens A, B, and C have a probability of .3, .2, "
        "and .1 and the top-p value is .5, then the model will select either A or B as "
        "the next token (using temperature). "
    ))

inputs = [
    google_key_component,
    text_prompt_component,
    image_prompt_component,
    temperature_component,
    max_output_tokens_component,
    stop_sequences_component,
    top_k_component,
    top_p_component,
    chatbot_component
]

with gr.Blocks() as demo:
    gr.HTML(TITLE)
    gr.HTML(SUBTITLE)
    gr.HTML(DUPLICATE)
    with gr.Column():
        google_key_component.render()
        with gr.Row():
            image_prompt_component.render()
            chatbot_component.render()
        text_prompt_component.render()
        run_button_component.render()
        with gr.Accordion("Parameters", open=False):
            temperature_component.render()
            max_output_tokens_component.render()
            stop_sequences_component.render()
            with gr.Accordion("Advanced", open=False):
                top_k_component.render()
                top_p_component.render()

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
