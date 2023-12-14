from typing import List, Tuple

import gradio as gr
import google.generativeai as genai


def predict(google_key: str, text_prompt: str, chatbot: List[Tuple[str, str]]):
    if not google_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Please follow the instructions in the README to set it up.")

    genai.configure(api_key=google_key)
    model = genai.GenerativeModel('models/gemini-pro')
    response = model.generate_content(text_prompt, stream=True)
    response.resolve()
    chatbot.append((text_prompt, response.text))
    return "", chatbot


google_key_component = gr.Textbox(
    label="GOOGLE API KEY",
    value="",
    type="password",
    placeholder="...",
    info="You have to provide your own GPT4 keys for this app to function properly",
)

chatbot_component = gr.Chatbot(label='Gemini')
text_prompt_component = gr.Textbox(
    placeholder="Hi there!",
    label="Type an input and press Enter"
)
run_button_component = gr.Button()

with gr.Blocks() as demo:
    with gr.Column():
        google_key_component.render()
        with gr.Row():
            chatbot_component.render()
        text_prompt_component.render()
        run_button_component.render()

    run_button_component.click(
        fn=predict,
        inputs=[google_key_component, text_prompt_component, chatbot_component],
        outputs=[text_prompt_component, chatbot_component],
    )

    text_prompt_component.submit(
        fn=predict,
        inputs=[google_key_component, text_prompt_component, chatbot_component],
        outputs=[text_prompt_component, chatbot_component],
    )

demo.queue(max_size=99).launch(debug=True)
