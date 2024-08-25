---
title: Gemini-Contextual-TextGen
emoji: 💬
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false
---

# Gemini-Contextual-TextGen

**Gemini-Contextual-TextGen** is a project that leverages the Google AI Python SDK for the Gemini API (see [generative-ai-python](https://github.com/google-gemini/generative-ai-python)) to generate contextual text responses. It maintains conversation history and provides a simple API interface using FastAPI and Gradio for seamless integration with Hugging Face Spaces.

## Features

- **Contextual Text Generation**: The system preserves the chat history to maintain context across multiple interactions, leading to more coherent and relevant responses.
- **Gradio Interface**: Provides a user-friendly web interface for testing and interacting with the model.
- **Hugging Face Integration**: Easily deployable on Hugging Face Spaces, making it accessible and shareable.

### Planned Features
- Support for multimodal files, including images, videos, and audio files.
- Transition to using [Docker](https://www.docker.com/) and [FastAPI](https://fastapi.tiangolo.com/) for improved deployment and scalability, replacing Gradio.

## Setup

### Prerequisites

- Python 3.8+
- Pip (Python package manager)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/FearMyShotz/Gemini-Contextual-TextGen.git
   cd Gemini-Contextual-TextGen
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for the Google Gemini API key and other configurations as needed.

### Usage

1. **Run the API**:

   ```bash
   python app.py
   ```

   The API will be accessible locally, and you can interact with it using HTTP POST requests or via the Gradio interface.

2. **Test the Gradio Interface**:

   After running the API, navigate to `http://0.0.0.0:7860` in your browser to access the Gradio interface.

### Deployment on Hugging Face Spaces

1. Fork this repository to your Hugging Face account.
2. Deploy the project as a Space on Hugging Face by following their [documentation](https://huggingface.co/docs/hub/spaces).

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

### Acknowledgments

- [Google Generative AI](https://cloud.google.com/generative-ai) for the Gemini model.
- [Hugging Face](https://huggingface.co/) for providing the platform to share and deploy the model.
