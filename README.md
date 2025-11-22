
# Gemini Voice-Native Assistant

A voice-enabled AI assistant powered by Google Gemini models that accepts text or voice input, generates natural language responses, and creates personalized workout plans. The assistant uses dynamic web search grounding to provide up-to-date information and converts text replies to speech for a seamless conversational experience.

---

## Features

- **Multimodal input**: Accepts text and audio (voice) inputs.
- **Dynamic grounding**: Uses Google Search Retrieval for real-time information.
- **Structured workout plans**: Generates and displays exercise plans in a markdown table.
- **Text-to-speech**: Converts responses to natural-sounding audio using Microsoft Neural Voices (`edge-tts`).
- **Interactive UI**: Built with Gradio, supports chat history and audio playback.

---

## Installation

Run the following to install required dependencies:

```bash
pip install -qU google-genai>=1.0.0 python-telegram-bot nest_asyncio gradio edge-tts pydantic ffmpeg-python
```

---

## Setup

1. **API Key**: Obtain a Google API key with access to Gemini models.

2. **Authentication**: Set your API key in the code or in Colab secrets:

```python
client = genai.Client(api_key='YOUR_API_KEY')
```

3. **Async Patch**: `nest_asyncio` is applied to support async operations in environments like Colab.

---

## Usage

1. Run the assistant script.

2. Use the Gradio web interface to:

   - Enter text queries or
   - Record voice messages via the microphone input.

3. The assistant responds with:

   - Natural language reply,
   - A structured workout plan (if requested),
   - And an audio playback of the response.

4. Clear conversation history anytime with the "Clear" button.

---

## Code Structure

- **`GeminiAssistant`**:  
  Core class that handles:

  - Model selection and grounding setup.
  - Conversation history management.
  - Processing multimodal user requests.
  - Formatting workout plans into markdown tables.

- **Text-to-Speech**:  
  Asynchronously generates speech audio files using Microsoft Neural Voices.

- **Gradio UI**:  
  Provides a friendly web interface with:

  - Chatbot history.
  - Text input box.
  - Voice recording input.
  - Audio output player.

---

## Dependencies

- [google-genai](https://pypi.org/project/google-genai/) — Gemini SDK  
- [python-telegram-bot](https://pypi.org/project/python-telegram-bot/)  
- [nest_asyncio](https://pypi.org/project/nest-asyncio/)  
- [gradio](https://gradio.app/)  
- [edge-tts](https://pypi.org/project/edge-tts/)  
- [pydantic](https://pydantic.dev/)  
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)

---

## Notes

- The assistant leverages conversation history for context-aware replies.
- Dynamic web grounding enables access to fresh information beyond training data.
- Workout plans are returned as structured JSON and rendered as markdown tables in the UI.
- Asynchronous TTS processing ensures smooth user experience without blocking.

---

## License

MIT License

---

## Acknowledgements

- Google Gemini models and Unified SDK  
- Microsoft Neural Voices (Edge TTS)  
- Gradio UI framework

---

Feel free to open issues or submit pull requests for improvements!
