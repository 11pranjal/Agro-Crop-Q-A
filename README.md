# Agro QA Chatbot

Simple Flask app that lets farmers upload PDFs and ask natural language questions.

This app supports two modes:
- **Free local mode**: uses TF-IDF retrieval and simple local summarization, no API key required.
- **OpenAI mode**: if you set `OPENAI_API_KEY`, the app uses OpenAI to generate a higher-quality answer.

Setup

1. Create and activate a Python virtualenv.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root if you want OpenAI features:

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
HOST=127.0.0.1
PORT=7860
FLASK_DEBUG=1
```

Leave `OPENAI_API_KEY` blank to use the free local mode. If you want a higher-quality answer later, install the OpenAI package and add a valid key.

Run

```bash
python app.py
```

Open in browser:

```bash
http://127.0.0.1:7860
```

How it works

- Upload a PDF on the homepage.
- Ask a question.
- If no API key is provided, the app will use local TF-IDF retrieval and summarization.
- If `OPENAI_API_KEY` is set, the app will use OpenAI to generate the final answer.

Note

OpenAI is optional. You can run the application without paying for OpenAI by leaving `OPENAI_API_KEY` empty.
# Agro-Crop-Q-A
