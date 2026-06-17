from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import re
from dotenv import load_dotenv
from vectorstore import VectorStore
from pdf_utils import extract_text_from_pdf, chunk_text

load_dotenv()
app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

VS = VectorStore(storage_path=os.path.join(DATA_DIR, "vector_store"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    import sys
    print("DEBUG: Upload endpoint called", flush=True)
    sys.stdout.flush()
    
    if "file" not in request.files:
        return render_template("index.html", error="Missing file.")
    
    f = request.files["file"]
    if f.filename == "":
        return render_template("index.html", error="Empty filename.")

    try:
        print(f"DEBUG: Starting PDF extraction for {f.filename}", flush=True)
        sys.stdout.flush()
        
        text = extract_text_from_pdf(f.stream)
        print(f"DEBUG: Extraction complete, got {len(text)} characters", flush=True)
        sys.stdout.flush()
        
        if not text.strip():
            return render_template("index.html", error="PDF is empty or contains no extractable text.")
        
        print("DEBUG: Starting text chunking", flush=True)
        sys.stdout.flush()
        
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        print(f"DEBUG: Chunking complete, got {len(chunks)} chunks", flush=True)
        sys.stdout.flush()
        
        if not chunks:
            return render_template("index.html", error="Failed to create text chunks from PDF.")
        
        print(f"DEBUG: Adding {len(chunks)} documents to vector store", flush=True)
        sys.stdout.flush()
        
        VS.add_documents(chunks, source=f.filename)
        print("DEBUG: Documents added successfully", flush=True)
        sys.stdout.flush()
        
        return render_template("index.html", message=f"Uploaded {f.filename} with {len(chunks)} chunks.")
    except Exception as e:
        print(f"DEBUG: Error during upload: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return render_template("index.html", error=f"Upload failed: {str(e)}")


def summarize_snippets(snippets, question):
    if not snippets:
        return "I could not find any relevant document content."

    combined = " ".join(snippets[:3])
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    keywords = set(re.findall(r"\w+", question.lower()))
    selected = []
    for sentence in sentences:
        sentence_words = set(re.findall(r"\w+", sentence.lower()))
        if keywords & sentence_words:
            selected.append(sentence.strip())
            if len(selected) >= 3:
                break
    if not selected:
        selected = sentences[:2]
    return " ".join(selected).strip()


@app.route("/chat", methods=["POST"])
def chat():
    question = request.form.get("question", "").strip()
    if not question:
        return render_template("index.html", error="Please enter a question.")

    hits = VS.query(question, top_k=5)
    if not hits:
        return render_template("index.html", error="No relevant document sections found. Upload a PDF first and try again.")

    snippets = [h["text"] for h in hits]
    answer = None
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        try:
            import openai
        except ImportError:
            return render_template(
                "index.html",
                error="OpenAI package is not installed. Install it only if you want OpenAI answers.",
            )

        openai.api_key = api_key
        context_text = "\n---\n".join(snippets)
        prompt = (
            "You are an expert agricultural assistant. Given the following document snippets, "
            "answer the user's question concisely using only the provided context. If the "
            "context does not contain the answer, say you don't know and suggest next steps.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        )
        resp = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        answer = resp["choices"][0]["message"]["content"].strip()
        source_note = "Using OpenAI for the final answer."
    else:
        answer = summarize_snippets(snippets, question)
        source_note = "Using free local summarization; no OpenAI key required."

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        sources=[h["source"] for h in hits],
        source_note=source_note,
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 7860))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
