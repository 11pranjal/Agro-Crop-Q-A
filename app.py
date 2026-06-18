"""
AGRO QA Chatbot - Main Entry Point
Run with: python app.py
"""
from src.core.config import settings


def main() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        print("ERROR: uvicorn is not installed in the current Python environment.")
        print("Install dependencies with: python -m pip install -r requirements.txt")
        raise exc

    print(f"\n🚀 Starting AGRO QA Chatbot Backend...")
    print(f"📡 Backend: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📚 API Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"🔌 Frontend URL: {settings.FRONTEND_URL}\n")

    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
