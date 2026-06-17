"""
AGRO QA Chatbot - Main Entry Point
Run with: python app.py
"""
import uvicorn
from src.core.config import settings

if __name__ == "__main__":
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
