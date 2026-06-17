"""Database models for conversation history and sessions"""
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from src.core.config import settings

Base = declarative_base()


class Conversation(Base):
    """Model for storing conversation history"""
    __tablename__ = "conversations"
    
    id = Column(String(50), primary_key=True, index=True)
    session_id = Column(String(50), index=True)
    user_message = Column(Text)
    assistant_response = Column(Text)
    pdf_filename = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_crisis = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, session_id={self.session_id})>"


class Session(Base):
    """Model for managing user sessions"""
    __tablename__ = "sessions"
    
    session_id = Column(String(50), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Session(session_id={self.session_id})>"


class Document(Base):
    """Model for tracking uploaded documents"""
    __tablename__ = "documents"
    
    doc_id = Column(String(50), primary_key=True, index=True)
    session_id = Column(String(50), index=True)
    filename = Column(String(255))
    file_path = Column(String(255))
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)  # in bytes
    total_chunks = Column(Integer)
    
    def __repr__(self):
        return f"<Document(doc_id={self.doc_id}, filename={self.filename})>"


# Database setup
engine = create_engine(
    str(settings.DATABASE_URL),
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")
