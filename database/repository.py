"""Database repository for CRUD operations"""
from datetime import datetime
from sqlalchemy.orm import Session as DBSession
from typing import List, Optional
import uuid

from .models import Conversation, Session, Document


class ConversationRepository:
    """Repository for conversation operations"""
    
    def __init__(self, db: DBSession):
        self.db = db
    
    def create(self, session_id: str, user_message: str, assistant_response: str,
               pdf_filename: Optional[str] = None, is_crisis: bool = False,
               confidence_score: Optional[float] = None) -> Conversation:
        """Create new conversation record"""
        conversation = Conversation(
            id=str(uuid.uuid4()),
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            pdf_filename=pdf_filename,
            is_crisis=is_crisis,
            confidence_score=confidence_score
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
    
    def get_by_session(self, session_id: str, limit: int = 20) -> List[Conversation]:
        """Get conversations for a session"""
        return self.db.query(Conversation)\
            .filter(Conversation.session_id == session_id)\
            .order_by(Conversation.created_at.desc())\
            .limit(limit)\
            .all()
    
    def get_all_by_session(self, session_id: str) -> List[Conversation]:
        """Get all conversations for a session"""
        return self.db.query(Conversation)\
            .filter(Conversation.session_id == session_id)\
            .order_by(Conversation.created_at)\
            .all()


class SessionRepository:
    """Repository for session operations"""
    
    def __init__(self, db: DBSession):
        self.db = db
    
    def create(self, session_id: str) -> Session:
        """Create new session"""
        session = Session(session_id=session_id)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def get(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self.db.query(Session)\
            .filter(Session.session_id == session_id)\
            .first()
    
    def update_last_accessed(self, session_id: str):
        """Update last accessed time"""
        sess = self.get(session_id)
        if sess:
            sess.last_accessed = datetime.utcnow()
            sess.message_count += 1
            self.db.commit()


class DocumentRepository:
    """Repository for document operations"""
    
    def __init__(self, db: DBSession):
        self.db = db
    
    def create(self, session_id: str, filename: str, file_path: str,
               file_size: int, total_chunks: int) -> Document:
        """Create document record"""
        document = Document(
            doc_id=str(uuid.uuid4()),
            session_id=session_id,
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            total_chunks=total_chunks
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document
    
    def get_by_session(self, session_id: str) -> List[Document]:
        """Get all documents for a session"""
        return self.db.query(Document)\
            .filter(Document.session_id == session_id)\
            .order_by(Document.upload_date.desc())\
            .all()
