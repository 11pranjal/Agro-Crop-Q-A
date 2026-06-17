"""Memory management system for conversation history"""
from datetime import datetime, timedelta
from typing import Optional, List
import uuid
import json


class SessionMemory:
    """In-memory session management"""
    
    def __init__(self, max_history: int = 20, timeout_seconds: int = 3600):
        self.max_history = max_history
        self.timeout_seconds = timeout_seconds
        self.sessions = {}  # session_id -> {messages: [...], created_at, last_accessed}
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create new session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only recent messages
        if len(session["messages"]) > self.max_history:
            session["messages"] = session["messages"][-self.max_history:]
        
        session["last_accessed"] = datetime.utcnow()
    
    def get_history(self, session_id: str) -> list:
        """Get conversation history for session"""
        if session_id not in self.sessions:
            return []
        
        return self.sessions[session_id]["messages"]
    
    def get_context(self, session_id: str, limit: int = 5) -> str:
        """Get conversation context as string"""
        if session_id not in self.sessions:
            return ""
        
        messages = self.sessions[session_id]["messages"][-limit:]
        context = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        return context
    
    def clear_session(self, session_id: str):
        """Clear session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.utcnow()
        timeout = timedelta(seconds=self.timeout_seconds)
        
        expired = [
            sid for sid, sess in self.sessions.items()
            if now - sess["last_accessed"] > timeout
        ]
        
        for sid in expired:
            del self.sessions[sid]
        
        return len(expired)


class PersistentMemory:
    """Persistent memory using database (to be integrated with SQLAlchemy)"""
    
    def __init__(self, db_session=None):
        self.db_session = db_session
    
    def save_conversation(self, session_id: str, user_message: str, 
                         assistant_response: str, pdf_filename: Optional[str] = None):
        """Save conversation to database"""
        from database.models import Conversation
        
        if not self.db_session:
            return False
        
        try:
            conversation = Conversation(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                pdf_filename=pdf_filename
            )
            
            self.db_session.add(conversation)
            self.db_session.commit()
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> list:
        """Get conversation history from database"""
        from database.models import Conversation
        
        if not self.db_session:
            return []
        
        try:
            conversations = self.db_session.query(Conversation)\
                .filter(Conversation.session_id == session_id)\
                .order_by(Conversation.created_at.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    "user": c.user_message,
                    "assistant": c.assistant_response,
                    "timestamp": c.created_at.isoformat()
                }
                for c in conversations
            ]
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []


class HybridMemory:
    """Hybrid memory system combining session and persistent memory"""
    
    def __init__(self, db_session=None, max_history: int = 20):
        self.session_memory = SessionMemory(max_history=max_history)
        self.persistent_memory = PersistentMemory(db_session)
    
    def add_conversation(self, session_id: str, user_message: str, 
                        assistant_response: str, pdf_filename: Optional[str] = None):
        """Add conversation to both memories"""
        # Add to session memory
        self.session_memory.add_message(session_id, "user", user_message)
        self.session_memory.add_message(session_id, "assistant", assistant_response)
        
        # Save to persistent memory
        self.persistent_memory.save_conversation(
            session_id, user_message, assistant_response, pdf_filename
        )
    
    def get_context(self, session_id: str) -> str:
        """Get conversation context"""
        return self.session_memory.get_context(session_id)
