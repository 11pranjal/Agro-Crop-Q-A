import React, { useState, useRef, useEffect } from 'react'
import './Chat.css'

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  async function handleSend() {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const form = new FormData()
      form.append('query', userMessage)
      if (sessionId) form.append('session_id', sessionId)

      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) throw new Error('Failed to get response')

      const data = await res.json()
      const assistantResponse = data.response?.response || 'Sorry, I could not generate a response.'
      const newSessionId = data.response?.session_id

      setMessages((prev) => [...prev, { role: 'assistant', content: assistantResponse }])
      if (newSessionId) setSessionId(newSessionId)
    } catch (error) {
      console.error('Error:', error)
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: '❌ Error connecting to backend. Make sure the server is running on http://localhost:8000'
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>🌾 AGRO QA Chatbot</h1>
        <p>Ask me anything about agriculture</p>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <h2>👋 Welcome!</h2>
            <p>Upload a PDF with agricultural information, then ask me questions about it.</p>
            <p>Example: "What is wheat farming?" or "How to irrigate rice?"</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`chat-message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👨‍🌾' : '🤖'}
              </div>
              <div className="message-content">
                <p>{msg.content}</p>
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="chat-message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <p className="typing">Thinking...</p>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your question here... (Press Enter to send)"
          disabled={loading}
          rows={3}
        />
        <button onClick={handleSend} disabled={loading || !input.trim()}>
          {loading ? '⏳ Sending...' : '📤 Send'}
        </button>
      </div>
    </div>
  )
}
