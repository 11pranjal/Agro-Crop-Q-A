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
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
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

  async function handleFileUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const files = event.target.files
    if (!files || files.length === 0) return

    const file = files[0]
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus('❌ Please upload a PDF file')
      return
    }

    setUploading(true)
    setUploadStatus('⏳ Uploading...')

    try {
      const formData = new FormData()
      formData.append('file', file)
      if (sessionId) {
        formData.append('session_id', sessionId)
      }

      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        const errorMsg = errorData?.detail || errorData?.error || 'Upload failed'
        throw new Error(errorMsg)
      }

      const data = await res.json()
      const newSessionId = data.session_id
      if (newSessionId) {
        setSessionId(newSessionId)
      }
      setUploadStatus(`✅ Uploaded: ${file.name}`)
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: `📄 Document "${file.name}" uploaded successfully! You can now ask me questions about it.`
      }])
    } catch (error) {
      console.error('Upload error:', error)
      const errorMsg = error instanceof Error ? error.message : 'Unknown error'
      setUploadStatus(`❌ Upload failed: ${errorMsg}`)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>🌾 AGRO QA Chatbot</h1>
        <p>Ask me anything about agriculture</p>
        <p className="chat-note">Using OpenAI with uploaded PDF content only. No web search is performed.</p>
      </div>

      <div className="upload-section">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileUpload}
          disabled={uploading}
          style={{ display: 'none' }}
        />
        <button
          className="upload-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? '⏳ Uploading...' : '📄 Upload PDF'}
        </button>
        {uploadStatus && (
          <p className="upload-status">{uploadStatus}</p>
        )}
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
