import React, { useState } from 'react'

export default function Chat() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [sessionId, setSessionId] = useState(null)

  async function send() {
    if (!query) return
    const form = new FormData()
    form.append('query', query)
    if (sessionId) form.append('session_id', sessionId)

    const res = await fetch('http://localhost:8000/chat', { method: 'POST', body: form })
    const data = await res.json()
    const resp = data.response

    setMessages((m) => [...m, { user: query, assistant: resp.response }])
    setSessionId(resp.session_id)
    setQuery('')
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>AGRO QA Chat</h2>
      <div style={{ marginBottom: 10 }}>
        <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={4} cols={80} />
      </div>
      <button onClick={send}>Send</button>

      <div style={{ marginTop: 20 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 10 }}>
            <b>Farmer:</b> {m.user}
            <br />
            <b>AGRO:</b> {m.assistant}
          </div>
        ))}
      </div>
    </div>
  )
}
