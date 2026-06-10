import { useState, useEffect, useRef, useCallback } from 'react'
import { Upload, FileText, Trash2, Send, Bot, User, Loader2, Database, Zap, Menu, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API_BASE = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState(null)
  const [docs, setDocs] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const dropRef = useRef(null)

  useEffect(() => { checkHealth(); fetchDocs() }, [])
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`)
      if (res.ok) setHealth(await res.json())
    } catch { setHealth({ status: 'unhealthy', vector_store_documents: 0 }) }
  }, [])

  const fetchDocs = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`)
      if (res.ok) setDocs((await res.json()).documents || [])
    } catch {}
  }, [])

  const clearDocs = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`, { method: 'DELETE' })
      if (res.ok) {
        setDocs([])
        setMessages([])
        addMessage('system', 'All documents cleared')
      }
    } catch {}
  }

  const handleFiles = async (fileList) => {
    const files = Array.from(fileList).filter(f => f.name.endsWith('.pdf'))
    if (!files.length) return

    setUploading(true)
    setUploadProgress(0)
    const formData = new FormData()
    files.forEach(f => formData.append('files', f))

    try {
      const xhr = new XMLHttpRequest()
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setUploadProgress(Math.round((e.loaded / e.total) * 100))
      }
      
      const result = await new Promise((resolve, reject) => {
        xhr.open('POST', `${API_BASE}/ingest`)
        xhr.onload = () => {
          try { resolve(JSON.parse(xhr.responseText)) } catch { reject(xhr.responseText) }
        }
        xhr.onerror = reject
        xhr.send(formData)
      })

      addMessage('system', `Ingested ${files.map(f => f.name).join(', ')} — ${result.chunk_count} chunks`)
      await checkHealth()
      await fetchDocs()
    } catch (e) {
      addMessage('error', `Upload failed: ${e?.detail || e}`)
    } finally {
      setUploading(false)
      setUploadProgress(0)
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const query = input.trim()
    setInput('')
    setLoading(true)
    addMessage('user', query)
    addMessage('assistant', '...loading...')

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 })
      })
      const data = await res.json()
      
      setMessages(prev => prev.map((m, i) => 
        i === prev.length - 1 ? { ...m, content: data.response, role: 'assistant', time: new Date().toLocaleTimeString() } : m
      ))
    } catch (e) {
      setMessages(prev => prev.map((m, i) => 
        i === prev.length - 1 ? { ...m, content: 'Connection error. Is the server running?', role: 'error' } : m
      ))
    } finally {
      setLoading(false)
    }
  }

  const addMessage = (role, content) => {
    setMessages(prev => [...prev, { role, content, time: new Date().toLocaleTimeString() }])
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    dropRef.current?.classList.add('drag-over')
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    dropRef.current?.classList.remove('drag-over')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    dropRef.current?.classList.remove('drag-over')
    handleFiles(e.dataTransfer.files)
  }

  const hasDocs = docs.length > 0 && health?.vector_store_documents > 0

  return (
    <div className="app">
      <button className="mobile-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2><Zap size={20} /> RAG Chat</h2>
          <p className="subtitle">Legal document intelligence</p>
        </div>

        <div className="upload-area">
          <label 
            ref={dropRef}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="upload-icon">
              {uploading ? <Loader2 size={24} className="spin" /> : <Upload size={24} />}
            </div>
            <div className="upload-text">
              {uploading ? 'Processing...' : 'Upload PDF files'}
            </div>
            <div className="upload-sub">Drag & drop or click to browse</div>
            <input
              type="file"
              accept=".pdf"
              multiple
              onChange={(e) => handleFiles(e.target.files)}
              disabled={uploading}
            />
          </label>
          {uploading && (
            <div className="upload-progress">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
              </div>
              <div className="progress-text">{uploadProgress}%</div>
            </div>
          )}
        </div>

        <div className="docs-list">
          <div className="section-title">
            <Database size={12} /> Documents ({docs.length})
          </div>
          {docs.length === 0 ? (
            <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
              No documents uploaded yet
            </div>
          ) : (
            docs.map((doc, i) => (
              <div key={i} className="doc-item active">
                <div className="doc-icon"><FileText size={14} /></div>
                <div className="doc-info">
                  <div className="doc-name">{doc.name}</div>
                  <div className="doc-meta">{doc.count} chunks</div>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          {docs.length > 0 && (
            <button className="clear-btn" onClick={clearDocs} style={{ display: 'flex', alignItems: 'center', gap: '6px', width: '100%', justifyContent: 'center', marginBottom: '8px' }}>
              <Trash2 size={14} /> Clear all documents
            </button>
          )}
          <div className="status-badge">
            <div className={`status-dot ${health?.status === 'healthy' ? 'connected' : health?.status ? 'disconnected' : 'connecting'}`} />
            <span>
              {health?.status === 'healthy' 
                ? `Connected • ${health.vector_store_documents} chunks` 
                : health?.status 
                  ? 'Disconnected' 
                  : 'Connecting...'}
            </span>
          </div>
        </div>
      </aside>

      {sidebarOpen && <div className="mobile-backdrop" onClick={() => setSidebarOpen(false)} />}

      <main className="main-area" onClick={() => sidebarOpen && setSidebarOpen(false)}>
        {hasDocs ? (
          <>
            <div className="chat-header">
              <div style={{ fontSize: '14px', fontWeight: 500 }}>Chat with your documents</div>
              <div className="model-badge">{health?.vector_store_documents || 0} chunks indexed</div>
            </div>

            <div className="messages">
              {messages.map((msg, i) => (
                <div key={i} className={`msg ${msg.role} ${msg.role === 'assistant' && msg.content === '...loading...' ? 'loading' : ''}`}>
                  <div className="avatar">
                    {msg.role === 'user' ? <User size={16} /> : 
                     msg.role === 'assistant' ? <Bot size={16} /> :
                     msg.role === 'system' ? <Database size={14} /> :
                     <X size={14} />}
                  </div>
                  <div className="msg-body">
                    <div className="msg-role">
                      {msg.role === 'user' ? 'You' : 
                       msg.role === 'assistant' ? 'Assistant' :
                       msg.role === 'system' ? 'System' : 'Error'}
                    </div>
                    <div className="msg-content">
                      {msg.role === 'assistant' && msg.content === '...loading...' ? (
                        <div className="typing-dots"><span /><span /><span /></div>
                      ) : msg.role === 'assistant' ? (
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      ) : (
                        msg.content
                      )}
                    </div>
                    <div className="msg-time">{msg.time}</div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
              <div className="input-wrapper">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Ask a question about your documents..."
                  disabled={loading}
                  autoFocus
                />
                <button className="send-btn" onClick={sendMessage} disabled={!input.trim() || loading}>
                  <Send size={16} />
                </button>
              </div>
              <div className="input-hint">
                Press Enter to send — answers are based on your uploaded documents
              </div>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <div className="empty-icon"><FileText size={36} /></div>
            <h2>Upload documents to get started</h2>
            <p>
              Drag and drop PDF files into the sidebar to begin.
              Once uploaded, you can ask questions about your documents
              and get AI-powered answers backed by the source material.
            </p>
          </div>
        )}
      </main>
    </div>
  )
}

export default App