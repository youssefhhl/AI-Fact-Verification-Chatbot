import { useState } from 'react'
import { ChatView } from './components/ChatView'
import { Sidebar } from './components/Sidebar'

export default function App() {
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  // Changes only when the user navigates, so the chat view (and its send state) resets then,
  // but not when a new conversation gets its id from the backend.
  const [viewKey, setViewKey] = useState(0)

  function open(id: string | null) {
    setConversationId(id)
    setViewKey((k) => k + 1)
    setMenuOpen(false)
  }

  return (
    <div className={`layout ${menuOpen ? 'menu-open' : ''}`}>
      <button className="menu-toggle" onClick={() => setMenuOpen((v) => !v)} aria-label="Toggle conversations">
        ☰
      </button>
      <Sidebar selectedId={conversationId} onSelect={open} onNew={() => open(null)} />
      <div className="backdrop" onClick={() => setMenuOpen(false)} />
      <ChatView key={viewKey} conversationId={conversationId} onConversationChange={setConversationId} />
    </div>
  )
}
