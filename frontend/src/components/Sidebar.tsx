import { useConversations } from '../api/hooks'
import { errorMessage } from '../api/client'
import { relativeTime } from '../format'

interface Props {
  selectedId: string | null
  onSelect: (id: string) => void
  onNew: () => void
}

export function Sidebar({ selectedId, onSelect, onNew }: Props) {
  const { data, isPending, isError, error, refetch } = useConversations()

  return (
    <nav className="sidebar" aria-label="Conversations">
      <div className="brand">
        Truth<span>Lens</span>
      </div>
      <button className="new-chat" onClick={onNew}>
        + New conversation
      </button>
      <div className="conversation-list">
        {isPending && <p className="muted">Loading conversations…</p>}
        {isError && (
          <div className="sidebar-error">
            <p>{errorMessage(error)}</p>
            <button onClick={() => refetch()}>Retry</button>
          </div>
        )}
        {data?.length === 0 && <p className="muted">No conversations yet.</p>}
        {data?.map((c) => (
          <button
            key={c.id}
            className={`conversation ${c.id === selectedId ? 'active' : ''}`}
            onClick={() => onSelect(c.id)}
            title={c.title}
          >
            <span className="title">{c.title}</span>
            <span className="meta">{relativeTime(c.updated_at)}</span>
          </button>
        ))}
      </div>
    </nav>
  )
}
