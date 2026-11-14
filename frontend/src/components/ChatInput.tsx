import { type FormEvent, type KeyboardEvent, useState } from 'react'

const MAX = 4000 // backend limit (ChatRequest.message)

export function ChatInput({ disabled, onSend }: { disabled: boolean; onSend: (text: string) => void }) {
  const [text, setText] = useState('')
  const trimmed = text.trim()

  function submit(event?: FormEvent) {
    event?.preventDefault()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) submit(event)
  }

  return (
    <form className="chat-input" onSubmit={submit}>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Enter a claim to verify, or ask a follow-up…"
        rows={1}
        maxLength={MAX}
        aria-label="Message"
      />
      <button type="submit" disabled={disabled || !trimmed}>
        {disabled ? <span className="spinner" aria-label="Sending" /> : 'Send'}
      </button>
    </form>
  )
}
