import { useEffect, useRef } from 'react'
import { useConversation, useSendMessage } from '../api/hooks'
import { errorConversationId, errorMessage } from '../api/client'
import { ChatInput } from './ChatInput'
import { MessageBubble, PendingBubbles } from './MessageBubble'

const EXAMPLES = [
  'Vitamin D supplements prevent bone fractures',
  'Kubernetes is written in Go',
  'The Great Wall of China is visible from space',
]

interface Props {
  conversationId: string | null
  onConversationChange: (id: string) => void
}

export function ChatView({ conversationId, onConversationChange }: Props) {
  const conversation = useConversation(conversationId)
  const send = useSendMessage()
  const bottom = useRef<HTMLDivElement>(null)

  const messages = conversation.data?.messages ?? []
  const pendingText = send.isPending ? send.variables?.message : undefined

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, pendingText])

  function handleSend(message: string) {
    send.mutate(
      { message, ...(conversationId ? { conversation_id: conversationId } : {}) },
      {
        onSuccess: (data) => onConversationChange(data.conversation_id),
        onError: (error) => {
          // The agent may fail after the user message was stored: show that conversation.
          const id = errorConversationId(error)
          if (id) onConversationChange(id)
        },
      },
    )
  }

  const empty = conversationId === null && !pendingText

  return (
    <main className="chat">
      <div className="messages">
        {empty && (
          <div className="welcome">
            <h1>What would you like to fact-check?</h1>
            <p>
              TruthLens checks claims against Wikipedia and scientific datasets, scientific literature and the web,
              then explains its verdict with cited evidence. Ask follow-up questions in the same conversation.
            </p>
            <div className="examples">
              {EXAMPLES.map((e) => (
                <button key={e} onClick={() => handleSend(e)} disabled={send.isPending}>
                  {e}
                </button>
              ))}
            </div>
          </div>
        )}
        {conversationId && conversation.isPending && <p className="muted center">Loading conversation…</p>}
        {conversation.isError && (
          <div className="error-banner" role="alert">
            {errorMessage(conversation.error)}
            <button onClick={() => conversation.refetch()}>Retry</button>
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {pendingText && <PendingBubbles text={pendingText} />}
        {send.isError && (
          <div className="error-banner" role="alert">
            {errorMessage(send.error)}
            <button onClick={() => send.variables && handleSend(send.variables.message)}>Retry</button>
          </div>
        )}
        <div ref={bottom} />
      </div>
      <ChatInput disabled={send.isPending} onSend={handleSend} />
    </main>
  )
}
