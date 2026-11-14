import type { Message } from '../api/types'
import { DECISION_LABEL } from '../format'
import { VerificationCard } from './VerificationCard'

export function MessageBubble({ message }: { message: Message }) {
  if (message.role === 'user') {
    return <div className="bubble bubble-user">{message.content}</div>
  }
  return (
    <div className="bubble bubble-assistant">
      {message.decision && <div className="decision">{DECISION_LABEL[message.decision]}</div>}
      <p className="answer">{message.content}</p>
      {message.verification && <VerificationCard result={message.verification} />}
      {message.limitations.length > 0 && (
        <div className="limitations">
          <strong>Research limitations</strong>
          <ul>
            {message.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export function PendingBubbles({ text }: { text: string }) {
  return (
    <>
      <div className="bubble bubble-user">{text}</div>
      <div className="bubble bubble-assistant pending" role="status">
        <span className="dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        Researching sources… this can take up to a minute.
      </div>
    </>
  )
}
