import type { Evidence, Stance, VerificationResult } from '../api/types'
import { VERDICT_LABEL, hostname } from '../format'

const GROUPS: { stance: Stance; title: string }[] = [
  { stance: 'SUPPORTS', title: 'Supporting evidence' },
  { stance: 'CONTRADICTS', title: 'Contradicting evidence' },
  { stance: 'NEUTRAL', title: 'Related evidence' },
]

export function VerificationCard({ result }: { result: VerificationResult }) {
  const percent = Math.round(result.confidence * 100)
  return (
    <section className="verification" aria-label="Fact-check result">
      <header className="verification-head">
        <span className={`verdict verdict-${result.verdict.toLowerCase()}`}>{VERDICT_LABEL[result.verdict]}</span>
        <div className="confidence" title="Confidence, capped by the number of independent sources">
          <div className="confidence-bar">
            <div style={{ width: `${percent}%` }} />
          </div>
          <span>{percent}% confidence</span>
        </div>
      </header>
      <p className="claim">“{result.claim}”</p>
      <p className="explanation">{result.explanation}</p>
      {GROUPS.map(({ stance, title }) => {
        const items = result.evidence.filter((e) => (e.stance ?? 'NEUTRAL') === stance)
        if (items.length === 0) return null
        return (
          <details key={stance} className={`evidence-group group-${stance.toLowerCase()}`} open={stance !== 'NEUTRAL'}>
            <summary>
              {title} <span className="count">{items.length}</span>
            </summary>
            <ul>
              {items.map((e, i) => (
                <EvidenceItem key={`${e.url ?? e.source}-${i}`} evidence={e} />
              ))}
            </ul>
          </details>
        )
      })}
    </section>
  )
}

function EvidenceItem({ evidence }: { evidence: Evidence }) {
  return (
    <li className="evidence">
      <div className="evidence-source">
        {evidence.url ? (
          <a href={evidence.url} target="_blank" rel="noopener noreferrer">
            {evidence.source}
          </a>
        ) : (
          evidence.source
        )}
        {evidence.url && <span className="host">{hostname(evidence.url)}</span>}
      </div>
      <p className="evidence-text">{evidence.content}</p>
    </li>
  )
}
