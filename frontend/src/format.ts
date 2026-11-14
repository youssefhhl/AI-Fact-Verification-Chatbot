import type { ContextDecision, Verdict } from './api/types'

export const VERDICT_LABEL: Record<Verdict, string> = {
  TRUE: 'True',
  FALSE: 'False',
  PARTIALLY_TRUE: 'Partially true',
  NOT_ENOUGH_EVIDENCE: 'Not enough evidence',
}

export const DECISION_LABEL: Record<ContextDecision, string> = {
  NEW_VERIFICATION: 'New verification',
  REUSE_CONTEXT: 'Follow-up · reused evidence',
  TARGETED_RESEARCH: 'Follow-up · additional research',
}

export function relativeTime(iso: string): string {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`
  return new Date(iso).toLocaleDateString()
}

export function hostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}
