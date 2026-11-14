// Mirrors the FastAPI schemas in backend/app/schemas (chat.py, verification.py, evidence.py).

export type Verdict = 'TRUE' | 'FALSE' | 'PARTIALLY_TRUE' | 'NOT_ENOUGH_EVIDENCE'
export type Stance = 'SUPPORTS' | 'CONTRADICTS' | 'NEUTRAL'
export type ContextDecision = 'NEW_VERIFICATION' | 'REUSE_CONTEXT' | 'TARGETED_RESEARCH'

export interface Evidence {
  source: string
  url: string | null
  content: string
  stance: Stance | null
  relevance: number | null
}

export interface VerificationResult {
  claim: string
  verdict: Verdict
  explanation: string
  confidence: number
  evidence: Evidence[]
}

export interface ChatRequest {
  message: string
  conversation_id?: string
}

export interface ChatResponse {
  conversation_id: string
  answer: string
  verification: VerificationResult | null
  decision: ContextDecision | null
  limitations: string[]
}

export interface ConversationSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  decision: ContextDecision | null
  claim: string | null
  verification: VerificationResult | null
  limitations: string[]
}

export interface ConversationDetail {
  id: string
  title: string
  created_at: string
  updated_at: string
  messages: Message[]
}

export interface ErrorResponse {
  error: { code: string; message: string; details: Record<string, unknown> | null }
}
