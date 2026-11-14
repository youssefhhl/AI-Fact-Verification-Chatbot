import axios, { AxiosError } from 'axios'
import type { ChatRequest, ChatResponse, ConversationDetail, ConversationSummary, ErrorResponse } from './types'

// In development, Vite proxies /api to the backend (see vite.config.ts).
// Set VITE_API_URL to call a backend on another origin instead.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '',
  timeout: 180_000, // a full verification (web + scientific research) can take a while
})

export async function sendMessage(body: ChatRequest): Promise<ChatResponse> {
  return (await api.post<ChatResponse>('/api/chat', body)).data
}

export async function listConversations(): Promise<ConversationSummary[]> {
  return (await api.get<ConversationSummary[]>('/api/conversations', { params: { limit: 100 } })).data
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  return (await api.get<ConversationDetail>(`/api/conversations/${id}`)).data
}

/** A user-facing message for any API failure. */
export function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const body = error.response?.data as Partial<ErrorResponse> | undefined
    if (body?.error?.message) return body.error.message
    if (error.code === 'ECONNABORTED') return 'The request took too long. Please try again.'
    if (!error.response) return 'Cannot reach the TruthLens backend. Is it running on port 8000?'
    return `Request failed (HTTP ${error.response.status}).`
  }
  return 'Something went wrong. Please try again.'
}

/** Conversation id attached to an error, e.g. when the agent failed after saving the message. */
export function errorConversationId(error: unknown): string | null {
  if (error instanceof AxiosError) {
    const id = (error.response?.data as Partial<ErrorResponse> | undefined)?.error?.details?.conversation_id
    return typeof id === 'string' ? id : null
  }
  return null
}
