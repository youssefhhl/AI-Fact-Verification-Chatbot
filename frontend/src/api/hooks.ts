import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { errorConversationId, getConversation, listConversations, sendMessage } from './client'
import type { ChatRequest } from './types'

export const keys = {
  conversations: ['conversations'] as const,
  conversation: (id: string) => ['conversation', id] as const,
}

export function useConversations() {
  return useQuery({ queryKey: keys.conversations, queryFn: listConversations })
}

export function useConversation(id: string | null) {
  return useQuery({
    queryKey: keys.conversation(id ?? ''),
    queryFn: () => getConversation(id!),
    enabled: id !== null,
  })
}

export function useSendMessage() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: ChatRequest) => sendMessage(body),
    // The backend history is the source of truth: reload it before the mutation settles,
    // so the pending bubble is replaced by the stored messages without flicker.
    onSuccess: async (data) => {
      await queryClient.fetchQuery({
        queryKey: keys.conversation(data.conversation_id),
        queryFn: () => getConversation(data.conversation_id),
        staleTime: 0,
      })
      await queryClient.invalidateQueries({ queryKey: keys.conversations })
    },
    // If the agent failed after the user message was stored, show that stored message.
    onError: async (error) => {
      const id = errorConversationId(error)
      if (id) {
        await queryClient.invalidateQueries({ queryKey: keys.conversation(id) })
        await queryClient.invalidateQueries({ queryKey: keys.conversations })
      }
    },
  })
}
