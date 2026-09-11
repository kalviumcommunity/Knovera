/**
 * TypeScript Data Models for Knovera RAG Frontend
 * Mirrors backend Pydantic schemas in src/models/schemas.py
 */

export interface SourceCitation {
  source: string;
  chunk_id?: string;
  score?: number;
  rank?: number;
  section?: string;
  doc_title?: string;
}

export interface QueryRequest {
  question: string;
  k?: number;
  score_threshold?: number;
  use_api?: boolean;
  metadata_filter?: Record<string, any>;
}

export interface QueryResponse {
  query: string;
  answer: string;
  sources: SourceCitation[];
  status: string; // 'answered' | 'refused_empty_context' | 'refused_low_similarity' | etc.
  latency_ms?: number;
  stage_latencies_ms?: Record<string, number>;
  timestamp?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  session_id: string;
  message: string;
  history?: ChatMessage[];
  k?: number;
  score_threshold?: number;
  use_api?: boolean;
}

export interface ChatResponse {
  session_id: string;
  message: string;
  sources: SourceCitation[];
  status: string;
  latency_ms?: number;
  timestamp?: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'unhealthy';
  service: string;
  version: string;
  environment: string;
  collection_name: string;
  total_indexed_chunks?: number;
  embedding_model: string;
  chat_model: string;
  vector_db_available: boolean;
  timestamp: string;
}

export interface DocumentInfo {
  filename: string;
  chunk_count: number;
  collection_name: string;
  indexed_at?: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total_documents: number;
  total_chunks: number;
}

export interface DocumentIndexingSummary {
  document: string;
  filename: string;
  chunks: number;
  indexed: number;
  raw_characters?: number;
  cleaned_characters?: number;
  collection_name?: string;
  stage_latencies_ms?: Record<string, number>;
}

export interface DocumentUploadResponse {
  status: string;
  filename: string;
  summary: DocumentIndexingSummary;
  message: string;
  timestamp?: string;
}

export interface EvaluationItem {
  query: string;
  expected_answer?: string;
  expected_doc?: string;
  min_score?: number;
}

export interface EvaluationRequest {
  test_cases: EvaluationItem[];
}

export interface EvaluationMetricSummary {
  total_queries: number;
  mean_precision_at_k: number;
  mean_recall: number;
  mean_reciprocal_rank: number;
  mean_groundedness: number;
  average_latency_ms: number;
}

export interface EvaluationDetailItem {
  query: string;
  answer: string;
  status: string;
  latency_ms: number;
  retrieved_sources: string[];
  hit?: boolean;
  reciprocal_rank?: number;
}

export interface EvaluationResponse {
  summary: EvaluationMetricSummary;
  details: EvaluationDetailItem[];
  timestamp: string;
}

export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceCitation[];
  latency_ms?: number;
  timestamp?: string;
  feedback?: 'up' | 'down';
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatTurn[];
  createdAt: string;
  updatedAt: string;
}

export type ConversationGroup = 'Today' | 'Yesterday' | 'Previous 7 Days' | 'Older';

export type GuardrailCategory =
  | 'hallucination'
  | 'injection'
  | 'pii'
  | 'policy'
  | 'retrieval'
  | 'custom';

export interface Guardrail {
  id: string;
  name: string;
  category: GuardrailCategory;
  description: string;
  rule: string;
  triggerCondition: string;
  action: string;
  priority: number;
  enabled: boolean;
  createdAt?: string;
  updatedAt?: string;
}

export interface LogSourceItem {
  source: string;
  doc_title?: string;
  section?: string;
  score?: number;
  chunk_id?: string;
}

export interface AuditLog {
  id: string;
  timestamp: string;
  sessionId: string;
  user: string;
  action: string;
  querySnippet?: string;
  latencyMs: number;
  groundednessScore?: number;
  guardrailStatus?: 'passed' | 'triggered' | 'bypassed' | 'refused' | string;
  guardrailName?: string;
  status: 'success' | 'warning' | 'error';
  details?: string;
  input?: string;
  output?: string;
  sources?: LogSourceItem[] | string[];
}

export interface AdminSettings {
  model: {
    primaryModel: string;
    temperature: number;
    maxTokens: number;
    systemPrompt: string;
  };
  chat: {
    sessionTimeoutMins: number;
    streamingEnabled: boolean;
    feedbackEnabled: boolean;
    defaultK: number;
  };
  retrieval: {
    similarityThreshold: number;
    hybridAlpha: number;
    rerankerEnabled: boolean;
    contextCompression: boolean;
  };
  guardrails: {
    strictness: 'low' | 'standard' | 'strict';
    piiMaskType: 'redacted' | 'asterisk' | 'hash';
    autoRefusalNotice: string;
  };
  system: {
    backendUrl: string;
    mongoHost?: string;
    chromaHost?: string;
    healthPollSec: number;
    organizationName: string;
  };
}

export interface ChunkItem {
  id: string;
  sourceDoc: string;
  section: string;
  chunkIndex: number;
  tokenCount: number;
  content: string;
  embeddingModel: string;
  indexedAt: string;
  metadata: Record<string, any>;
}

export interface ChunkListResponse {
  chunks: ChunkItem[];
  totalChunks: number;
  totalDocuments: number;
  embeddingModel: string;
  dimension: number;
}

export interface LogListResponse {
  logs: AuditLog[];
  total: number;
  page: number;
  pageSize: number;
  total_pages?: number;
}

export interface DashboardStatsResponse {
  totalQueries: number;
  avgLatencyMs: number;
  groundednessPercent: number;
  activeGuardrailsCount: number;
  totalChunks: number;
  totalDocuments: number;
  responseBreakdown: Record<string, number>;
  recentActivity: Array<{
    id: string;
    title: string;
    desc: string;
    time: string;
    status: string;
  }>;
}
