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
