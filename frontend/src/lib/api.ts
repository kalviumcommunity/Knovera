import {
  QueryRequest,
  QueryResponse,
  ChatRequest,
  ChatResponse,
  HealthResponse,
  DocumentListResponse,
  DocumentUploadResponse,
  EvaluationRequest,
  EvaluationResponse,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    const headers: Record<string, string> = {
      ...(options.headers as Record<string, string>),
    };

    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    try {
      const res = await fetch(url, {
        ...options,
        headers,
      });

      if (!res.ok) {
        let errorMessage = `HTTP ${res.status} ${res.statusText}`;
        try {
          const errorData = await res.json();
          if (errorData.detail) {
            errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
          }
        } catch {
          // ignore non-json error responses
        }
        throw new Error(errorMessage);
      }

      return await res.json();
    } catch (error: any) {
      if (error.message && error.message.includes('Failed to fetch')) {
        throw new Error(`Cannot connect to Knovera backend at ${this.baseUrl}. Please ensure the FastAPI server is running on port 8000.`);
      }
      throw error;
    }
  }

  /**
   * Health & telemetry status
   */
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health');
  }

  /**
   * Execute single-turn grounded question query
   */
  async submitQuery(payload: QueryRequest): Promise<QueryResponse> {
    return this.request<QueryResponse>('/api/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /**
   * Send a conversational message with session tracking
   */
  async sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /**
   * Get list of indexed documents
   */
  async getDocuments(): Promise<DocumentListResponse> {
    return this.request<DocumentListResponse>('/api/documents');
  }

  /**
   * Upload and immediately index a document into ChromaDB
   */
  async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.request<DocumentUploadResponse>('/api/documents', {
      method: 'POST',
      body: formData,
    });
  }

  /**
   * Run batch evaluation on RAG pipeline
   */
  async runEvaluation(payload: EvaluationRequest): Promise<EvaluationResponse> {
    return this.request<EvaluationResponse>('/api/evaluate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }
}

export const api = new ApiClient();
export default api;
