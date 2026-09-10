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
  Guardrail,
  AuditLog,
  Conversation,
  ChatTurn,
} from './types';
import { getStoredAuth } from './auth';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

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

    // Attach active user / admin session identity for multi-tenant isolation
    const auth = getStoredAuth();
    if (auth) {
      headers['X-User-Id'] = auth.email || auth.id;
      headers['X-User-Role'] = auth.role;
      if (auth.role === 'admin') {
        headers['X-Admin-Id'] = auth.email || auth.id;
      }
      if (auth.organization) {
        headers['X-Tenant-Id'] = auth.organization;
      }
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
        throw new Error(`Cannot connect to Knovera backend at ${this.baseUrl}. Ensure the FastAPI server is running on port 8000.`);
      }
      throw error;
    }
  }

  /** Health & telemetry status */
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health');
  }

  /** Execute single-turn grounded question query */
  async submitQuery(payload: QueryRequest): Promise<QueryResponse> {
    return this.request<QueryResponse>('/api/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /** Send a conversational message with session tracking */
  async sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /** Get list of indexed documents from server/MongoDB */
  async getDocuments(adminId?: string): Promise<DocumentListResponse> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<DocumentListResponse>(`/api/documents?admin_id=${encodeURIComponent(activeAdmin)}`);
  }

  /** Upload and immediately index a document into MongoDB Atlas */
  async uploadDocument(file: File, adminId?: string): Promise<DocumentUploadResponse> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    const formData = new FormData();
    formData.append('file', file);
    formData.append('admin_id', activeAdmin);

    return this.request<DocumentUploadResponse>(`/api/documents?admin_id=${encodeURIComponent(activeAdmin)}`, {
      method: 'POST',
      body: formData,
    });
  }

  /** Delete a document source from server and MongoDB */
  async deleteDocument(filename: string, adminId?: string): Promise<{ status: string; filename: string }> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<{ status: string; filename: string }>(
      `/api/documents/${encodeURIComponent(filename)}?admin_id=${encodeURIComponent(activeAdmin)}`,
      {
        method: 'DELETE',
      }
    );
  }

  /** Run batch evaluation on RAG pipeline */
  async runEvaluation(payload: EvaluationRequest): Promise<EvaluationResponse> {
    return this.request<EvaluationResponse>('/api/evaluate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /* -------------------------------------------------------------
   * GUARDRAILS DATABASE API (MONGODB)
   * ------------------------------------------------------------- */
  async getGuardrails(adminId?: string): Promise<Guardrail[]> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<Guardrail[]>(`/api/guardrails?admin_id=${encodeURIComponent(activeAdmin)}`);
  }

  async createGuardrail(payload: Partial<Guardrail>, adminId?: string): Promise<Guardrail> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<Guardrail>('/api/guardrails', {
      method: 'POST',
      body: JSON.stringify({ ...payload, admin_id: activeAdmin }),
    });
  }

  async updateGuardrail(id: string, payload: Partial<Guardrail>, adminId?: string): Promise<Guardrail> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<Guardrail>(`/api/guardrails/${id}?admin_id=${encodeURIComponent(activeAdmin)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
  }

  async deleteGuardrail(id: string, adminId?: string): Promise<{ status: string; id: string }> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<{ status: string; id: string }>(`/api/guardrails/${id}?admin_id=${encodeURIComponent(activeAdmin)}`, {
      method: 'DELETE',
    });
  }

  async reorderGuardrails(orderedIds: string[], adminId?: string): Promise<{ status: string; count: number }> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<{ status: string; count: number }>('/api/guardrails/reorder', {
      method: 'POST',
      body: JSON.stringify({ orderedIds, admin_id: activeAdmin }),
    });
  }

  /* -------------------------------------------------------------
   * AUDIT LOGS DATABASE API (MONGODB)
   * ------------------------------------------------------------- */
  async getLogs(
    params: { search?: string; status?: string; page?: number; pageSize?: number } = {},
    adminId?: string
  ): Promise<LogListResponse> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    const q = new URLSearchParams();
    if (params.search) q.append('search', params.search);
    if (params.status && params.status !== 'all') q.append('status', params.status);
    if (params.page) q.append('page', String(params.page));
    if (params.pageSize) q.append('page_size', String(params.pageSize));
    q.append('admin_id', activeAdmin);

    const qs = q.toString();
    return this.request<LogListResponse>(`/api/logs${qs ? `?${qs}` : ''}`);
  }

  async recordLog(payload: Partial<AuditLog>, adminId?: string): Promise<AuditLog> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<AuditLog>('/api/logs', {
      method: 'POST',
      body: JSON.stringify({ ...payload, admin_id: activeAdmin }),
    });
  }

  /* -------------------------------------------------------------
   * KNOWLEDGE CHUNKS DATABASE API (MONGODB ATLAS)
   * ------------------------------------------------------------- */
  async getChunks(
    params: { search?: string; doc?: string } = {},
    adminId?: string
  ): Promise<ChunkListResponse> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    const q = new URLSearchParams();
    if (params.search) q.append('search', params.search);
    if (params.doc && params.doc !== 'all') q.append('doc', params.doc);
    q.append('admin_id', activeAdmin);

    const qs = q.toString();
    return this.request<ChunkListResponse>(`/api/chunks${qs ? `?${qs}` : ''}`);
  }

  /* -------------------------------------------------------------
   * DASHBOARD TELEMETRY STATS API (MONGODB)
   * ------------------------------------------------------------- */
  async getDashboardStats(adminId?: string): Promise<DashboardStatsResponse> {
    const activeAdmin = adminId || getStoredAuth()?.email || 'admin@knovera.ai';
    return this.request<DashboardStatsResponse>(`/api/dashboard/stats?admin_id=${encodeURIComponent(activeAdmin)}`);
  }

  /* -------------------------------------------------------------
   * CONVERSATIONS & CHAT HISTORY DATABASE API (MONGODB)
   * ------------------------------------------------------------- */
  async getConversations(userId?: string): Promise<Conversation[]> {
    const activeUser = userId || getStoredAuth()?.email || 'user@knovera.ai';
    return this.request<Conversation[]>(`/api/conversations?user_id=${encodeURIComponent(activeUser)}`);
  }

  async createConversation(title: string = 'New chat', id?: string, userId?: string): Promise<Conversation> {
    const activeUser = userId || getStoredAuth()?.email || 'user@knovera.ai';
    return this.request<Conversation>('/api/conversations', {
      method: 'POST',
      body: JSON.stringify({ id, title, user_id: activeUser }),
    });
  }

  async renameConversation(id: string, title: string): Promise<Conversation> {
    return this.request<Conversation>(`/api/conversations/${id}`, {
      method: 'PUT',
      body: JSON.stringify({ title }),
    });
  }

  async deleteConversation(id: string): Promise<{ status: string; id: string }> {
    return this.request<{ status: string; id: string }>(`/api/conversations/${id}`, {
      method: 'DELETE',
    });
  }

  async saveChatMessage(conversationId: string, message: ChatTurn): Promise<ChatTurn> {
    return this.request<ChatTurn>(`/api/conversations/${conversationId}/messages`, {
      method: 'POST',
      body: JSON.stringify(message),
    });
  }
}

export const api = new ApiClient();
