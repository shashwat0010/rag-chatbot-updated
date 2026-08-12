const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Citation {
  title: string;
  journal: string;
  year?: number | null;
  pubmed_url: string;
  pmid: string;
  authors?: string | null;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence_note: string;
  confidence_score: number;
  insufficient_evidence: boolean;
  sources_searched: string[];
}

export interface HealthResponse {
  status: string;
  version: string;
  llm_configured: boolean;
}

export async function queryMedicalResearch(query: string, token?: string | null): Promise<QueryResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = err.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join(", ")
          : `Request failed (${res.status})`;
    throw new Error(message || `Request failed (${res.status})`);
  }

  return res.json();
}

export async function queryMedicalResearchStream(
  query: string,
  onToken: (token: string) => void,
  onMetadata: (metadata: QueryResponse) => void,
  onError: (error: string) => void,
  token?: string | null
): Promise<void> {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const res = await fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify({ query }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const detail = err.detail;
      const message =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join(", ")
            : `Request failed (${res.status})`;
      throw new Error(message || `Request failed (${res.status})`);
    }

    const reader = res.body?.getReader();
    if (!reader) {
      throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // keep incomplete line in buffer

      let currentEvent = "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        if (trimmed.startsWith("event:")) {
          currentEvent = trimmed.slice(6).trim();
        } else if (trimmed.startsWith("data:")) {
          const rawData = trimmed.slice(5).trim();
          if (rawData === "[DONE]") continue;
          try {
            const parsedData = JSON.parse(rawData);
            if (currentEvent === "token") {
              onToken(parsedData);
            } else if (currentEvent === "metadata") {
              onMetadata(parsedData);
            } else if (currentEvent === "error") {
              onError(parsedData);
            }
          } catch (e) {
            console.error("Failed to parse SSE data chunk:", e, rawData);
          }
        }
      }
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : "Connection failed";
    onError(message);
  }
}


export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}
