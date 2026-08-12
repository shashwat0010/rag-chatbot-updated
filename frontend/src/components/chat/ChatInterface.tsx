"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Send, Stethoscope, Trash2, History as HistoryIcon, User as UserIcon, LogOut, LogIn } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ThemeToggle } from "@/components/theme-toggle";
import { ResponseCard } from "@/components/chat/ResponseCard";
import { checkHealth, queryMedicalResearch, queryMedicalResearchStream, type QueryResponse } from "@/lib/api";
import { AuthModal } from "@/components/auth/AuthModal";
import { HistorySidebar, SearchHistoryItem } from "@/components/history/HistorySidebar";

function generateId() {
  if (typeof window !== "undefined" && window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID();
  }
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

interface Message {
  id: string;
  role: "user" | "assistant";
  query?: string;
  content?: string;
  response?: QueryResponse;
  error?: string;
}

const SAMPLE_QUERIES = [
  "What is the efficacy of SGLT2 inhibitors in heart failure with preserved ejection fraction?",
  "Meta-analysis evidence for metformin in gestational diabetes",
  "First-line antihypertensive therapy in adults under 55 — recent RCT evidence",
];

export function ChatInterface() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");

  // User Auth & History State
  const [user, setUser] = useState<{ id: number; email: string; name?: string } | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  useEffect(() => {
    // Check saved auth state
    if (typeof window !== "undefined") {
      const savedToken = localStorage.getItem("auth_token");
      const savedUserStr = localStorage.getItem("user_data");
      if (savedToken) setToken(savedToken);
      if (savedUserStr) {
        try {
          setUser(JSON.parse(savedUserStr));
        } catch (e) {
          console.error("Error parsing saved user:", e);
        }
      }
    }

    checkHealth()
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("user_data");
    setToken(null);
    setUser(null);
  };

  const handleAuthSuccess = (userData: { id: number; email: string; name?: string }, jwtToken: string) => {
    setUser(userData);
    setToken(jwtToken);
  };

  const handleSelectHistoryQuery = (item: SearchHistoryItem) => {
    const userMsg: Message = {
      id: generateId(),
      role: "user",
      content: item.query,
    };

    const assistantMsg: Message = {
      id: generateId(),
      role: "assistant",
      query: item.query,
      response: {
        answer: item.response,
        citations: [],
        confidence_note: `Saved Search History (${item.created_at ? new Date(item.created_at).toLocaleDateString() : "Saved"})`,
        confidence_score: item.confidence_score ? parseFloat(item.confidence_score) : 0.95,
        insufficient_evidence: false,
        sources_searched: ["Saved DB History"],
      },
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
  };

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = useCallback(async (text: string) => {
    const query = text.trim();
    if (!query || loading) return;

    const userMsg: Message = {
      id: generateId(),
      role: "user",
      content: query,
    };

    const assistantMsgId = generateId();
    const initialAssistantMsg: Message = {
      id: assistantMsgId,
      role: "assistant",
      query,
      response: {
        answer: "",
        citations: [],
        confidence_note: "Searching PubMed and analyzing literature...",
        confidence_score: 0.0,
        insufficient_evidence: false,
        sources_searched: ["PubMed"],
      },
    };
    
    setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
    setInput("");
    setLoading(true);

    try {
      await queryMedicalResearchStream(
        query,
        (tokenStr) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id === assistantMsgId && msg.response) {
                return {
                  ...msg,
                  response: {
                    ...msg.response,
                    answer: msg.response.answer + tokenStr,
                  },
                };
              }
              return msg;
            })
          );
        },
        (metadata) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id === assistantMsgId) {
                return {
                  ...msg,
                  response: {
                    ...msg.response,
                    ...metadata,
                  } as QueryResponse,
                };
              }
              return msg;
            })
          );
        },
        (errorMsg) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id === assistantMsgId) {
                return {
                  ...msg,
                  response: undefined,
                  error: errorMsg,
                };
              }
              return msg;
            })
          );
        },
        token
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong";
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id === assistantMsgId) {
            return {
              ...msg,
              response: undefined,
              error: message,
            };
          }
          return msg;
        })
      );
    } finally {
      setLoading(false);
    }
  }, [loading, token]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(input);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onAuthSuccess={handleAuthSuccess}
      />
      <HistorySidebar
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onSelectQuery={handleSelectHistoryQuery}
        token={token}
      />

      <header className="border-b border-border/60 bg-card/50 px-4 py-4 backdrop-blur sm:px-6">
        <div className="mx-auto flex max-w-4xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <Stethoscope className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">Medical Research Assistant</h1>
              <p className="text-xs text-muted-foreground">
                PubMed-grounded answers for clinicians · Not for emergency care
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsHistoryOpen(true)}
              className="gap-1.5 text-xs font-medium"
              title="View Search History"
            >
              <HistoryIcon className="h-4 w-4 text-teal-600 dark:text-teal-400" />
              <span className="hidden sm:inline">History</span>
            </Button>

            {user ? (
              <div className="flex items-center gap-2 border-l border-border/60 pl-2">
                <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 px-2.5 py-1 rounded-lg">
                  <UserIcon className="h-3.5 w-3.5 text-teal-600 dark:text-teal-400" />
                  <span className="max-w-[100px] truncate">{user.name || user.email}</span>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleLogout}
                  title="Log out"
                  className="h-8 w-8 text-muted-foreground hover:text-red-600"
                >
                  <LogOut className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <Button
                variant="default"
                size="sm"
                onClick={() => setIsAuthOpen(true)}
                className="gap-1.5 text-xs font-medium bg-teal-600 hover:bg-teal-700 text-white"
              >
                <LogIn className="h-3.5 w-3.5" />
                <span>Sign In</span>
              </Button>
            )}

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMessages([])}
              disabled={messages.length === 0}
              title="Clear chat"
            >
              <Trash2 className="h-4 w-4 text-muted-foreground" />
            </Button>
            <ThemeToggle />
            <span
              className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                apiStatus === "online"
                  ? "bg-emerald-100 text-emerald-800"
                  : apiStatus === "offline"
                    ? "bg-red-100 text-red-800"
                    : "bg-muted text-muted-foreground"
              }`}
            >
              API {apiStatus}
            </span>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto max-w-3xl space-y-6">
          {messages.length === 0 && (
            <div className="rounded-xl border border-dashed border-border bg-muted/20 p-6 text-center">
              <p className="text-sm text-muted-foreground">
                Ask a clinical research question. Answers are generated only from retrieved PubMed abstracts.
              </p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {SAMPLE_QUERIES.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => submit(q)}
                    className="rounded-lg border border-border bg-background px-3 py-2 text-left text-xs hover:bg-accent sm:max-w-xs"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={msg.role === "user" ? "flex justify-end" : "flex justify-start"}
            >
              {msg.role === "user" ? (
                <div className="max-w-[85%] rounded-2xl bg-primary px-4 py-3 text-sm text-primary-foreground">
                  {msg.content}
                </div>
              ) : msg.response ? (
                <div className="w-full max-w-3xl">
                  <ResponseCard query={msg.query || ""} response={msg.response} />
                </div>
              ) : (
                <div className="max-w-[85%] rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">
                  {msg.error}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Searching PubMed and synthesizing evidence…
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="border-t border-border/60 bg-card/80 p-4 backdrop-blur sm:px-6">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <Textarea
            ref={textareaRef}
            placeholder="e.g., What does recent evidence show about GLP-1 agonists and cardiovascular outcomes?"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading || apiStatus === "offline"}
            rows={1}
            className="min-h-[52px] max-h-[200px] resize-none overflow-y-auto"
          />
          <Button
            size="icon"
            className="h-[52px] w-[52px] shrink-0"
            onClick={() => submit(input)}
            disabled={loading || !input.trim() || apiStatus === "offline"}
            aria-label="Send query"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-muted-foreground">
          For licensed healthcare professionals. Does not replace clinical judgment or full-text review.
        </p>
      </footer>
    </div>
  );
}
