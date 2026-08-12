"use client";

import React, { useEffect, useState } from "react";
import { History, X, Trash2, Clock, CheckCircle, Search } from "lucide-react";

export interface SearchHistoryItem {
  id: number;
  query: str;
  response: str;
  confidence_score?: string;
  evidence_count: number;
  created_at: string;
}

interface HistorySidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectQuery: (item: SearchHistoryItem) => void;
  token: string | null;
}

export const HistorySidebar: React.FC<HistorySidebarProps> = ({
  isOpen,
  onClose,
  onSelectQuery,
  token,
}) => {
  const [historyItems, setHistoryItems] = useState<SearchHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiBase}/api/history`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        throw new Error("Failed to load search history.");
      }
      const data = await res.json();
      setHistoryItems(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && token) {
      fetchHistory();
    }
  }, [isOpen, token]);

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!token) return;
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiBase}/api/history/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setHistoryItems((prev) => prev.filter((item) => item.id !== id));
      }
    } catch (err) {
      console.error("Delete history error:", err);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-40 flex justify-start bg-black/50 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="w-full max-w-md bg-white dark:bg-slate-900 h-full shadow-2xl border-r border-slate-200 dark:border-slate-800 flex flex-col">
        {/* Sidebar Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-2 text-slate-900 dark:text-white font-semibold">
            <History className="w-5 h-5 text-teal-600 dark:text-teal-400" />
            <span>Search History</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Sidebar Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {!token ? (
            <div className="text-center py-12 px-4 text-slate-500">
              <Search className="w-12 h-12 mx-auto mb-3 text-slate-300 dark:text-slate-700" />
              <p className="font-medium text-slate-700 dark:text-slate-300">Sign in to view search history</p>
              <p className="text-xs mt-1">Your past research queries will be safely stored in your account.</p>
            </div>
          ) : loading ? (
            <div className="text-center py-12 text-slate-500 text-sm animate-pulse">
              Loading your previous searches...
            </div>
          ) : error ? (
            <div className="text-center py-8 text-red-500 text-sm">{error}</div>
          ) : historyItems.length === 0 ? (
            <div className="text-center py-12 px-4 text-slate-500">
              <Clock className="w-10 h-10 mx-auto mb-2 text-slate-300 dark:text-slate-700" />
              <p className="text-sm">No saved search history found yet.</p>
            </div>
          ) : (
            historyItems.map((item) => (
              <div
                key={item.id}
                onClick={() => {
                  onSelectQuery(item);
                  onClose();
                }}
                className="group relative p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 hover:bg-teal-50 dark:hover:bg-teal-950/40 border border-slate-200 dark:border-slate-800 hover:border-teal-300 dark:hover:border-teal-700/50 cursor-pointer transition-all shadow-xs"
              >
                <div className="flex items-start justify-between gap-2">
                  <h4 className="text-sm font-medium text-slate-900 dark:text-slate-100 line-clamp-2 pr-6">
                    {item.query}
                  </h4>
                  <button
                    onClick={(e) => handleDelete(item.id, e)}
                    title="Delete search"
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-red-500 transition-opacity rounded-md"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <div className="flex items-center gap-3 mt-3 text-xs text-slate-500 dark:text-slate-400">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                    {new Date(item.created_at).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>

                  {item.evidence_count > 0 && (
                    <span className="flex items-center gap-1 text-teal-600 dark:text-teal-400">
                      <CheckCircle className="w-3.5 h-3.5" />
                      {item.evidence_count} sources
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
