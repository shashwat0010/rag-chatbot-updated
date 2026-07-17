"use client";

import { ExternalLink, FileText, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StructuredAnswer } from "@/components/chat/StructuredAnswer";
import type { Citation, QueryResponse } from "@/lib/api";

interface ResponseCardProps {
  query: string;
  response: QueryResponse;
}

function getConfidenceStyles(score: number): string {
  if (score >= 0.72) {
    return "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/25";
  }
  if (score >= 0.55) {
    return "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/25";
  }
  return "bg-slate-500/10 text-slate-700 dark:text-slate-400 border-slate-500/25";
}

function CitationItem({ citation, index }: { citation: Citation; index: number }) {
  return (
    <li className="rounded-xl border border-border/80 bg-muted/20 p-3.5 text-sm transition-all hover:bg-muted/40">
      <div className="mb-2 flex items-start gap-2.5">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg border border-primary/20 bg-primary/10 text-xs font-bold text-primary">
          {index}
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-semibold leading-snug text-foreground/90">{citation.title}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {citation.journal}
            {citation.year ? ` · ${citation.year}` : ""}
            {citation.authors ? ` · ${citation.authors}` : ""}
          </p>
        </div>
      </div>
      <a
        href={citation.pubmed_url}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-1.5 inline-flex items-center gap-1.5 text-xs font-semibold text-primary hover:text-primary/80 transition-colors"
      >
        View on PubMed (PMID: {citation.pmid})
        <ExternalLink className="h-3.5 w-3.5" />
      </a>
    </li>
  );
}

export function ResponseCard({ query, response }: ResponseCardProps) {
  const scorePercent = Math.round(response.confidence_score * 100);

  return (
    <Card className="relative overflow-hidden border border-border/70 shadow-lg backdrop-blur-md bg-card/95 transition-all">
      {/* Top colored strip indicator */}
      <div className="absolute top-0 left-0 right-0 h-1.5 bg-gradient-to-r from-primary to-sky-500" />
      
      <CardHeader className="pb-4 pt-6">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="flex items-center gap-2 text-base font-bold tracking-tight">
            <FileText className="h-4 w-4 text-primary" />
            Evidence-based Synthesis
          </CardTitle>
          <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-bold transition-colors ${getConfidenceStyles(response.confidence_score)}`}>
            Confidence: {scorePercent}%
          </span>
          {response.insufficient_evidence && (
            <span className="inline-flex items-center rounded-full border border-amber-500/25 bg-amber-500/10 px-2.5 py-0.5 text-xs font-bold text-amber-700 dark:text-amber-400">
              Low Evidence
            </span>
          )}
        </div>
        <p className="text-[11px] text-muted-foreground mt-1 bg-muted/40 px-2 py-1 rounded-md font-mono border border-border/40 inline-block w-fit">
          Query: {query}
        </p>
      </CardHeader>
      
      <CardContent className="space-y-5">
        <StructuredAnswer text={response.answer} />

        <div className="flex items-start gap-2.5 rounded-xl border border-amber-500/15 bg-amber-500/5 p-3.5 text-sm text-amber-900 dark:text-amber-100/90">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
          <div>
            <p className="font-bold text-amber-700 dark:text-amber-400">Clinical Uncertainty Note</p>
            <p className="mt-1 leading-relaxed opacity-95 text-xs sm:text-sm">{response.confidence_note}</p>
          </div>
        </div>

        {response.citations.length > 0 && (
          <div className="pt-2">
            <h4 className="mb-3 text-sm font-bold tracking-tight text-foreground/80">Retrieved Citations ({response.citations.length})</h4>
            <ul className="space-y-2.5">
              {response.citations.map((c, i) => (
                <CitationItem key={c.pmid} citation={c} index={i + 1} />
              ))}
            </ul>
          </div>
        )}

        {response.sources_searched.length > 0 && (
          <p className="text-[11px] text-muted-foreground pt-1">
            Validated Sources: {response.sources_searched.join(", ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

