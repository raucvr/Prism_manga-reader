"use client";

import React, { useEffect, useState } from "react";
import { ChevronRight } from "lucide-react";
import { api, type JobSummary } from "@/lib/api";
import { STATUS_COPY, formatRelativeTime, isRedrawOnly, isTerminal, redrawLabel } from "@/lib/job-utils";
import { THEMES } from "@/components/CharacterSelector";
import { cn } from "@/lib/utils";

const COLLAPSED_COUNT = 4;

/** Latest jobs from `GET /api/jobs`; renders nothing if there are none or the backend is unreachable. */
export function RecentJobs({ onOpen }: { onOpen: (jobId: string) => void }) {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    api
      .listJobs({ signal: controller.signal })
      .then((res) => setJobs(Array.isArray(res?.jobs) ? res.jobs : []))
      .catch(() => {
        if (!controller.signal.aborted) setJobs([]);
      });
    return () => controller.abort();
  }, []);

  if (!jobs || jobs.length === 0) return null;
  const visible = expanded ? jobs : jobs.slice(0, COLLAPSED_COUNT);

  return (
    <section aria-labelledby="recent-mangas" className="mx-auto w-full max-w-2xl">
      <h2 id="recent-mangas" className="mb-3 text-lg font-medium">
        Recent mangas
      </h2>
      <ul className="space-y-2">
        {visible.map((job) => {
          const theme = THEMES.find((t) => t.id === job.theme);
          // A redraw of single panels: whole-job progress would describe panels that are not being drawn.
          const redrawOnly = isRedrawOnly(job);
          const status = redrawOnly ? redrawLabel(job.redraw_panels ?? []) : (STATUS_COPY[job.status]?.title ?? job.status);
          const inProgress = !isTerminal(job.status);
          const progress =
            !redrawOnly && job.progress && job.progress.total > 0 ? ` · ${job.progress.done}/${job.progress.total} panels` : "";
          return (
            <li key={job.job_id}>
              <a
                href={`/?job=${encodeURIComponent(job.job_id)}`}
                onClick={(e) => {
                  if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
                  e.preventDefault();
                  onOpen(job.job_id);
                }}
                className="flex items-center gap-3 rounded-2xl bg-white/80 p-3 shadow-sm ring-1 ring-black/5 transition-colors hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
              >
                <span className="text-2xl" aria-hidden="true">
                  {theme?.emoji ?? "📖"}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium">{job.title || "Untitled manga"}</span>
                  <span className="block truncate text-sm text-muted-foreground">
                    {formatRelativeTime(job.created_at)} ·{" "}
                    <span className={cn(inProgress && "text-primary-700", job.status === "failed" && "text-red-700")}>
                      {status}
                    </span>
                    {progress}
                  </span>
                </span>
                <ChevronRight className="h-5 w-5 shrink-0 text-muted-foreground" aria-hidden="true" />
              </a>
            </li>
          );
        })}
      </ul>
      {jobs.length > COLLAPSED_COUNT && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 rounded-full px-3 py-1 text-sm font-medium text-primary-700 hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
        >
          {expanded ? "Show fewer" : `Show all ${jobs.length}`}
        </button>
      )}
    </section>
  );
}
