"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  BookOpen,
  Brush,
  Check,
  FileText,
  Languages,
  Loader2,
  PenTool,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { JobState } from "@/lib/api";
import { formatEta } from "@/lib/eta";
import {
  PIPELINE,
  STATUS_COPY,
  isRedrawOnly,
  isTerminal,
  jobTitle,
  overallPercent,
  pipelineIndex,
  redrawLabel,
  type PipelineStage,
} from "@/lib/job-utils";

const STAGE_ICONS: Record<PipelineStage, LucideIcon> = {
  parsing: FileText,
  analyzing: Sparkles,
  storyboarding: PenTool,
  translating: Languages,
  rendering: Brush,
  assembling: BookOpen,
};

/** English jobs normally skip translation, so hide that step unless it is happening. */
function visibleStages(job: JobState) {
  return PIPELINE.filter((s) => s.id !== "translating" || job.language !== "en-US" || job.status === "translating");
}

export function ProgressBar({ value, label, className }: { value: number; label: string; className?: string }) {
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      className={cn("h-2 w-full overflow-hidden rounded-full bg-gray-200/80", className)}
    >
      <div
        className="h-full rounded-full bg-gradient-to-r from-primary-400 to-secondary-400 transition-[width] duration-700 ease-out"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function useStopConfirm(onCancel: () => Promise<void>, question = "Stop making this manga? Panels that are already drawn will stay.") {
  const [stopping, setStopping] = useState(false);
  const stop = async () => {
    if (!window.confirm(question)) return;
    setStopping(true);
    try {
      await onCancel();
    } finally {
      setStopping(false);
    }
  };
  return { stopping, stop };
}

interface ProgressProps {
  job: JobState;
  etaMs: number | null;
  onCancel: () => Promise<void>;
}

/** Full progress card, shown until the storyboard (and so the reader) exists. */
export function GenerationProgress({ job, etaMs, onCancel }: ProgressProps) {
  const stages = visibleStages(job);
  const current = pipelineIndex(job.status);
  const currentStageId = current >= 0 && current < PIPELINE.length ? PIPELINE[current].id : null;
  const activeIndex = currentStageId ? stages.findIndex((s) => s.id === currentStageId) : current < 0 ? -1 : stages.length;
  const copy = STATUS_COPY[job.status] ?? STATUS_COPY.queued;
  const percent = overallPercent(job);
  const eta = formatEta(etaMs);
  const { stopping, stop } = useStopConfirm(onCancel);

  return (
    <div className="mx-auto w-full max-w-2xl px-4 py-6 sm:py-10">
      <Card className="p-5 sm:p-8">
        <h2 className="text-center text-xl font-bold sm:text-2xl">{job.title ? jobTitle(job) : "Preparing your manga"}</h2>

        {/* Stage stepper */}
        <div className="relative mt-6">
          <div className="absolute left-4 right-4 top-4 h-0.5 bg-gray-200 sm:left-5 sm:right-5 sm:top-5" aria-hidden="true">
            <motion.div
              className="h-full bg-primary-400"
              initial={false}
              animate={{ width: `${(Math.max(0, Math.min(activeIndex, stages.length - 1)) / (stages.length - 1)) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <ol className="relative flex justify-between" aria-label="Steps">
            {stages.map((stage, i) => {
              const Icon = STAGE_ICONS[stage.id];
              const done = i < activeIndex;
              const active = i === activeIndex;
              return (
                <li
                  key={stage.id}
                  className="relative z-10 flex flex-col items-center"
                  aria-current={active ? "step" : undefined}
                >
                  <span
                    className={cn(
                      "flex h-8 w-8 items-center justify-center rounded-full sm:h-10 sm:w-10",
                      done || active ? "bg-primary-400 text-white" : "bg-gray-200 text-gray-500",
                    )}
                  >
                    {active ? (
                      <Loader2 className="h-4 w-4 animate-spin sm:h-5 sm:w-5" aria-hidden="true" />
                    ) : done ? (
                      <Check className="h-4 w-4 sm:h-5 sm:w-5" aria-hidden="true" />
                    ) : (
                      <Icon className="h-4 w-4 sm:h-5 sm:w-5" aria-hidden="true" />
                    )}
                  </span>
                  <span
                    aria-hidden="true"
                    className={cn(
                      "mt-2 hidden text-xs font-medium sm:block",
                      done || active ? "text-foreground" : "text-muted-foreground",
                    )}
                  >
                    {stage.label}
                  </span>
                  <span className="sr-only">
                    {stage.label}: {done ? "done" : active ? "in progress" : "not started"}
                  </span>
                </li>
              );
            })}
          </ol>
        </div>

        {/* What's happening now */}
        <div className="mt-8 text-center" aria-live="polite">
          <p className="text-lg font-medium">{copy.message}</p>
          {job.stage_message && <p className="mt-1 text-sm text-muted-foreground">{job.stage_message}</p>}
        </div>

        <div className="mt-6">
          <ProgressBar value={percent} label="Overall progress" />
          <div className="mt-2 flex justify-between text-sm text-muted-foreground">
            <span>{percent}%</span>
            <span>{eta ?? (job.status === "rendering" ? "Estimating time left…" : "")}</span>
          </div>
        </div>

        {!isTerminal(job.status) && (
          <div className="mt-6 flex justify-center">
            <button
              type="button"
              onClick={stop}
              disabled={stopping}
              className="rounded-full px-4 py-2 text-sm text-muted-foreground underline-offset-4 hover:text-red-600 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 disabled:opacity-50"
            >
              {stopping ? "Stopping…" : "Stop generating"}
            </button>
          </div>
        )}
      </Card>

      <p className="mt-4 text-center text-sm text-muted-foreground">
        You can close this tab: open the same link later to pick up where you left off.
      </p>
    </div>
  );
}

/** One-line status shown above the reader while panels are still being drawn. */
export function JobStatusBar({ job, etaMs, onCancel }: ProgressProps) {
  // A redraw of single panels: whole-job progress and time left would describe panels that are not being drawn.
  const redrawOnly = isRedrawOnly(job);
  const copy = STATUS_COPY[job.status] ?? STATUS_COPY.queued;
  const eta = formatEta(etaMs);
  const { stopping, stop } = useStopConfirm(
    onCancel,
    redrawOnly ? "Stop this redraw? Pictures that are already drawn will stay." : undefined,
  );
  const total = job.progress?.total || job.panels.length;
  const done = job.progress?.done ?? job.panels.filter((p) => p.status === "done").length;

  return (
    <div className="rounded-2xl bg-white/85 px-3 py-2 shadow-sm ring-1 ring-black/5 sm:px-4">
      <div className="flex items-center gap-2 text-sm">
        <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary-500" aria-hidden="true" />
        <p className="min-w-0 flex-1 truncate" aria-live="polite">
          {redrawOnly ? (
            <span className="font-medium">{redrawLabel(job.redraw_panels ?? [])}</span>
          ) : (
            <>
              <span className="font-medium">{job.stage_message || copy.message}</span>
              <span className="text-muted-foreground">
                {" · "}
                {done}/{total} drawn{eta ? ` · ${eta}` : ""}
              </span>
            </>
          )}
        </p>
        {!isTerminal(job.status) && (
          <button
            type="button"
            onClick={stop}
            disabled={stopping}
            className="shrink-0 rounded-full px-2 py-1 text-xs text-muted-foreground hover:bg-red-50 hover:text-red-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 disabled:opacity-50"
          >
            {stopping ? "Stopping…" : "Stop"}
          </button>
        )}
      </div>
      {!redrawOnly && <ProgressBar value={overallPercent(job)} label="Overall progress" className="mt-1.5 h-1.5" />}
    </div>
  );
}
