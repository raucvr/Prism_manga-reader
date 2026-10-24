"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, describeError, type JobPanel, type JobState } from "@/lib/api";
import { isRedrawOnly, isTerminal } from "@/lib/job-utils";
import { PanelEtaTracker } from "@/lib/eta";

/** A local job used instead of the API (`/?demo=1`): no polling. */
export interface DemoSource {
  job: JobState;
  /** Returns the job with panel `n` redrawn (simulates the backend). */
  redraw: (job: JobState, n: number) => JobState;
}

export interface UseJobResult {
  job: JobState | null;
  /** The first request hasn't come back yet. */
  loading: boolean;
  /** The latest poll failed (polling keeps retrying with backoff). */
  pollError: string | null;
  /** The backend doesn't know this job id. */
  notFound: boolean;
  /** Estimated milliseconds until all panels are drawn (null = unknown). */
  etaMs: number | null;
  /** Error from the last regenerate / cancel action. */
  actionError: string | null;
  clearActionError: () => void;
  regenerate: (n: number) => Promise<void>;
  cancel: () => Promise<void>;
  /** Poll right now (also restarts polling after errors). */
  refresh: () => void;
}

const POLL_MS = 2000;
const MAX_BACKOFF_MS = 15000;
/** Stop following a regenerated panel that is still drawing after this long. */
const REGEN_WATCH_MS = 10 * 60 * 1000;
/** Drop the optimistic "redrawing" state if the server never reports the redraw. */
const REGEN_UNSEEN_MS = 60 * 1000;
const DEMO_REDRAW_MS = 2500;

interface RegenWatch {
  /** The panel's `attempts`, `image_url` and `error`, and the job's `updated_at`, when the regenerate was requested. */
  attempts: number;
  imageUrl: string | null;
  error: string | null;
  updatedAt: string;
  /** The server has reported the panel as rendering/qa since then. */
  sawActive: boolean;
  since: number;
}

const isDrawing = (p: JobPanel) => p.status === "rendering" || p.status === "qa";

/** Panels sharing one strip image are redrawn together. */
function siblingsOf(job: JobState, n: number): number[] {
  const target = job.panels.find((p) => p.n === n);
  if (!target) return [];
  if (!target.image_url || (target.strip_index !== 0 && target.strip_index !== 1)) return [n];
  return job.panels.filter((p) => p.image_url === target.image_url).map((p) => p.n);
}

function withPanelStatus(job: JobState, numbers: number[], status: JobPanel["status"]): JobState {
  const set = new Set(numbers);
  return { ...job, panels: job.panels.map((p) => (set.has(p.n) ? { ...p, status } : p)) };
}

export function useJob(jobId: string | null, demo?: DemoSource | null): UseJobResult {
  const [serverJob, setServerJob] = useState<JobState | null>(demo?.job ?? null);
  const [loading, setLoading] = useState(!demo);
  const [pollError, setPollError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [etaMs, setEtaMs] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [watchVersion, setWatchVersion] = useState(0);

  const jobRef = useRef<JobState | null>(serverJob);
  const watchRef = useRef(new Map<number, RegenWatch>());
  const etaRef = useRef(new PanelEtaTracker());
  const pollNowRef = useRef<() => void>(() => undefined);
  const demoTimersRef = useRef<Array<ReturnType<typeof setTimeout>>>([]);

  const commitJob = useCallback((next: JobState | null) => {
    jobRef.current = next;
    setServerJob(next);
  }, []);

  // ---- Demo mode: a fixed local job --------------------------------------
  useEffect(() => {
    if (!demo) return;
    commitJob(demo.job);
    setLoading(false);
    setNotFound(false);
    setPollError(null);
    setEtaMs(null);
    const timers = demoTimersRef.current;
    return () => {
      timers.forEach(clearTimeout);
      timers.length = 0;
    };
  }, [demo, commitJob]);

  // ---- Polling --------------------------------------------------------------
  useEffect(() => {
    if (demo || !jobId) return;

    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let inflight: AbortController | null = null;
    let failures = 0;
    let settlePolls = 0; // extra polls after the job ends while panels are still settling (e.g. right after Stop)
    const watches = watchRef.current;
    watches.clear();

    commitJob(null);
    setLoading(true);
    setNotFound(false);
    setPollError(null);
    setEtaMs(null);

    const schedule = (ms: number) => {
      if (disposed) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => void tick(), ms);
    };

    const settleWatches = (next: JobState) => {
      const now = Date.now();
      const idle = isTerminal(next.status) && !next.redraw_panels?.length;
      watches.forEach((watch, n) => {
        const panel = next.panels.find((p) => p.n === n);
        const age = now - watch.since;
        if (!panel) {
          watches.delete(n);
        } else if (isDrawing(panel)) {
          watch.sawActive = true;
          if (age > REGEN_WATCH_MS) watches.delete(n);
        } else if (
          watch.sawActive ||
          panel.attempts > watch.attempts ||
          panel.image_url !== watch.imageUrl ||
          (!!panel.error && panel.error !== watch.error) ||
          // Saved since the request and idle again: the redraw ran, even one that failed before drawing anything.
          (idle && next.updated_at !== watch.updatedAt) ||
          age > REGEN_UNSEEN_MS
        ) {
          // Redrawn, failed (the panel shows why), or never picked up: show the server's state again.
          watches.delete(n);
        }
      });
    };

    const tick = async () => {
      timer = null;
      inflight?.abort();
      const controller = new AbortController();
      inflight = controller;
      try {
        const next = await api.getJob(jobId, { signal: controller.signal });
        if (disposed) return;
        failures = 0;
        settleWatches(next);
        if (isRedrawOnly(next)) {
          setEtaMs(null); // a redraw of single panels: no time-left estimate for the whole job
        } else {
          etaRef.current.observe(next);
          setEtaMs(etaRef.current.estimateMs(next));
        }
        commitJob(next);
        setPollError(null);
        setLoading(false);
        // Keep going until the job is over and no regenerated panel is outstanding.
        const settling = next.panels.some((p) => p.status === "rendering" || p.status === "qa");
        if (!isTerminal(next.status) || watches.size > 0) {
          settlePolls = 0;
          schedule(POLL_MS);
        } else if (settling && settlePolls < 15) {
          settlePolls += 1;
          schedule(POLL_MS);
        }
      } catch (err) {
        if (disposed || controller.signal.aborted) return;
        setLoading(false);
        if (err instanceof ApiError && err.status === 404) {
          setNotFound(true);
          return;
        }
        failures += 1;
        setPollError(describeError(err));
        schedule(Math.min(POLL_MS * 2 ** failures, MAX_BACKOFF_MS));
      }
    };

    pollNowRef.current = () => {
      failures = 0;
      schedule(0);
    };
    void tick();

    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      inflight?.abort();
      pollNowRef.current = () => undefined;
    };
  }, [jobId, demo, commitJob]);

  // While a regenerate is outstanding but the server still reports the old
  // panel (finished, failed or never drawn), show it as "rendering" so the
  // button press is visible.
  const job = useMemo(() => {
    void watchVersion; // recompute when watches change
    const watches = watchRef.current;
    if (!serverJob || watches.size === 0) return serverJob;
    let changed = false;
    const panels = serverJob.panels.map((p) => {
      const watch = watches.get(p.n);
      if (watch && !watch.sawActive && !isDrawing(p)) {
        changed = true;
        return { ...p, status: "rendering" as const };
      }
      return p;
    });
    return changed ? { ...serverJob, panels } : serverJob;
  }, [serverJob, watchVersion]);

  const regenerate = useCallback(
    async (n: number) => {
      const current = jobRef.current;
      const panel = current?.panels.find((p) => p.n === n);
      if (!current || !panel) return;
      setActionError(null);

      if (demo) {
        commitJob(withPanelStatus(current, siblingsOf(current, n), "rendering"));
        const t = setTimeout(() => {
          if (jobRef.current) commitJob(demo.redraw(jobRef.current, n));
        }, DEMO_REDRAW_MS);
        demoTimersRef.current.push(t);
        return;
      }

      watchRef.current.set(n, {
        attempts: panel.attempts,
        imageUrl: panel.image_url,
        error: panel.error ?? null,
        updatedAt: current.updated_at,
        sawActive: false,
        since: Date.now(),
      });
      setWatchVersion((v) => v + 1);
      try {
        await api.regeneratePanel(current.job_id, n);
        pollNowRef.current(); // also restarts polling if the job had finished
      } catch (err) {
        watchRef.current.delete(n);
        setWatchVersion((v) => v + 1);
        setActionError(`Couldn't redraw panel ${n}: ${describeError(err)}`);
      }
    },
    [demo, commitJob],
  );

  const cancel = useCallback(async () => {
    const current = jobRef.current;
    if (!current) return;
    setActionError(null);
    if (demo) {
      commitJob({ ...current, status: "cancelled", stage_message: "Stopped" });
      return;
    }
    try {
      await api.cancelJob(current.job_id);
      pollNowRef.current();
    } catch (err) {
      setActionError(`Couldn't stop this manga: ${describeError(err)}`);
    }
  }, [demo, commitJob]);

  const refresh = useCallback(() => pollNowRef.current(), []);
  const clearActionError = useCallback(() => setActionError(null), []);

  return {
    job,
    loading,
    pollError,
    notFound,
    etaMs,
    actionError,
    clearActionError,
    regenerate,
    cancel,
    refresh,
  };
}
