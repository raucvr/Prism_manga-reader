/**
 * Time-left estimate for the drawing stage.
 *
 * Every poll is fed to `observe()`. Whenever the number of finished panels
 * goes up, the time since the previous increase divided by the number of new
 * panels is one "seconds per panel" sample (this is throughput, so it stays
 * right when the backend draws several panels in parallel). The estimate uses
 * a rolling average of the most recent samples.
 */

import type { JobState } from "@/lib/api";

const WINDOW = 6;

export class PanelEtaTracker {
  private jobId: string | null = null;
  private samples: number[] = [];
  private lastDone = -1;
  private lastAt = 0;
  /**
   * True when the baseline was taken in the middle of drawing (page opened or
   * reloaded mid-job): the first interval would be partial, so it is skipped.
   */
  private coldBaseline = false;
  private sawEarlierStage = false;

  observe(job: Pick<JobState, "job_id" | "status" | "progress">, now: number = Date.now()): void {
    if (job.job_id !== this.jobId) {
      this.jobId = job.job_id;
      this.samples = [];
      this.lastDone = -1;
      this.coldBaseline = false;
      this.sawEarlierStage = false;
    }

    if (job.status !== "rendering") {
      // Time spent outside the drawing stage must not count as drawing time.
      if (this.lastDone < 0) this.sawEarlierStage = true;
      this.lastDone = -1;
      return;
    }

    const done = job.progress?.done ?? 0;
    if (this.lastDone < 0 || done < this.lastDone) {
      this.lastDone = done;
      this.lastAt = now;
      // Only a baseline taken right as drawing starts marks a full interval.
      this.coldBaseline = !this.sawEarlierStage;
      this.sawEarlierStage = false;
      return;
    }

    if (done > this.lastDone) {
      if (!this.coldBaseline) {
        this.samples.push((now - this.lastAt) / (done - this.lastDone));
        if (this.samples.length > WINDOW) this.samples.shift();
      }
      this.coldBaseline = false;
      this.lastDone = done;
      this.lastAt = now;
    }
  }

  /** Milliseconds left, or null while there isn't enough data. */
  estimateMs(job: Pick<JobState, "status" | "progress">, now: number = Date.now()): number | null {
    if (job.status !== "rendering" || this.samples.length === 0) return null;
    const remaining = (job.progress?.total ?? 0) - (job.progress?.done ?? 0);
    if (remaining <= 0) return null;
    const avg = this.samples.reduce((a, b) => a + b, 0) / this.samples.length;
    const sinceLast = Math.max(0, now - this.lastAt);
    // The panel in progress is partly done; never let it drop below 20 % of a panel.
    const current = Math.max(avg - sinceLast, avg * 0.2);
    return avg * (remaining - 1) + current;
  }
}

/** Gentle, rounded wording: exact countdowns add pressure. */
export function formatEta(ms: number | null): string | null {
  if (ms === null || !Number.isFinite(ms)) return null;
  const minutes = ms / 60000;
  if (minutes < 1) return "less than a minute left";
  if (minutes < 60) return `about ${Math.round(minutes)} min left`;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes - h * 60);
  return m > 0 ? `about ${h} h ${m} min left` : `about ${h} h left`;
}
