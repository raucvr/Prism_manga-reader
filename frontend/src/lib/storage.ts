/**
 * localStorage helpers that never throw.
 *
 * Storage can be unavailable (private windows, blocked site data, SSR) and any
 * access may throw, so every read/write is wrapped and failures are ignored:
 * the UI must work without persistence.
 */

export function readStorage(key: string): string | null {
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStorage(key: string, value: string): void {
  try {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(key, value);
  } catch {
    /* quota exceeded / storage blocked: ignore */
  }
}

export function readJSON<T>(key: string, isValid: (value: unknown) => value is T): T | null {
  const raw = readStorage(key);
  if (raw === null) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    return isValid(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function writeJSON(key: string, value: unknown): void {
  try {
    writeStorage(key, JSON.stringify(value));
  } catch {
    /* unserialisable value: ignore */
  }
}

const positionKey = (jobId: string) => `prism.reader.position.${jobId}`;

/** Last panel number the reader was on for this job, if remembered. */
export function readReaderPosition(jobId: string): number | null {
  const raw = readStorage(positionKey(jobId));
  if (raw === null) return null;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

export function writeReaderPosition(jobId: string, panelNumber: number): void {
  writeStorage(positionKey(jobId), String(panelNumber));
}
