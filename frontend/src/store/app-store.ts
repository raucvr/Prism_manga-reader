import { create } from "zustand";
import type { JobLanguage, JobLayout, JobPanelRange, JobState, JobTheme } from "@/lib/api";

export type MangaTheme = JobTheme;
export type Language = JobLanguage;
export type PanelLayout = JobLayout;
export type PanelRange = JobPanelRange;

/** Options sent with `POST /api/jobs`. */
export interface GenerationOptions {
  mangaTheme: MangaTheme;
  language: Language;
  layout: PanelLayout;
  qa: boolean;
  panelRange: PanelRange;
  fast: boolean;
}

interface AppState extends GenerationOptions {
  /** The PDF picked in the dropzone. */
  file: File | null;
  /** PDFs of jobs created in this tab (in memory only), so a failed job can be retried. */
  jobFiles: Record<string, File>;

  // Settings panel
  isSettingsOpen: boolean;

  // Actions
  setFile: (file: File | null) => void;
  rememberJobFile: (jobId: string, file: File) => void;
  setLanguage: (lang: Language) => void;
  setLayout: (layout: PanelLayout) => void;
  setQa: (qa: boolean) => void;
  setPanelRange: (range: PanelRange) => void;
  setFast: (fast: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  /** Pre-fill the options from an existing job (used by "try again"). */
  applyJobOptions: (job: Pick<JobState, "theme" | "language" | "layout">) => void;
  /** Forget the picked file; keeps the user's option choices. */
  clearFile: () => void;
  reset: () => void;
}

export const DEFAULT_OPTIONS: GenerationOptions = {
  mangaTheme: "kumomo",
  language: "en-US",
  layout: "single",
  qa: true,
  panelRange: "auto",
  fast: false,
};

const initialState = {
  ...DEFAULT_OPTIONS,
  file: null,
  jobFiles: {},
  isSettingsOpen: false,
};

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setFile: (file) => set({ file }),
  rememberJobFile: (jobId, file) => set((s) => ({ jobFiles: { ...s.jobFiles, [jobId]: file } })),
  setLanguage: (language) => set({ language }),
  setLayout: (layout) => set({ layout }),
  setQa: (qa) => set({ qa }),
  setPanelRange: (panelRange) => set({ panelRange }),
  setFast: (fast) => set({ fast }),
  setSettingsOpen: (isSettingsOpen) => set({ isSettingsOpen }),
  applyJobOptions: (job) =>   // kumomo is the only cast: an older job may name a theme that no longer exists
    set({ mangaTheme: "kumomo", language: job.language, layout: job.layout }),
  clearFile: () => set({ file: null }),
  reset: () => set(initialState),
}));

/** Multipart body for `POST /api/jobs`. */
export function buildJobForm(file: File, options: GenerationOptions): FormData {
  const form = new FormData();
  form.append("file", file);
  form.append("theme", options.mangaTheme);
  form.append("language", options.language);
  form.append("layout", options.layout);
  form.append("qa", String(options.qa));
  form.append("panel_range", options.panelRange);
  form.append("fast", String(options.fast));
  return form;
}
