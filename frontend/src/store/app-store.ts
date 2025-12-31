import { create } from "zustand";
import type { MangaResponse, StoryboardResponse } from "@/lib/api";

export type AppStage =
  | "idle"
  | "uploading"
  | "parsing"
  | "generating-storyboard"
  | "generating-manga"
  | "completed"
  | "error";

export type MangaTheme =
  | "chiikawa"
  | "ghibli";

export type Language = "en-US" | "zh-CN" | "ja-JP";

interface AppState {
  // Current stage
  stage: AppStage;
  error: string | null;

  // File related
  file: File | null;
  extractedText: string;
  textPreview: string;

  // Generation settings
  mangaTheme: MangaTheme;
  numPanels: number;
  language: Language;
  title: string;

  // Generation results
  storyboard: StoryboardResponse | null;
  manga: MangaResponse | null;

  // Settings panel
  isSettingsOpen: boolean;

  // Actions
  setStage: (stage: AppStage) => void;
  setError: (error: string | null) => void;
  setFile: (file: File | null) => void;
  setExtractedText: (text: string, preview?: string) => void;
  setMangaTheme: (theme: MangaTheme) => void;
  setNumPanels: (num: number) => void;
  setLanguage: (lang: Language) => void;
  setTitle: (title: string) => void;
  setStoryboard: (storyboard: StoryboardResponse | null) => void;
  setManga: (manga: MangaResponse | null) => void;
  setSettingsOpen: (open: boolean) => void;
  reset: () => void;
}

const initialState = {
  stage: "idle" as AppStage,
  error: null,
  file: null,
  extractedText: "",
  textPreview: "",
  mangaTheme: "chiikawa" as MangaTheme,
  numPanels: 6,
  language: "en-US" as Language,
  title: "",
  storyboard: null,
  manga: null,
  isSettingsOpen: false,
};

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setStage: (stage) => set({ stage }),
  setError: (error) => set({ error, stage: error ? "error" : "idle" }),
  setFile: (file) => set({ file }),
  setExtractedText: (text, preview) =>
    set({ extractedText: text, textPreview: preview || text.slice(0, 500) }),
  setMangaTheme: (mangaTheme) => set({ mangaTheme }),
  setNumPanels: (numPanels) => set({ numPanels }),
  setLanguage: (language) => set({ language }),
  setTitle: (title) => set({ title }),
  setStoryboard: (storyboard) => set({ storyboard }),
  setManga: (manga) => set({ manga }),
  setSettingsOpen: (isSettingsOpen) => set({ isSettingsOpen }),
  reset: () => set(initialState),
}));
