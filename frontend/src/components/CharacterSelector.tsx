import type { MangaTheme } from "@/store/app-store";

export interface ThemeOption {
  id: MangaTheme;
  name: string;
  emoji: string;
  color: string;
  description: string;
  style: string;
}

/** Prism's own cast (Papi, Kumo and Nezu, drawn from config/character_images): the only theme. */
export const THEMES: ThemeOption[] = [
  {
    id: "kumomo",
    name: "Kumomo",
    emoji: "☁️",
    color: "bg-sky-100 hover:bg-sky-200",
    description: "Original cute characters by Prism",
    style: "Kumomo style, original cute creatures (papi, kumo, nezu), pastel colors, soft lines",
  },
];
