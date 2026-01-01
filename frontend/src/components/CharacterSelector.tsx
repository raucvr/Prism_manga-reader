"use client";

import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useAppStore, MangaTheme } from "@/store/app-store";

interface ThemeOption {
  id: MangaTheme;
  name: string;
  emoji: string;
  color: string;
  description: string;
  style: string;
}

const themes: ThemeOption[] = [
  {
    id: "chiikawa",
    name: "Chiikawa",
    emoji: "🐹",
    color: "bg-pink-100 border-pink-300 hover:bg-pink-200",
    description: "Cute Chiikawa characters by Nagano",
    style: "Chiikawa style by Nagano, round fluffy creatures, thick outlines, pastel colors, simple dot eyes",
  },
  {
    id: "ghibli",
    name: "Studio Ghibli",
    emoji: "🌿",
    color: "bg-emerald-100 border-emerald-300 hover:bg-emerald-200",
    description: "Dreamy watercolor style",
    style: "Studio Ghibli style, soft watercolor, whimsical characters, nature themes, hand-painted look",
  },
];

export function ThemeSelector() {
  const { mangaTheme, setMangaTheme } = useAppStore();

  return (
    <div className="w-full max-w-xl mx-auto">
      <h3 className="text-lg font-medium mb-4 text-center">Choose Manga Style</h3>
      <div className="flex justify-center gap-4">
        {themes.map((theme) => (
          <motion.button
            key={theme.id}
            onClick={() => setMangaTheme(theme.id)}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
            className={cn(
              "relative p-4 rounded-2xl border-2 transition-all duration-200 text-left w-48",
              theme.color,
              mangaTheme === theme.id
                ? "ring-2 ring-primary ring-offset-2 border-primary"
                : "border-transparent"
            )}
          >
            <div className="text-3xl mb-2">{theme.emoji}</div>
            <div className="font-medium">{theme.name}</div>
            <div className="text-xs text-muted-foreground mt-1">
              {theme.description}
            </div>

            {mangaTheme === theme.id && (
              <motion.div
                layoutId="theme-indicator"
                className="absolute -top-1 -right-1 w-6 h-6 bg-primary rounded-full flex items-center justify-center text-white text-xs"
                initial={false}
              >
                ✓
              </motion.div>
            )}
          </motion.button>
        ))}
      </div>
    </div>
  );
}

// Export theme styles for use in generation
export function getThemeStyle(themeId: MangaTheme): string {
  const theme = themes.find((t) => t.id === themeId);
  return theme?.style || themes[0].style;
}
