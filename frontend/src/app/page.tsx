"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wand2, RotateCcw } from "lucide-react";

import { Header } from "@/components/Header";
import { Dropzone } from "@/components/Dropzone";
import { ThemeSelector } from "@/components/CharacterSelector";
import { GenerationProgress } from "@/components/GenerationProgress";
import { MangaReader } from "@/components/MangaReader";
import { SettingsPanel } from "@/components/SettingsPanel";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/app-store";
import { api } from "@/lib/api";

export default function Home() {
  const {
    stage,
    file,
    extractedText,
    textPreview,
    mangaTheme,
    numPanels,
    language,
    title,
    manga,
    error,
    setStage,
    setStoryboard,
    setManga,
    setError,
    reset,
  } = useAppStore();

  const handleGenerate = async () => {
    if (!extractedText) {
      setError("Please upload a PDF file first");
      return;
    }

    try {
      // Step 1: Generate storyboard
      setStage("generating-storyboard");
      const storyboard = await api.generateStoryboard({
        text: extractedText,
        title,
        character: mangaTheme,
        num_panels: numPanels,
        language,
      });
      setStoryboard(storyboard);

      // Step 2: Generate manga
      setStage("generating-manga");
      const mangaResult = await api.generateManga({
        text: extractedText,
        title,
        character: mangaTheme,
        num_panels: numPanels,
        language,
      });
      setManga(mangaResult);

      // Done
      setStage("completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed, please try again");
    }
  };

  const showUploadSection = (stage === "idle" || stage === "uploading" || stage === "parsing") && !manga;
  const showGenerating = ["generating-storyboard", "generating-manga"].includes(stage);
  const showResult = stage === "completed" && manga;

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background decoration */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary/20 rounded-full blur-3xl" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-secondary/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-lavender/30 rounded-full blur-3xl" />
      </div>

      {/* Settings panel */}
      <SettingsPanel />

      <div className="container mx-auto px-4 pb-12">
        <Header />

        <AnimatePresence mode="wait">
          {/* Upload section */}
          {showUploadSection && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              <Dropzone />

              {/* Show options after file upload */}
              {file && extractedText && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="space-y-8"
                >
                  {/* Text preview */}
                  <div className="max-w-2xl mx-auto">
                    <h3 className="text-lg font-medium mb-2">Content Preview ({extractedText.length.toLocaleString()} chars)</h3>
                    <div className="p-4 bg-white rounded-2xl shadow-sm max-h-40 overflow-y-auto text-sm text-muted-foreground">
                      {textPreview}
                    </div>
                  </div>

                  {/* Theme selection */}
                  <ThemeSelector />

                  {/* Generate button */}
                  <div className="flex justify-center">
                    <Button
                      size="xl"
                      onClick={handleGenerate}
                      className="gap-2 px-12"
                    >
                      <Wand2 className="w-5 h-5" />
                      Generate Manga
                    </Button>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}

          {/* Generation progress */}
          {showGenerating && (
            <motion.div
              key="generating"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <GenerationProgress />
            </motion.div>
          )}

          {/* Result display */}
          {showResult && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <MangaReader />

              {/* New manga button */}
              <div className="flex justify-center">
                <Button variant="outline" onClick={reset} className="gap-2">
                  <RotateCcw className="w-4 h-4" />
                  Create New Manga
                </Button>
              </div>
            </motion.div>
          )}

          {/* Error message */}
          {stage === "error" && error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="max-w-md mx-auto text-center"
            >
              <div className="p-6 bg-red-50 rounded-3xl border border-red-200">
                <div className="text-4xl mb-4">😿</div>
                <h3 className="text-lg font-medium text-red-600 mb-2">
                  Oops! Something went wrong
                </h3>
                <p className="text-red-500 mb-4">{error}</p>
                <Button variant="outline" onClick={reset}>
                  Try Again
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="fixed bottom-4 left-1/2 -translate-x-1/2 text-sm text-muted-foreground/50">
        Powered by Prism AI 🔮
      </div>
    </main>
  );
}
