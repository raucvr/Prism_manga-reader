"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Loader2, FileText, Wand2, Image, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore, AppStage } from "@/store/app-store";
import { api } from "@/lib/api";

interface StepInfo {
  stage: AppStage;
  icon: React.ReactNode;
  title: string;
  description: string;
}

const steps: StepInfo[] = [
  {
    stage: "parsing",
    icon: <FileText className="w-5 h-5" />,
    title: "Parse",
    description: "Extracting text and figures from paper...",
  },
  {
    stage: "generating-storyboard",
    icon: <Wand2 className="w-5 h-5" />,
    title: "Storyboard",
    description: "Prism is creating the manga script...",
  },
  {
    stage: "generating-manga",
    icon: <Image className="w-5 h-5" />,
    title: "Drawing",
    description: "Prism is drawing your manga...",
  },
  {
    stage: "completed",
    icon: <Check className="w-5 h-5" />,
    title: "Done",
    description: "Your manga is ready!",
  },
];

export function GenerationProgress() {
  const { stage, storyboard } = useAppStore();
  const [panelProgress, setPanelProgress] = useState({ current: 0, total: 0, message: "" });

  const currentStepIndex = steps.findIndex((s) => s.stage === stage);
  const isGenerating = ["parsing", "generating-storyboard", "generating-manga"].includes(stage);

  // Debug logging
  console.log("[GenerationProgress] Rendering:", { stage, currentStepIndex, isGenerating });

  // Poll progress when generating manga
  useEffect(() => {
    if (stage !== "generating-manga") {
      return;
    }

    const pollProgress = async () => {
      try {
        const progress = await api.getProgress();
        if (progress.stage === "generating" || progress.stage === "completed") {
          setPanelProgress({
            current: progress.current_panel,
            total: progress.total_panels,
            message: progress.message || "",
          });
        }
      } catch (e) {
        // Ignore errors during polling
      }
    };

    // Poll immediately and then every 1 second (faster updates during validation)
    pollProgress();
    const interval = setInterval(pollProgress, 1000);

    return () => clearInterval(interval);
  }, [stage]);

  if (!isGenerating && stage !== "completed") return null;

  return (
    <div className="w-full max-w-2xl mx-auto py-8">
      {/* 步骤指示器 */}
      <div className="relative flex justify-between mb-8">
        {/* 连接线 */}
        <div className="absolute top-5 left-0 right-0 h-0.5 bg-gray-200">
          <motion.div
            className="h-full bg-primary"
            initial={{ width: "0%" }}
            animate={{ width: `${(currentStepIndex / (steps.length - 1)) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>

        {/* 步骤点 */}
        {steps.map((step, index) => {
          const isActive = index === currentStepIndex;
          const isCompleted = index < currentStepIndex;

          return (
            <div key={step.stage} className="relative flex flex-col items-center">
              <motion.div
                className={cn(
                  "w-10 h-10 rounded-full flex items-center justify-center z-10",
                  isCompleted
                    ? "bg-primary text-white"
                    : isActive
                    ? "bg-primary text-white"
                    : "bg-gray-200 text-gray-400"
                )}
                animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                {isActive ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : isCompleted ? (
                  <Check className="w-5 h-5" />
                ) : (
                  step.icon
                )}
              </motion.div>
              <span
                className={cn(
                  "mt-2 text-sm font-medium",
                  isActive || isCompleted ? "text-foreground" : "text-muted-foreground"
                )}
              >
                {step.title}
              </span>
            </div>
          );
        })}
      </div>

      {/* 当前步骤详情 */}
      <motion.div
        key={stage}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <p className="text-lg text-muted-foreground">
          {steps[currentStepIndex]?.description}
        </p>

        {/* Panel progress during manga generation */}
        {stage === "generating-manga" && panelProgress.total > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-4 inline-flex flex-col items-center gap-2"
          >
            <span className="px-4 py-2 bg-primary/10 rounded-full text-2xl font-bold text-primary">
              Panel {panelProgress.current} / {panelProgress.total}
            </span>
            {panelProgress.message && (
              <span className="text-sm text-muted-foreground">
                {panelProgress.message}
              </span>
            )}
          </motion.div>
        )}

        {/* Storyboard preview */}
        {stage === "generating-manga" && storyboard && Array.isArray(storyboard.panels) && storyboard.panels.length > 0 && (
          <div className="mt-6 p-4 bg-white rounded-2xl shadow-sm">
            <p className="text-sm text-muted-foreground mb-2">Storyboard Preview</p>
            <div className="flex flex-wrap justify-center gap-2">
              {storyboard.panels.slice(0, 6).map((panel, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  className="w-16 h-20 bg-gray-100 rounded-lg flex items-center justify-center text-xs text-gray-400"
                >
                  {panel?.panel_type === "title" ? "📖" : "🎨"}
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </motion.div>

      {/* 可爱的加载动画 */}
      {isGenerating && (
        <div className="mt-8 flex justify-center">
          <motion.div
            className="flex gap-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {["🔮", "✨", "🎨"].map((emoji, i) => (
              <motion.span
                key={i}
                className="text-2xl"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 0.6, delay: i * 0.2, repeat: Infinity }}
              >
                {emoji}
              </motion.span>
            ))}
          </motion.div>
        </div>
      )}
    </div>
  );
}
