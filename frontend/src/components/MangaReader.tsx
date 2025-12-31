"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, ZoomIn, ZoomOut, ChevronLeft, ChevronRight, Image as ImageIcon, Grid } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/app-store";
import type { MangaPanel } from "@/lib/api";

// Helper to format dialogue from Record<string, string> to display string
function formatDialogue(dialogue: Record<string, string> | undefined): string {
  if (!dialogue || Object.keys(dialogue).length === 0) return "";
  return Object.entries(dialogue)
    .map(([character, text]) => `${character}: ${text}`)
    .join("\n");
}

export function MangaReader() {
  const { manga, title } = useAppStore();
  const [selectedPanel, setSelectedPanel] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [viewMode, setViewMode] = useState<"combined" | "grid">("combined"); // 默认显示带对白的合并图

  if (!manga) return null;

  const handleDownload = () => {
    if (manga.combined_image_base64) {
      const link = document.createElement("a");
      link.href = `data:image/png;base64,${manga.combined_image_base64}`;
      link.download = `${title || "manga"}.png`;
      link.click();
    }
  };

  const handlePanelClick = (panelNumber: number) => {
    setSelectedPanel(panelNumber);
  };

  const navigatePanel = (direction: "prev" | "next") => {
    if (selectedPanel === null) return;
    const newPanel =
      direction === "prev"
        ? Math.max(1, selectedPanel - 1)
        : Math.min(manga.panels.length, selectedPanel + 1);
    setSelectedPanel(newPanel);
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* 标题和工具栏 */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">{manga.title}</h2>
        <div className="flex items-center gap-2">
          {/* 视图切换按钮 */}
          <div className="flex items-center border rounded-lg overflow-hidden mr-2">
            <Button
              variant={viewMode === "combined" ? "default" : "ghost"}
              size="sm"
              onClick={() => setViewMode("combined")}
              className="rounded-none"
            >
              <ImageIcon className="w-4 h-4 mr-1" />
              长图
            </Button>
            <Button
              variant={viewMode === "grid" ? "default" : "ghost"}
              size="sm"
              onClick={() => setViewMode("grid")}
              className="rounded-none"
            >
              <Grid className="w-4 h-4 mr-1" />
              分格
            </Button>
          </div>
          <Button variant="outline" size="icon" onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}>
            <ZoomOut className="w-4 h-4" />
          </Button>
          <span className="text-sm text-muted-foreground w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <Button variant="outline" size="icon" onClick={() => setZoom((z) => Math.min(2, z + 0.25))}>
            <ZoomIn className="w-4 h-4" />
          </Button>
          <Button onClick={handleDownload} className="ml-4">
            <Download className="w-4 h-4 mr-2" />
            下载长图
          </Button>
        </div>
      </div>

      {/* 合并视图 - 显示带对白气泡的完整长图 */}
      {viewMode === "combined" && manga.combined_image_base64 && (
        <div
          className="flex justify-center"
          style={{
            transform: `scale(${zoom})`,
            transformOrigin: "top center",
          }}
        >
          <img
            src={`data:image/png;base64,${manga.combined_image_base64}`}
            alt={manga.title}
            className="max-w-full rounded-2xl shadow-lg"
          />
        </div>
      )}

      {/* 网格视图 - 显示单独的面板 */}
      {viewMode === "grid" && (
        <div
          className="grid gap-4 transition-transform duration-300"
          style={{
            gridTemplateColumns: `repeat(auto-fit, minmax(${280 * zoom}px, 1fr))`,
            transform: `scale(${zoom})`,
            transformOrigin: "top center",
          }}
        >
          {manga.panels.map((panel, index) => (
            <motion.div
              key={panel.panel_number}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => handlePanelClick(panel.panel_number)}
              className="manga-panel cursor-pointer group"
            >
              <div className="relative aspect-[3/4] bg-white rounded-2xl overflow-hidden shadow-lg">
                <img
                  src={`data:image/png;base64,${panel.image_base64}`}
                  alt={`Panel ${panel.panel_number}`}
                  className="w-full h-full object-cover"
                />

                {/* 面板编号 */}
                <div className="absolute top-2 left-2 w-8 h-8 bg-black/60 rounded-full flex items-center justify-center text-white text-sm font-bold">
                  {panel.panel_number}
                </div>

                {/* 悬浮时显示对白 */}
                {panel.dialogue && Object.keys(panel.dialogue).length > 0 && (
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <p className="text-white text-sm whitespace-pre-line">{formatDialogue(panel.dialogue)}</p>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* 面板详情弹窗 */}
      <AnimatePresence>
        {selectedPanel !== null && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
            onClick={() => setSelectedPanel(null)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="relative max-w-3xl max-h-[90vh] bg-white rounded-3xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              {/* 导航按钮 */}
              <button
                onClick={() => navigatePanel("prev")}
                disabled={selectedPanel === 1}
                className="absolute left-2 top-1/2 -translate-y-1/2 z-10 p-2 bg-white/80 rounded-full shadow disabled:opacity-30"
              >
                <ChevronLeft className="w-6 h-6" />
              </button>
              <button
                onClick={() => navigatePanel("next")}
                disabled={selectedPanel === manga.panels.length}
                className="absolute right-2 top-1/2 -translate-y-1/2 z-10 p-2 bg-white/80 rounded-full shadow disabled:opacity-30"
              >
                <ChevronRight className="w-6 h-6" />
              </button>

              {/* 图片 */}
              <img
                src={`data:image/png;base64,${manga.panels[selectedPanel - 1]?.image_base64}`}
                alt={`Panel ${selectedPanel}`}
                className="w-full h-auto"
              />

              {/* 对白 */}
              {manga.panels[selectedPanel - 1]?.dialogue && Object.keys(manga.panels[selectedPanel - 1].dialogue).length > 0 && (
                <div className="p-4 bg-white">
                  <div className="speech-bubble max-w-md mx-auto whitespace-pre-line">
                    {formatDialogue(manga.panels[selectedPanel - 1].dialogue)}
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
