"use client";

import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, X, Loader2 } from "lucide-react";
import { cn, formatFileSize } from "@/lib/utils";
import { useAppStore } from "@/store/app-store";
import { api } from "@/lib/api";

export function Dropzone() {
  const { file, stage, setFile, setStage, setExtractedText, setError, setTitle } =
    useAppStore();

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const pdfFile = acceptedFiles[0];
      if (!pdfFile) return;

      setFile(pdfFile);
      setTitle(pdfFile.name.replace(".pdf", ""));
      setStage("uploading");

      try {
        setStage("parsing");
        console.log("[Dropzone] Uploading PDF...");
        const result = await api.uploadPDF(pdfFile);
        console.log("[Dropzone] Upload result:", result);

        // 确保有文本内容
        const fullText = result.full_text || "";
        const preview = result.text_preview || fullText.slice(0, 500) + "...";

        if (!fullText) {
          console.error("[Dropzone] No text extracted from PDF");
          setError("Failed to extract text from PDF. Please try a different file.");
          return;
        }

        console.log("[Dropzone] Setting extracted text, length:", fullText.length);
        setExtractedText(fullText, preview);
        setStage("idle");
      } catch (error) {
        console.error("[Dropzone] Upload failed:", error);
        setError(error instanceof Error ? error.message : "Upload failed");
      }
    },
    [setFile, setStage, setExtractedText, setError, setTitle]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
    },
    maxFiles: 1,
    disabled: stage === "uploading" || stage === "parsing",
  });

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setExtractedText("", "");
    setTitle("");
  };

  const isProcessing = stage === "uploading" || stage === "parsing";

  return (
    <div
      {...getRootProps()}
      className={cn(
        "relative w-full max-w-2xl mx-auto rounded-4xl border-3 border-dashed p-12 transition-all duration-300 cursor-pointer",
        isDragActive
          ? "border-primary bg-primary/10 scale-[1.02]"
          : "border-border bg-white/50 hover:border-primary/50 hover:bg-white/80",
        isProcessing && "pointer-events-none opacity-70"
      )}
    >
      <input {...getInputProps()} />

      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex flex-col items-center text-center"
          >
            <motion.div
              animate={isDragActive ? { scale: 1.1, rotate: 5 } : { scale: 1 }}
              className="mb-6"
            >
              <div className="w-24 h-24 rounded-full bg-primary/20 flex items-center justify-center">
                <Upload className="w-10 h-10 text-primary" />
              </div>
            </motion.div>

            <h3 className="text-xl font-medium mb-2">
              {isDragActive ? "Drop to upload" : "Drag PDF here"}
            </h3>
            <p className="text-muted-foreground">
              or <span className="text-primary font-medium">click to browse</span>
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              Supports academic papers, research reports, and PDF documents
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="file"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="flex items-center gap-4"
          >
            <div className="w-16 h-16 rounded-2xl bg-secondary/20 flex items-center justify-center shrink-0">
              {isProcessing ? (
                <Loader2 className="w-8 h-8 text-secondary animate-spin" />
              ) : (
                <FileText className="w-8 h-8 text-secondary" />
              )}
            </div>

            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{file.name}</p>
              <p className="text-sm text-muted-foreground">
                {formatFileSize(file.size)}
                {isProcessing && (
                  <span className="ml-2">
                    {stage === "uploading" ? "Uploading..." : "Parsing..."}
                  </span>
                )}
              </p>
            </div>

            {!isProcessing && (
              <button
                onClick={removeFile}
                className="p-2 rounded-full hover:bg-red-100 text-muted-foreground hover:text-red-500 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* 装饰性背景 */}
      <div className="absolute inset-0 -z-10 overflow-hidden rounded-4xl">
        <div className="absolute -top-20 -right-20 w-40 h-40 bg-primary/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-20 -left-20 w-40 h-40 bg-secondary/10 rounded-full blur-3xl" />
      </div>
    </div>
  );
}
