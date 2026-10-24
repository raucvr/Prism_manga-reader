"use client";

import React, { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, X, RefreshCw } from "lucide-react";
import { cn, formatFileSize } from "@/lib/utils";
import { useAppStore } from "@/store/app-store";

/** Same limit as the backend (MAX_PDF_BYTES in backend/routes/jobs.py), checked before anything is uploaded. */
const MAX_PDF_BYTES = 60 * 1024 * 1024;

/**
 * Picks the PDF. Nothing is uploaded here: the file is sent together with the
 * generation options when the user presses "Generate".
 */
export function Dropzone({ disabled = false }: { disabled?: boolean }) {
  const file = useAppStore((s) => s.file);
  const setFile = useAppStore((s) => s.setFile);
  const [rejection, setRejection] = useState<string | null>(null);

  const onDrop = useCallback(
    (accepted: File[], rejected: FileRejection[]) => {
      const pdf = accepted[0];
      if (pdf) {
        setFile(pdf);
        setRejection(null);
        return;
      }
      const error = rejected[0]?.errors[0];
      if (error) {
        setRejection(
          error.code === "file-invalid-type"
            ? "That doesn't look like a PDF. Please choose a .pdf file."
            : error.code === "file-too-large"
              ? `That PDF is ${formatFileSize(rejected[0].file.size)}, and the limit is 60 MB. Please choose a smaller file.`
              : error.code === "too-many-files"
                ? "Please choose just one PDF."
                : error.message,
        );
      }
    },
    [setFile],
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxSize: MAX_PDF_BYTES,
    maxFiles: 1,
    multiple: false,
    disabled,
    // Once a file is picked the card shows explicit buttons instead of being one big button.
    noClick: !!file,
    noKeyboard: !!file,
  });

  const removeFile = () => {
    setFile(null);
    setRejection(null);
  };

  return (
    <div className="mx-auto w-full max-w-2xl">
      <div
        {...getRootProps(
          file
            ? {}
            : { role: "button", "aria-label": "Choose a PDF to turn into manga, or drop one here" },
        )}
        className={cn(
          "relative w-full rounded-4xl border-3 border-dashed p-8 transition-all duration-300 sm:p-12",
          !file && "cursor-pointer",
          isDragActive
            ? "scale-[1.02] border-primary bg-primary/10"
            : "border-border bg-white/50 hover:border-primary/50 hover:bg-white/80",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 focus-visible:ring-offset-2",
          disabled && "pointer-events-none opacity-70",
        )}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait" initial={false}>
          {!file ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center text-center"
            >
              <motion.div animate={isDragActive ? { scale: 1.1, rotate: 5 } : { scale: 1 }} className="mb-6">
                <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary/20 sm:h-24 sm:w-24">
                  <Upload className="h-10 w-10 text-primary-500" aria-hidden="true" />
                </div>
              </motion.div>

              <h2 className="mb-2 text-xl font-medium">{isDragActive ? "Drop to add it" : "Drag a PDF here"}</h2>
              <p className="text-muted-foreground">
                or <span className="font-medium text-primary-600">click to browse</span>
              </p>
              <p className="mt-2 text-sm text-muted-foreground">Academic papers, research reports and other PDFs</p>
            </motion.div>
          ) : (
            <motion.div
              key="file"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-wrap items-center gap-4"
            >
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-secondary/20 sm:h-16 sm:w-16">
                <FileText className="h-8 w-8 text-secondary-600" aria-hidden="true" />
              </div>

              <div className="min-w-0 flex-1">
                <p className="truncate font-medium" title={file.name}>
                  {file.name}
                </p>
                <p className="text-sm text-muted-foreground">{formatFileSize(file.size)} · ready</p>
              </div>

              <div className="flex shrink-0 gap-1">
                <button
                  type="button"
                  onClick={open}
                  className="flex h-10 items-center gap-1.5 rounded-full px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-primary/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
                  aria-label="Choose a different PDF"
                >
                  <RefreshCw className="h-4 w-4" aria-hidden="true" />
                  <span className="hidden sm:inline">Change</span>
                </button>
                <button
                  type="button"
                  onClick={removeFile}
                  className="flex h-10 w-10 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-red-100 hover:text-red-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
                  aria-label={`Remove ${file.name}`}
                >
                  <X className="h-5 w-5" aria-hidden="true" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Decorative background */}
        <div className="absolute inset-0 -z-10 overflow-hidden rounded-4xl" aria-hidden="true">
          <div className="absolute -right-20 -top-20 h-40 w-40 rounded-full bg-primary/10 blur-3xl" />
          <div className="absolute -bottom-20 -left-20 h-40 w-40 rounded-full bg-secondary/10 blur-3xl" />
        </div>
      </div>

      {rejection && (
        <p role="alert" className="mt-3 text-center text-sm text-red-600">
          {rejection}
        </p>
      )}
    </div>
  );
}
