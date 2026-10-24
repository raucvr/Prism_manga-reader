"use client";

import React, { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import { Loader2 } from "lucide-react";

import { Header } from "@/components/Header";
import { Dropzone } from "@/components/Dropzone";
import { GenerateOptions } from "@/components/GenerateOptions";
import { RecentJobs } from "@/components/RecentJobs";
import { JobView } from "@/components/JobView";
import { SettingsPanel } from "@/components/SettingsPanel";
import type { DemoSource } from "@/hooks/useJob";
import { useAppStore } from "@/store/app-store";
import { useReaderPrefs } from "@/store/reader-prefs";

/**
 * Routing lives in the query string so a reload or a shared link restores the view:
 *   /             upload + options
 *   /?job=<id>    job progress + reader (polls the backend)
 *   /?demo=1      demo job without a backend (`strip` = strip layout, `early` = before panels exist)
 */
function navigate(search: string) {
  // Next.js keeps useSearchParams in sync with history.pushState (no server round trip).
  window.history.pushState(null, "", search ? `?${search}` : window.location.pathname);
}

export default function Page() {
  const calm = useReaderPrefs((s) => s.calm);

  useEffect(() => {
    useReaderPrefs.getState().hydrate();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.calm = calm ? "true" : "false";
  }, [calm]);

  return (
    <MotionConfig reducedMotion={calm ? "always" : "user"}>
      <BackgroundDecor />
      <Suspense fallback={<LoadingScreen />}>
        <Home />
      </Suspense>
    </MotionConfig>
  );
}

function BackgroundDecor() {
  return (
    <div className="page-decor pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
      <div className="absolute left-10 top-20 h-72 w-72 rounded-full bg-primary/20 blur-3xl" />
      <div className="absolute bottom-20 right-10 h-96 w-96 rounded-full bg-secondary/20 blur-3xl" />
      <div className="absolute left-1/2 top-1/2 h-[800px] w-[800px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-lavender/30 blur-3xl" />
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="flex min-h-screen items-center justify-center" role="status">
      <Loader2 className="h-8 w-8 animate-spin text-primary-500" aria-hidden="true" />
      <span className="sr-only">Loading…</span>
    </div>
  );
}

function Home() {
  const params = useSearchParams();
  const jobId = params.get("job");
  const demoParam = params.get("demo");
  const [demo, setDemo] = useState<{ key: string; source: DemoSource } | null>(null);

  // The demo fixture is only downloaded when asked for.
  useEffect(() => {
    if (!demoParam) return;
    let cancelled = false;
    import("@/lib/demoJob")
      .then((m) => {
        if (cancelled) return;
        setDemo({
          key: demoParam,
          source: {
            job: demoParam === "strip" ? m.demoStripJob : demoParam === "early" ? m.demoEarlyJob : m.demoJob,
            redraw: m.redrawDemoPanel,
          },
        });
      })
      .catch(() => {
        /* chunk failed to load: stay on the loading screen */
      });
    return () => {
      cancelled = true;
    };
  }, [demoParam]);

  const openJob = useCallback((id: string) => navigate(`job=${encodeURIComponent(id)}`), []);
  const goHome = useCallback(() => {
    const store = useAppStore.getState();
    store.clearFile();
    store.setSettingsOpen(false);
    navigate("");
  }, []);

  if (demoParam) {
    return (
      <>
        <SettingsPanel showTrigger={false} />
        {demo && demo.key === demoParam ? (
          <JobView key={`demo-${demoParam}`} jobId={null} demo={demo.source} onHome={goHome} onOpenJob={openJob} />
        ) : (
          <LoadingScreen />
        )}
      </>
    );
  }

  if (jobId) {
    return (
      <>
        <SettingsPanel showTrigger={false} />
        <JobView key={jobId} jobId={jobId} demo={null} onHome={goHome} onOpenJob={openJob} />
      </>
    );
  }

  return <UploadScreen onOpenJob={openJob} />;
}

function UploadScreen({ onOpenJob }: { onOpenJob: (jobId: string) => void }) {
  const file = useAppStore((s) => s.file);

  return (
    <>
      <SettingsPanel />
      <main className="container mx-auto px-4 pb-12">
        <Header />

        <div className="space-y-8">
          <Dropzone />

          <AnimatePresence mode="wait" initial={false}>
            {file ? (
              <motion.div
                key="options"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
              >
                <GenerateOptions onCreated={onOpenJob} />
              </motion.div>
            ) : (
              <motion.div
                key="recent"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              >
                <RecentJobs onOpen={onOpenJob} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <footer className="mt-16 text-center text-sm text-muted-foreground/70">Powered by Prism</footer>
      </main>
    </>
  );
}
