"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings, X, Check, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/app-store";
import { api, describeError, type ConfigResponse } from "@/lib/api";

interface SettingsPanelProps {
  /** Show the floating gear button (pages with their own settings button pass false). */
  showTrigger?: boolean;
}

export function SettingsPanel({ showTrigger = true }: SettingsPanelProps) {
  const isSettingsOpen = useAppStore((s) => s.isSettingsOpen);
  const setSettingsOpen = useAppStore((s) => s.setSettingsOpen);

  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  const loadConfig = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setConfig(await api.getConfig());
    } catch (err) {
      setError(describeError(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isSettingsOpen && !config) void loadConfig();
  }, [isSettingsOpen, config, loadConfig]);

  // Escape closes; focus moves into the panel when it opens.
  useEffect(() => {
    if (!isSettingsOpen) return;
    closeRef.current?.focus();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSettingsOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isSettingsOpen, setSettingsOpen]);

  return (
    <>
      {showTrigger && (
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setSettingsOpen(true)}
          className="fixed right-4 top-4 z-40 bg-white/70 shadow-sm backdrop-blur"
          aria-label="Open settings"
        >
          <Settings className="h-5 w-5" aria-hidden="true" />
        </Button>
      )}

      <AnimatePresence>
        {isSettingsOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-black/30"
              onClick={() => setSettingsOpen(false)}
              aria-hidden="true"
            />

            {/* Sidebar */}
            <motion.div
              role="dialog"
              aria-modal="true"
              aria-labelledby="settings-title"
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed right-0 top-0 z-50 h-full w-full max-w-md overflow-y-auto bg-cream shadow-2xl"
            >
              <div className="p-6">
                {/* Header */}
                <div className="mb-6 flex items-center justify-between">
                  <h2 id="settings-title" className="text-2xl font-bold">
                    Settings
                  </h2>
                  <Button
                    ref={closeRef}
                    variant="ghost"
                    size="icon"
                    onClick={() => setSettingsOpen(false)}
                    aria-label="Close settings"
                  >
                    <X className="h-5 w-5" aria-hidden="true" />
                  </Button>
                </div>

                {/* API Status */}
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0">
                    <CardTitle className="text-base">API Status</CardTitle>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void loadConfig()}
                      disabled={loading}
                      aria-label="Refresh API status"
                    >
                      <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} aria-hidden="true" />
                    </Button>
                  </CardHeader>
                  <CardContent>
                    {config ? (
                      <div className="space-y-3">
                        {config.providers.map((provider) => (
                          <div key={provider.name} className="flex items-center justify-between rounded-xl bg-white p-3">
                            <div>
                              <div className="font-medium capitalize">{provider.name.replace("_", " ")}</div>
                              <div className="text-xs text-muted-foreground">{provider.models.length} models</div>
                            </div>
                            <div className="flex items-center gap-2">
                              {provider.has_api_key ? (
                                <span className="flex items-center text-sm text-green-700">
                                  <Check className="mr-1 h-4 w-4" aria-hidden="true" />
                                  Configured
                                </span>
                              ) : (
                                <span className="text-sm text-yellow-700">Not configured</span>
                              )}
                              <div
                                className={cn("h-2 w-2 rounded-full", provider.enabled ? "bg-green-500" : "bg-gray-300")}
                                aria-label={provider.enabled ? "Enabled" : "Disabled"}
                                role="img"
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="py-4 text-center text-sm text-muted-foreground">
                        {loading ? "Loading…" : error ? `Couldn't load the configuration. ${error}` : "No data yet."}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Tip */}
                <p className="mt-6 text-center text-sm text-muted-foreground">
                  Configure API keys in <code className="rounded bg-white px-1">config/api_config.yaml</code>
                </p>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
