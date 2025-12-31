"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings, X, Check, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useAppStore, Language } from "@/store/app-store";
import { api, ConfigResponse } from "@/lib/api";

const languages: { id: Language; name: string; flag: string }[] = [
  { id: "en-US", name: "English", flag: "🇺🇸" },
  { id: "zh-CN", name: "中文", flag: "🇨🇳" },
  { id: "ja-JP", name: "日本語", flag: "🇯🇵" },
];

export function SettingsPanel() {
  const {
    isSettingsOpen,
    setSettingsOpen,
    language,
    setLanguage,
  } = useAppStore();

  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isSettingsOpen && !config) {
      loadConfig();
    }
  }, [isSettingsOpen]);

  const loadConfig = async () => {
    setLoading(true);
    try {
      const data = await api.getConfig();
      setConfig(data);
    } catch (error) {
      console.error("Failed to load config:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Settings button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setSettingsOpen(true)}
        className="fixed top-4 right-4 z-40"
      >
        <Settings className="w-5 h-5" />
      </Button>

      {/* Settings panel */}
      <AnimatePresence>
        {isSettingsOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/30 z-40"
              onClick={() => setSettingsOpen(false)}
            />

            {/* Sidebar */}
            <motion.div
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed top-0 right-0 h-full w-full max-w-md bg-cream z-50 shadow-2xl overflow-y-auto"
            >
              <div className="p-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold">Settings</h2>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setSettingsOpen(false)}
                  >
                    <X className="w-5 h-5" />
                  </Button>
                </div>

                {/* Language settings */}
                <Card className="mb-4">
                  <CardHeader>
                    <CardTitle className="text-base">Output Language</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-2">
                      {languages.map((lang) => (
                        <button
                          key={lang.id}
                          onClick={() => setLanguage(lang.id)}
                          className={cn(
                            "p-3 rounded-xl border-2 transition-all",
                            language === lang.id
                              ? "border-primary bg-primary/10"
                              : "border-transparent bg-white hover:bg-gray-50"
                          )}
                        >
                          <div className="text-2xl mb-1">{lang.flag}</div>
                          <div className="text-sm font-medium">{lang.name}</div>
                        </button>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* AI automatically determines panel count */}

                {/* API Status */}
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle className="text-base">API Status</CardTitle>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={loadConfig}
                      disabled={loading}
                    >
                      <RefreshCw
                        className={cn("w-4 h-4", loading && "animate-spin")}
                      />
                    </Button>
                  </CardHeader>
                  <CardContent>
                    {config ? (
                      <div className="space-y-3">
                        {config.providers.map((provider) => (
                          <div
                            key={provider.name}
                            className="flex items-center justify-between p-3 bg-white rounded-xl"
                          >
                            <div>
                              <div className="font-medium capitalize">
                                {provider.name.replace("_", " ")}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {provider.models.length} models
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              {provider.has_api_key ? (
                                <span className="flex items-center text-green-600 text-sm">
                                  <Check className="w-4 h-4 mr-1" />
                                  Configured
                                </span>
                              ) : (
                                <span className="text-yellow-600 text-sm">
                                  Not configured
                                </span>
                              )}
                              <div
                                className={cn(
                                  "w-2 h-2 rounded-full",
                                  provider.enabled
                                    ? "bg-green-500"
                                    : "bg-gray-300"
                                )}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center text-muted-foreground py-4">
                        {loading ? "Loading..." : "Failed to load config"}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Tip */}
                <p className="text-sm text-muted-foreground mt-6 text-center">
                  Configure API keys in{" "}
                  <code className="bg-white px-1 rounded">
                    config/api_config.yaml
                  </code>
                </p>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
