"use client";

import React, { useEffect, useState } from "react";
import { Columns2, Loader2, RectangleVertical, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ChoiceGroup, type ChoiceOption } from "@/components/ui/choice-group";
import { Switch } from "@/components/ui/switch";
import { api, describeError } from "@/lib/api";
import { isNewJobLanguage } from "@/lib/job-utils";
import {
  buildJobForm,
  useAppStore,
  type Language,
  type PanelLayout,
  type PanelRange,
} from "@/store/app-store";

/** The languages of `NEW_JOB_LANGUAGES` ("ja-JP" is only found in older jobs). */
const LANGUAGES: ReadonlyArray<ChoiceOption<Language>> = [
  { value: "en-US", label: "English", icon: <span className="text-xl">🇺🇸</span> },
  { value: "zh-CN", label: "中文", icon: <span className="text-xl">🇨🇳</span> },
];

const LAYOUTS: ReadonlyArray<ChoiceOption<PanelLayout>> = [
  {
    value: "single",
    label: "Single",
    hint: "One panel per picture",
    icon: <RectangleVertical className="h-5 w-5 text-secondary-600" />,
  },
  {
    value: "strip",
    label: "Strip",
    hint: "Two panels side by side",
    icon: <Columns2 className="h-5 w-5 text-secondary-600" />,
  },
];

const RANGES: ReadonlyArray<ChoiceOption<PanelRange>> = [
  { value: "auto", label: "Auto", hint: "Prism decides" },
  { value: "short", label: "Short", hint: "≈ 12–18 panels" },
  { value: "long", label: "Long", hint: "≈ 28–40 panels" },
];

interface GenerateOptionsProps {
  /** Called with the new job id once the backend accepted the job. */
  onCreated: (jobId: string) => void;
}

export function GenerateOptions({ onCreated }: GenerateOptionsProps) {
  const file = useAppStore((s) => s.file);
  const language = useAppStore((s) => s.language);
  const layout = useAppStore((s) => s.layout);
  const panelRange = useAppStore((s) => s.panelRange);
  const qa = useAppStore((s) => s.qa);
  const fast = useAppStore((s) => s.fast);
  const setLanguage = useAppStore((s) => s.setLanguage);
  const setLayout = useAppStore((s) => s.setLayout);
  const setPanelRange = useAppStore((s) => s.setPanelRange);
  const setQa = useAppStore((s) => s.setQa);
  const setFast = useAppStore((s) => s.setFast);
  const rememberJobFile = useAppStore((s) => s.rememberJobFile);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Options copied from an older job ("try again") may name a language that is no longer offered.
  useEffect(() => {
    if (!isNewJobLanguage(language)) setLanguage("en-US");
  }, [language, setLanguage]);

  const handleGenerate = async () => {
    if (!file || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const { job_id } = await api.createJob(buildJobForm(file, useAppStore.getState()));
      rememberJobFile(job_id, file);
      onCreated(job_id);
    } catch (err) {
      setError(describeError(err));
      setSubmitting(false);
    }
  };

  return (
    <section aria-label="Generation options" className="mx-auto w-full max-w-2xl space-y-8">
      <Card className="space-y-6 p-4 sm:p-6">
        <ChoiceGroup
          legend="Output language"
          description="💡 English gives the best results and costs less"
          name="language"
          value={language}
          options={LANGUAGES}
          onChange={setLanguage}
          columns={2}
        />
        <ChoiceGroup
          legend="Layout"
          description={
            language === "en-US"
              ? "Strip costs about half as much"
              : "💡 Single keeps Chinese lettering most accurate; Strip costs about half as much"
          }
          name="layout"
          value={layout}
          options={LAYOUTS}
          onChange={setLayout}
          columns={2}
        />
        <ChoiceGroup legend="Length" name="panel-range" value={panelRange} options={RANGES} onChange={setPanelRange} />

        <div className="-mx-3 divide-y divide-border">
          <Switch
            label="Quality check"
            description="Checks every panel and redraws the ones with mistakes. A little slower."
            checked={qa}
            onCheckedChange={setQa}
          />
          <Switch
            label="Fast mode"
            description="Draws quicker, with slightly less detail."
            checked={fast}
            onCheckedChange={setFast}
          />
        </div>
      </Card>

      {error && (
        <div role="alert" className="rounded-2xl border border-red-200 bg-red-50 p-4 text-center text-sm text-red-600">
          {error}
        </div>
      )}

      <div className="flex justify-center">
        <Button size="xl" onClick={handleGenerate} disabled={!file || submitting} className="gap-2 px-12">
          {submitting ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
              Uploading…
            </>
          ) : (
            <>
              <Wand2 className="h-5 w-5" aria-hidden="true" />
              Generate manga
            </>
          )}
        </Button>
      </div>
    </section>
  );
}
