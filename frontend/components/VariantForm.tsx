"use client";

import { Loader2, RotateCcw, Search } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import type { AnalyzeRequest } from "@/types";

const EXAMPLE_SEQUENCE =
  "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

const INITIAL_FORM = {
  variant_name: "",
  gene: "",
  sequence: "",
  notes: ""
};

type VariantFormProps = {
  isLoading: boolean;
  onAnalyze: (input: AnalyzeRequest) => void;
  onClearResult: () => void;
};

export function VariantForm({ isLoading, onAnalyze, onClearResult }: VariantFormProps) {
  const [form, setForm] = useState(INITIAL_FORM);

  const sequenceLength = useMemo(() => form.sequence.replace(/\s+/g, "").length, [form.sequence]);

  function updateField(field: keyof typeof form, value: string) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onAnalyze({
      variant_name: form.variant_name.trim() || null,
      gene: form.gene.trim() || null,
      sequence: form.sequence,
      notes: form.notes.trim() || null
    });
  }

  function handleClear() {
    setForm(INITIAL_FORM);
    onClearResult();
  }

  return (
    <form className="toolPanel" onSubmit={handleSubmit}>
      <div className="panelHeader">
        <h2>Analyze Sequence</h2>
        <span className="buildPill">GRCh38</span>
      </div>

      <div className="formGrid twoColumn">
        <label>
          <span>Variant name</span>
          <input
            value={form.variant_name}
            onChange={(event) => updateField("variant_name", event.target.value)}
            placeholder="GRCh38-7-140753336-A-T"
          />
        </label>

        <label>
          <span>Gene</span>
          <input
            value={form.gene}
            onChange={(event) => updateField("gene", event.target.value.toUpperCase())}
            placeholder="BRAF"
          />
        </label>
      </div>

      <label className="sequenceField">
        <span>DNA sequence</span>
        <textarea
          required
          value={form.sequence}
          onChange={(event) => updateField("sequence", event.target.value.toUpperCase())}
          placeholder={EXAMPLE_SEQUENCE}
          rows={8}
        />
      </label>
      <div className="fieldHelp">
        <span>Use A, C, G, T, or N only.</span>
        <span>{sequenceLength.toLocaleString()} characters</span>
      </div>

      <label className="notesField">
        <span>Notes</span>
        <textarea
          value={form.notes}
          onChange={(event) => updateField("notes", event.target.value)}
          placeholder="Optional notes for this demo run"
          rows={3}
        />
      </label>

      <div className="buttonRow">
        <button className="primaryButton" disabled={isLoading} type="submit">
          {isLoading ? <Loader2 aria-hidden className="spin" size={18} /> : <Search aria-hidden size={18} />}
          Analyze Variant
        </button>
        <button className="secondaryButton" disabled={isLoading} type="button" onClick={handleClear}>
          <RotateCcw aria-hidden size={18} />
          Clear
        </button>
      </div>
    </form>
  );
}
