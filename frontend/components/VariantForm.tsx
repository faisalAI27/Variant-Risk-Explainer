"use client";

import { Loader2, RotateCcw, Search } from "lucide-react";
import { FormEvent, useState } from "react";
import type { VariantRequest } from "@/types";

const INITIAL_FORM = {
  chromosome: "7",
  position: "140753336",
  reference: "A",
  alternate: "T",
  gene: "BRAF",
  sequence_context: ""
};

type VariantFormProps = {
  isLoading: boolean;
  onAnalyze: (input: VariantRequest) => void;
};

export function VariantForm({ isLoading, onAnalyze }: VariantFormProps) {
  const [form, setForm] = useState(INITIAL_FORM);

  function updateField(field: keyof typeof form, value: string) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onAnalyze({
      chromosome: form.chromosome.trim(),
      position: Number(form.position),
      reference: form.reference.trim(),
      alternate: form.alternate.trim(),
      gene: form.gene.trim() || null,
      sequence_context: form.sequence_context.trim() || null
    });
  }

  function handleReset() {
    setForm(INITIAL_FORM);
  }

  return (
    <form className="toolPanel" onSubmit={handleSubmit}>
      <div className="panelHeader">
        <h2>Variant</h2>
        <span className="buildPill">GRCh38</span>
      </div>

      <div className="formGrid">
        <label>
          <span>Chromosome</span>
          <input
            required
            value={form.chromosome}
            onChange={(event) => updateField("chromosome", event.target.value)}
            placeholder="7"
          />
        </label>

        <label>
          <span>Position</span>
          <input
            required
            min={1}
            type="number"
            value={form.position}
            onChange={(event) => updateField("position", event.target.value)}
            placeholder="140753336"
          />
        </label>

        <label>
          <span>Reference</span>
          <input
            required
            maxLength={50}
            value={form.reference}
            onChange={(event) => updateField("reference", event.target.value.toUpperCase())}
            placeholder="A"
          />
        </label>

        <label>
          <span>Alternate</span>
          <input
            required
            maxLength={50}
            value={form.alternate}
            onChange={(event) => updateField("alternate", event.target.value.toUpperCase())}
            placeholder="T"
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
        <span>Sequence Context</span>
        <textarea
          value={form.sequence_context}
          onChange={(event) => updateField("sequence_context", event.target.value.toUpperCase())}
          placeholder="Optional GRCh38 sequence context"
          rows={5}
        />
      </label>

      <div className="buttonRow">
        <button className="primaryButton" disabled={isLoading} type="submit">
          {isLoading ? <Loader2 aria-hidden className="spin" size={18} /> : <Search aria-hidden size={18} />}
          Analyze
        </button>
        <button className="secondaryButton" disabled={isLoading} type="button" onClick={handleReset}>
          <RotateCcw aria-hidden size={18} />
          Reset
        </button>
      </div>
    </form>
  );
}
