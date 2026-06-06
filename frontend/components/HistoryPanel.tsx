"use client";

import { Trash2 } from "lucide-react";
import type { AnalysisHistoryItem } from "@/types";

type HistoryPanelProps = {
  history: AnalysisHistoryItem[];
  selectedId?: string;
  onClear: () => void;
  onSelect: (result: AnalysisHistoryItem) => void;
};

export function HistoryPanel({ history, selectedId, onClear, onSelect }: HistoryPanelProps) {
  return (
    <aside className="historyPanel">
      <div className="panelHeader">
        <div>
          <p className="sectionEyebrow">Session</p>
          <h2>Recent Analyses</h2>
        </div>
        <button className="iconButton" disabled={history.length === 0} type="button" onClick={onClear} aria-label="Clear history">
          <Trash2 aria-hidden size={18} />
        </button>
      </div>

      {history.length === 0 ? (
        <p className="emptyHistory">Completed analyses will appear here.</p>
      ) : (
        <div className="historyList">
          {history.map((item) => (
            <button
              className={item.id === selectedId ? "historyItem selected" : "historyItem"}
              key={item.id}
              type="button"
              onClick={() => onSelect(item)}
            >
              <span className="historyVariant">{item.variant_name || item.gene || "Unnamed sequence"}</span>
              <span className="historyMeta">
                {item.risk_level} - {(item.pathogenic_probability * 100).toFixed(1)}% pathogenic
              </span>
              <span className="historySequence">{item.sequence_preview}</span>
            </button>
          ))}
        </div>
      )}
    </aside>
  );
}
