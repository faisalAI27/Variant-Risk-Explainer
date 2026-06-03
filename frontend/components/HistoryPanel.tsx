"use client";

import { Trash2 } from "lucide-react";
import type { VariantAnalysisResponse } from "@/types";

type HistoryPanelProps = {
  history: VariantAnalysisResponse[];
  selectedId?: string;
  onClear: () => void;
  onSelect: (result: VariantAnalysisResponse) => void;
};

export function HistoryPanel({ history, selectedId, onClear, onSelect }: HistoryPanelProps) {
  return (
    <aside className="historyPanel">
      <div className="panelHeader">
        <h2>History</h2>
        <button className="iconButton" disabled={history.length === 0} type="button" onClick={onClear} aria-label="Clear history">
          <Trash2 aria-hidden size={18} />
        </button>
      </div>

      {history.length === 0 ? (
        <p className="emptyHistory">No saved analyses</p>
      ) : (
        <div className="historyList">
          {history.map((item) => (
            <button
              className={item.request_id === selectedId ? "historyItem selected" : "historyItem"}
              key={item.request_id}
              type="button"
              onClick={() => onSelect(item)}
            >
              <span className="historyVariant">
                {item.input.chromosome}:{item.input.position} {item.input.reference}&gt;{item.input.alternate}
              </span>
              <span className="historyMeta">
                {item.risk_label.replaceAll("_", " ")} - {Math.round(item.confidence * 100)}%
              </span>
            </button>
          ))}
        </div>
      )}
    </aside>
  );
}
