import { useEffect, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

const API = import.meta.env.VITE_API_URL;

type Patient = { id: number; code: string; fused_risk: number };
type Evidence = { text: string; score: number };
type Prediction = {
  patient_id: number;
  code: string;
  ts_risk: number;
  text_risk: number;
  fused_risk: number;
  summary: string;
  evidence: Evidence[];
};
type AskResult = { answer: string; evidence: Evidence[] };

const riskColor = (v: number) =>
  v > 0.7 ? "#c0392b" : v > 0.4 ? "#e67e22" : "#27ae60";

function RiskCard({ label, value }: { label: string; value: number }) {
  const color = riskColor(value);
  return (
    <div style={{ flex: 1, border: `2px solid ${color}`, borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color }}>{value.toFixed(2)}</div>
    </div>
  );
}

function VitalChart({
  data, dataKey, color, label, unit,
}: {
  data: any[]; dataKey: string; color: string; label: string; unit: string;
}) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color, marginBottom: 6 }}>{label}</div>
      <div style={{ height: 160 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="hour" tick={{ fontSize: 11 }} label={{ value: "hour", position: "insideBottom", offset: -2, fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} unit={unit} />
            <Tooltip formatter={(v: number) => [`${v}${unit}`, label]} />
            <Line dataKey={dataKey} stroke={color} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function App() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [vitals, setVitals] = useState<any[]>([]);
  const [question, setQuestion] = useState("");
  const [askResult, setAskResult] = useState<AskResult | null>(null);
  const [asking, setAsking] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API}/patients`)
      .then((r) => r.json())
      .then((p: Patient[]) => {
        const sorted = [...p].sort((a, b) => b.fused_risk - a.fused_risk);
        setPatients(sorted);
        if (sorted.length) setSel(sorted[0].id);
      });
  }, []);

  useEffect(() => {
    if (sel == null) return;
    setAskResult(null);
    fetch(`${API}/patient/${sel}`)
      .then((r) => r.json())
      .then((d) => {
        setPred(d.prediction);
        setVitals(
          (d.vitals || []).map((row: any) => ({
            hour: row.hour, hr: row.hr, spo2: row.spo2, rr: row.rr,
          }))
        );
      });
  }, [sel]);

  const handleAsk = async () => {
    if (!question.trim() || sel == null) return;
    setAsking(true);
    setAskResult(null);
    try {
      const res = await fetch(`${API}/ask?pid=${sel}&q=${encodeURIComponent(question)}`);
      setAskResult(await res.json());
    } finally {
      setAsking(false);
    }
  };

  return (
    <div style={{ fontFamily: "Inter, system-ui, sans-serif", maxWidth: 920, margin: "0 auto", padding: 32 }}>
      <h1 style={{ fontSize: 22, marginBottom: 2 }}>VitalRAG — ICU Deterioration Dashboard</h1>
      <p style={{ color: "#999", fontSize: 12, marginTop: 0 }}>
        Prototype on MIMIC-IV demo vitals + synthetic notes. Not for clinical use.
      </p>

      {/* Patient selector */}
      <div style={{ margin: "20px 0 24px" }}>
        <label style={{ fontSize: 13, marginRight: 8, fontWeight: 600 }}>Patient:</label>
        <select
          value={sel ?? ""}
          onChange={(e) => setSel(Number(e.target.value))}
          style={{ padding: "6px 10px", fontSize: 14, borderRadius: 6, border: "1px solid #ddd" }}
        >
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              {p.code} — risk {p.fused_risk.toFixed(2)}{p.fused_risk > 0.7 ? " ⚠" : ""}
            </option>
          ))}
        </select>
      </div>

      {/* Risk cards */}
      {pred && (
        <div style={{ display: "flex", gap: 12, marginBottom: 32 }}>
          <RiskCard label="Time-series risk" value={pred.ts_risk} />
          <RiskCard label="Notes risk" value={pred.text_risk} />
          <RiskCard label="Fused risk" value={pred.fused_risk} />
        </div>
      )}

      {/* Vitals — 3 separate charts */}
      {vitals.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <h3 style={{ fontSize: 15, marginBottom: 16 }}>Vitals over first 24h</h3>
          <VitalChart data={vitals} dataKey="hr"   color="#c0392b" label="Heart Rate"        unit=" bpm" />
          <VitalChart data={vitals} dataKey="spo2" color="#2980b9" label="SpO₂"              unit="%" />
          <VitalChart data={vitals} dataKey="rr"   color="#8e44ad" label="Respiratory Rate"  unit=" /min" />
        </div>
      )}

      {/* Evidence summary */}
      {pred && (
        <div style={{ marginBottom: 32 }}>
          <h3 style={{ fontSize: 15, marginBottom: 8 }}>Evidence from clinical notes</h3>
          <p style={{ fontStyle: "italic", color: "#333", background: "#f7f7f7", padding: 12, borderRadius: 8, margin: "0 0 12px" }}>
            {pred.summary}
          </p>
          <ul style={{ color: "#444", lineHeight: 1.7, paddingLeft: 20 }}>
            {pred.evidence.map((e, i) => (
              <li key={i}>
                {e.text}{" "}
                <span style={{ color: "#bbb", fontSize: 11 }}>({e.score.toFixed(2)})</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Q&A panel */}
      <div style={{ borderTop: "1px solid #eee", paddingTop: 24 }}>
        <h3 style={{ fontSize: 15, marginBottom: 12 }}>Ask about this patient</h3>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            placeholder='e.g. "Is this patient showing signs of sepsis?"'
            style={{
              flex: 1, padding: "8px 12px", fontSize: 14,
              border: "1px solid #ddd", borderRadius: 6, outline: "none",
            }}
          />
          <button
            onClick={handleAsk}
            disabled={asking || !question.trim()}
            style={{
              padding: "8px 18px", fontSize: 14, borderRadius: 6, border: "none",
              background: asking ? "#bbb" : "#2980b9", color: "#fff",
              cursor: asking ? "default" : "pointer", fontWeight: 600,
            }}
          >
            {asking ? "Asking…" : "Ask"}
          </button>
        </div>

        {askResult && (
          <div style={{ marginTop: 16 }}>
            <div style={{
              background: "#f0f7ff", border: "1px solid #bdd8f5",
              borderRadius: 8, padding: 14, marginBottom: 12,
              fontSize: 14, lineHeight: 1.6, color: "#1a3a5c",
            }}>
              {askResult.answer}
            </div>
            <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>Sources:</div>
            <ul style={{ fontSize: 12, color: "#666", lineHeight: 1.6, paddingLeft: 18 }}>
              {askResult.evidence.map((e, i) => (
                <li key={i}>
                  {e.text}{" "}
                  <span style={{ color: "#bbb" }}>({e.score.toFixed(2)})</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
