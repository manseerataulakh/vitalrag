import { useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import "./dashboard.css";

const API = import.meta.env.VITE_API_URL;

type Patient    = { id: number; code: string; fused_risk: number };
type Evidence   = { text: string; score: number };
type Prediction = { patient_id: number; code: string; ts_risk: number; text_risk: number; fused_risk: number; summary: string; evidence: Evidence[] };
type AskResult  = { answer: string; evidence: Evidence[] };

const riskClass  = (v: number) => v > 0.7 ? "high"     : v > 0.4 ? "medium"  : "low";
const riskColor  = (v: number) => v > 0.7 ? "#dc2626"  : v > 0.4 ? "#d97706" : "#16a34a";
const riskStatus = (v: number) => v > 0.7 ? "critical" : v > 0.4 ? "warning" : "stable";
const riskLabel  = (v: number) => v > 0.7 ? "Critical" : v > 0.4 ? "Elevated Risk" : "Stable";

const TOOLTIP_STYLE = {
  fontSize: 11, borderRadius: 6,
  border: "1px solid var(--border)", boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
};

function VitalChart({ data, dataKey, color, label, unit }: {
  data: any[]; dataKey: string; color: string; label: string; unit: string;
}) {
  return (
    <>
      <div className="chart-label">{label}</div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 2, right: 4, left: -30, bottom: 0 }}>
            <CartesianGrid strokeDasharray="2 3" stroke="var(--border)" />
            <XAxis dataKey="hour" tick={{ fontSize: 10, fill: "var(--text-3)" }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "var(--text-3)" }} tickLine={false} axisLine={false} unit={unit} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [`${Number(v)}${unit}`, label]} />
            <Line dataKey={dataKey} stroke={color} dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

export default function App() {
  const [patients,   setPatients]   = useState<Patient[]>([]);
  const [sel,        setSel]        = useState<number | null>(null);
  const [pred,       setPred]       = useState<Prediction | null>(null);
  const [vitals,     setVitals]     = useState<any[]>([]);
  const [question,   setQuestion]   = useState("");
  const [askResult,  setAskResult]  = useState<AskResult | null>(null);
  const [asking,     setAsking]     = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API}/patients`).then(r => r.json()).then((p: Patient[]) => {
      const sorted = [...p].sort((a, b) => b.fused_risk - a.fused_risk);
      setPatients(sorted);
      if (sorted.length) setSel(sorted[0].id);
    });
  }, []);

  useEffect(() => {
    if (sel == null) return;
    setAskResult(null);
    fetch(`${API}/patient/${sel}`).then(r => r.json()).then(d => {
      setPred(d.prediction);
      setVitals((d.vitals || []).map((r: any) => ({ hour: r.hour, hr: r.hr, spo2: r.spo2, rr: r.rr })));
    });
  }, [sel]);

  const handleAsk = async () => {
    if (!question.trim() || sel == null) return;
    setAsking(true); setAskResult(null);
    try {
      const res = await fetch(`${API}/ask?pid=${sel}&q=${encodeURIComponent(question)}`);
      setAskResult(await res.json());
    } finally { setAsking(false); }
  };

  const selPt = patients.find(p => p.id === sel);

  return (
    <div className="app">

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">Vital<span>RAG</span></div>
          <div className="sidebar-subtitle">ICU Deterioration Monitor</div>
        </div>
        <div className="sidebar-section-label">Patients — {patients.length}</div>
        <div className="patient-list">
          {patients.map(p => (
            <div
              key={p.id}
              className={`patient-item${p.id === sel ? " active" : ""}`}
              onClick={() => setSel(p.id)}
            >
              <span className="patient-code">{p.code}</span>
              <span className={`risk-pill ${riskClass(p.fused_risk)}`}>{p.fused_risk.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">

        {/* Header */}
        <div className="page-header">
          <span className="patient-title">{selPt?.code ?? "—"}</span>
          {pred && <>
            <span className="header-dot" />
            <span className="patient-meta">Patient #{pred.patient_id}</span>
          </>}
          {pred && <span className={`status-badge ${riskStatus(pred.fused_risk)}`}>{riskLabel(pred.fused_risk)}</span>}
        </div>

        {/* Risk cards */}
        {pred && (
          <div className="cards-row">
            {([
              { label: "Time-Series Risk", value: pred.ts_risk,    color: "#6366f1" },
              { label: "Notes Risk",       value: pred.text_risk,  color: "#0ea5e9" },
              { label: "Fused Risk",       value: pred.fused_risk, color: "#8b5cf6" },
            ] as const).map(({ label, value, color }) => (
              <div key={label} className="metric-card">
                <div className="metric-label">{label}</div>
                <div className="metric-value" style={{ color: riskColor(value) }}>{value.toFixed(2)}</div>
                <div className="metric-bar">
                  <div className="metric-bar-fill" style={{ width: `${value * 100}%`, background: color }} />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Content grid */}
        <div className="content-grid">

          {/* Left — vitals */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Vitals — First 24h</span>
              <span className="panel-tag">Hourly</span>
            </div>
            <div className="panel-body">
              {vitals.length > 0 ? <>
                <VitalChart data={vitals} dataKey="hr"   color="#ef4444" label="Heart Rate"       unit=" bpm" />
                <div className="chart-divider" />
                <VitalChart data={vitals} dataKey="spo2" color="#3b82f6" label="SpO₂"             unit="%" />
                <div className="chart-divider" />
                <VitalChart data={vitals} dataKey="rr"   color="#8b5cf6" label="Respiratory Rate" unit=" /min" />
              </> : <div className="empty">No vitals available</div>}
            </div>
          </div>

          {/* Right — evidence + Q&A */}
          <div>
            {pred && (
              <div className="panel">
                <div className="panel-header">
                  <span className="panel-title">Clinical Evidence</span>
                  <span className="panel-tag">RAG</span>
                </div>
                <div className="panel-body">
                  <div className="summary-text">{pred.summary}</div>
                  <ul className="evidence-list">
                    {pred.evidence.map((e, i) => (
                      <li key={i} className="evidence-item">
                        <span className="evidence-score">{e.score.toFixed(2)}</span>
                        <span>{e.text}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Ask about this patient</span>
                <span className="panel-tag">AI</span>
              </div>
              <div className="panel-body">
                <div className="qa-input-row">
                  <input
                    ref={inputRef}
                    className="qa-input"
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && handleAsk()}
                    placeholder="e.g. Signs of sepsis?"
                  />
                  <button className="qa-btn" onClick={handleAsk} disabled={asking || !question.trim()}>
                    {asking ? "…" : "Ask"}
                  </button>
                </div>
                {askResult && <>
                  <div className="qa-answer">{askResult.answer}</div>
                  <div className="qa-sources-label">Sources</div>
                  <ul className="evidence-list">
                    {askResult.evidence.map((e, i) => (
                      <li key={i} className="evidence-item">
                        <span className="evidence-score">{e.score.toFixed(2)}</span>
                        <span>{e.text}</span>
                      </li>
                    ))}
                  </ul>
                </>}
              </div>
            </div>
          </div>
        </div>

        <div className="footer">MIMIC-IV demo · Not for clinical use</div>
      </main>
    </div>
  );
}
