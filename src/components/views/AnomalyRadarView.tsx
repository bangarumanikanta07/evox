import React, { useState } from 'react';
import {
  Radio,
  AlertTriangle,
  CheckCircle2,
  Filter,
  Search,
  ArrowRight,
  Database,
  Layers,
  Sparkles,
} from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { AnomalySummary, Dataset, AnomalyPoint } from '../../types';
import { api } from '../../services/api';

interface AnomalyRadarViewProps {
  dataset: Dataset | null;
  anomalies: AnomalySummary | null;
  onUpdateAnomalies: (anomalies: AnomalySummary) => void;
  onNavigate: (view: string) => void;
}

export const AnomalyRadarView: React.FC<AnomalyRadarViewProps> = ({
  dataset,
  anomalies,
  onUpdateAnomalies,
  onNavigate,
}) => {
  const [selectedPoint, setSelectedPoint] = useState<AnomalyPoint | null>(null);
  const [contamination, setContamination] = useState(0.05);
  const [isDetecting, setIsDetecting] = useState(false);

  if (!dataset) {
    return (
      <div className="p-8 max-w-xl mx-auto text-center space-y-4 my-16">
        <Radio className="w-12 h-12 text-slate-500 mx-auto" />
        <h2 className="text-lg font-bold text-white">No Dataset for Anomaly Radar</h2>
        <p className="text-xs text-slate-400">
          Load a dataset to isolate multidimensional outliers and generate the anomaly radar.
        </p>
        <button
          onClick={() => onNavigate('dataset')}
          className="px-4 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
        >
          Open Dataset Lab
        </button>
      </div>
    );
  }

  const handleRecalculate = async (rate: number) => {
    setContamination(rate);
    setIsDetecting(true);
    try {
      const updated = await api.runAnomalyDetection(dataset, rate);
      onUpdateAnomalies(updated);
    } catch (err) {
      console.error('Anomaly detection error:', err);
    } finally {
      setIsDetecting(false);
    }
  };

  const currentAnomalies = anomalies || {
    totalRecords: dataset.rowCount,
    normalCount: dataset.rowCount,
    anomalyCount: 0,
    anomalyRate: 0,
    threshold: 0,
    anomalies: [],
  };

  const normalPoints = currentAnomalies.anomalies.filter((a) => !a.isAnomaly);
  const outlierPoints = currentAnomalies.anomalies.filter((a) => a.isAnomaly);

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Unsupervised Isolation Forest
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">Anomaly Radar</h1>
        </div>

        {/* Contamination Slider */}
        <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 px-3 py-2 rounded-lg text-xs">
          <span className="text-slate-400">Contamination Rate:</span>
          <span className="text-cyan-400 font-mono font-semibold tabular-nums">
            {(contamination * 100).toFixed(0)}%
          </span>
          <input
            type="range"
            min="0.01"
            max="0.15"
            step="0.01"
            value={contamination}
            onChange={(e) => handleRecalculate(parseFloat(e.target.value))}
            className="w-24 accent-cyan-400 bg-slate-800 h-1 rounded appearance-none cursor-pointer"
          />
        </div>
      </div>

      {/* KPI Stats (Section 17: Total Records, Normal Records, Anomalies, Anomaly Rate) */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400">Total Records</span>
          <span className="text-2xl font-bold text-white block tabular-nums">
            {currentAnomalies.totalRecords.toLocaleString()}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400">Normal Instances</span>
          <span className="text-2xl font-bold text-emerald-400 block tabular-nums">
            {currentAnomalies.normalCount.toLocaleString()}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400">Flagged Anomalies</span>
          <span className="text-2xl font-bold text-rose-400 block tabular-nums">
            {currentAnomalies.anomalyCount}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400">Contamination Ratio</span>
          <span className="text-2xl font-bold text-amber-400 block tabular-nums">
            {currentAnomalies.anomalyRate}%
          </span>
        </div>
      </div>

      {/* Radar Scatter Plot & Detail Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Radar Scatter Plot */}
        <div className="lg:col-span-8 p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">2D Isolation Radar Projection</h3>
              <p className="text-xs text-slate-400">
                Outer orbit instances indicate severe statistical path length deviations
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-500" />
                <span className="text-slate-400">Nominal</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                <span className="text-slate-300 font-semibold">Anomaly</span>
              </div>
            </div>
          </div>

          <div className="h-80 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <XAxis type="number" dataKey="projectedX" tick={false} axisLine={false} domain={[-6, 6]} />
                <YAxis type="number" dataKey="projectedY" tick={false} axisLine={false} domain={[-6, 6]} />
                <Tooltip
                  content={({ payload }) => {
                    if (payload && payload.length) {
                      const d = payload[0].payload as AnomalyPoint;
                      return (
                        <div className="p-3 bg-slate-950 border border-slate-700 rounded-lg text-xs space-y-1">
                          <p className="font-bold text-white">Record ID: {d.id}</p>
                          <p className={d.isAnomaly ? 'text-rose-400 font-semibold' : 'text-emerald-400'}>
                            {d.isAnomaly ? '🚨 Outlier Detected' : 'Nominal Sample'}
                          </p>
                          <p className="text-slate-400 font-mono">
                            Isolation Score: {d.anomalyScore}
                          </p>
                          <p className="text-[10px] text-cyan-400">Click point to inspect full record</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter
                  name="Nominal"
                  data={normalPoints}
                  fill="#38bdf8"
                  opacity={0.4}
                  onClick={(e) => setSelectedPoint(e as any)}
                />
                <Scatter
                  name="Anomalies"
                  data={outlierPoints}
                  fill="#f43f5e"
                  onClick={(e) => setSelectedPoint(e as any)}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Selected Anomaly Inspection */}
        <div className="lg:col-span-4 p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">Instance Inspector</h3>
            <p className="text-xs text-slate-400">
              {selectedPoint ? `Audit report for ${selectedPoint.id}` : 'Select a point on the radar to inspect'}
            </p>

            {selectedPoint ? (
              <div className="space-y-4 pt-4">
                <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Classification</span>
                    <span
                      className={`font-semibold ${
                        selectedPoint.isAnomaly ? 'text-rose-400' : 'text-emerald-400'
                      }`}
                    >
                      {selectedPoint.isAnomaly ? 'ANOMALOUS' : 'NOMINAL'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Isolation Score</span>
                    <span className="font-mono text-cyan-300 tabular-nums">
                      {selectedPoint.anomalyScore}
                    </span>
                  </div>
                </div>

                {/* Top Contributing Deviations */}
                {selectedPoint.topContributingFactors.length > 0 && (
                  <div className="space-y-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                      Primary Driving Outliers
                    </span>
                    <div className="space-y-1.5">
                      {selectedPoint.topContributingFactors.map((f) => (
                        <div
                          key={f.feature}
                          className="flex items-center justify-between text-xs p-2 bg-slate-950 rounded border border-slate-800"
                        >
                          <span className="text-slate-300">{f.feature}</span>
                          <span className="text-rose-400 font-mono tabular-nums">
                            +{f.deviation}σ deviation
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Raw Feature Values Snapshot */}
                <div className="space-y-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Feature Snapshot
                  </span>
                  <div className="p-3 bg-slate-950 rounded border border-slate-800 max-h-36 overflow-y-auto font-mono text-[11px] space-y-1 text-slate-300">
                    {Object.entries(selectedPoint.features).map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <span className="text-slate-400 truncate pr-2">{k}:</span>
                        <span className="text-white truncate">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-xs text-slate-500">
                Click any anomaly point to inspect root-cause feature deviations.
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-slate-800 text-[11px] text-slate-400">
            Isolation Forest trees: 100 · Subsample size: 256
          </div>
        </div>
      </div>

      {/* Flagged Anomalies Table */}
      <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
        <h3 className="text-sm font-semibold text-white">Flagged Anomaly Register</h3>
        <div className="overflow-x-auto max-h-60 border border-slate-800 rounded-lg">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="sticky top-0 bg-slate-950 text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Record ID</th>
                <th className="py-2.5 px-3">Isolation Score</th>
                <th className="py-2.5 px-3">Top Deviation Factor</th>
                <th className="py-2.5 px-3">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
              {outlierPoints.map((pt) => (
                <tr key={pt.id} className="hover:bg-slate-800/40">
                  <td className="py-2 px-3 font-mono font-medium text-white">{pt.id}</td>
                  <td className="py-2 px-3 font-mono text-rose-400 tabular-nums">
                    {pt.anomalyScore}
                  </td>
                  <td className="py-2 px-3 text-slate-300">
                    {pt.topContributingFactors[0]?.feature
                      ? `${pt.topContributingFactors[0].feature} (+${pt.topContributingFactors[0].deviation}σ)`
                      : 'Multivariate distance'}
                  </td>
                  <td className="py-2 px-3">
                    <button
                      onClick={() => setSelectedPoint(pt)}
                      className="text-cyan-400 hover:text-cyan-300 text-[11px]"
                    >
                      Inspect Record &rarr;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
