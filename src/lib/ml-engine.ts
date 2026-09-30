import {
  Dataset,
  CandidateModel,
  ObjectiveWeights,
  ParetoPoint,
} from '../types';

/**
 * Multi-objective weighted score calculator for user UI slider adjustments.
 * Operates directly on genuine model metrics returned by the Python backend.
 */
export function calculateWeightedScores(
  models: CandidateModel[],
  weights: ObjectiveWeights
): { modelId: string; compositeScore: number; rank: number }[] {
  if (!models || models.length === 0) return [];

  const perfs = models.map((m) => (m.problemType === 'classification' ? m.metrics.accuracy || 0 : m.metrics.r2 || 0));
  const lats = models.map((m) => m.metrics.inferenceLatencyMs);
  const comps = models.map((m) => m.metrics.modelComplexityScore);

  const minPerf = Math.min(...perfs);
  const maxPerf = Math.max(...perfs);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minComp = Math.min(...comps);
  const maxComp = Math.max(...comps);

  const totalWeight = (weights.performance + weights.latency + weights.simplicity + weights.robustness) || 1;
  const wPerf = weights.performance / totalWeight;
  const wLat = weights.latency / totalWeight;
  const wSimp = weights.simplicity / totalWeight;
  const wRob = weights.robustness / totalWeight;

  const scored = models.map((m, idx) => {
    const rawPerf = perfs[idx];
    const rawLat = lats[idx];
    const rawComp = comps[idx];

    // Normalized 0 to 1
    const normPerf = maxPerf > minPerf ? (rawPerf - minPerf) / (maxPerf - minPerf) : 1;
    const normSpeed = maxLat > minLat ? 1 - (rawLat - minLat) / (maxLat - minLat) : 1;
    const normSimp = maxComp > minComp ? 1 - (rawComp - minComp) / (maxComp - minComp) : 1;
    const normRobust = m.isParetoOptimal ? 1.0 : 0.7;

    const composite = wPerf * normPerf + wLat * normSpeed + wSimp * normSimp + wRob * normRobust;
    return {
      modelId: m.id,
      compositeScore: Math.round(composite * 1000) / 1000,
      rank: 1,
    };
  });

  scored.sort((a, b) => b.compositeScore - a.compositeScore);
  scored.forEach((item, idx) => {
    item.rank = idx + 1;
  });

  return scored;
}
