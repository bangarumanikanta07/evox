export type ProblemType = 'classification' | 'regression' | 'clustering' | 'anomaly_detection';

export type DataType = 'numerical' | 'categorical' | 'datetime' | 'text' | 'boolean' | 'unknown';

export interface ColumnProfile {
  name: string;
  dataType: DataType;
  missingCount: number;
  missingPercentage: number;
  uniqueCount: number;
  isPotentialTarget: boolean;
  stats?: {
    min?: number;
    max?: number;
    mean?: number;
    median?: number;
    stdDev?: number;
    topCategories?: { value: string; count: number }[];
  };
  outlierCount: number;
}

export interface DataQualityReport {
  score: number; // 0 - 100
  completeness: number;
  validity: number;
  uniqueness: number;
  consistency: number;
  reasons: string[];
  recommendations: string[];
}

export interface ProblemDetectionResult {
  detectedProblem: ProblemType;
  confidence: number; // 0 - 100
  reason: string;
  suggestedTarget: string | null;
  classesCount?: number;
}

export interface Dataset {
  id: string;
  name: string;
  rowCount: number;
  columnCount: number;
  fileSize: number;
  uploadedAt: string;
  columns: ColumnProfile[];
  previewRows: Record<string, any>[];
  qualityReport: DataQualityReport;
  problemDetection: ProblemDetectionResult;
  targetColumn: string | null;
  features: string[];
}

export interface PreprocessingStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'skipped';
  details: string;
}

export interface PreprocessingPipeline {
  datasetId: string;
  targetColumn: string;
  selectedFeatures: string[];
  missingValueStrategy: 'impute_mean' | 'impute_median' | 'impute_mode' | 'drop';
  scalingMethod: 'standard' | 'minmax' | 'robust' | 'none';
  encodingMethod: 'onehot' | 'label' | 'target';
  handleOutliers: boolean;
  trainTestSplit: number; // e.g. 0.8
  steps: PreprocessingStep[];
  processedRowCount: number;
  processedFeatureCount: number;
}

export interface ModelMetrics {
  // Classification
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  rocAuc?: number;
  confusionMatrix?: {
    labels: string[];
    matrix: number[][];
  };

  // Regression
  mae?: number;
  mse?: number;
  rmse?: number;
  r2?: number;

  // Resource metrics
  trainingTimeMs: number;
  inferenceLatencyMs: number;
  modelComplexityScore: number; // 1 to 10
  parameterCount: number;
}

export interface CandidateModel {
  id: string;
  name: string;
  algorithm: string;
  problemType: ProblemType;
  status: 'trained' | 'evaluating' | 'failed' | 'optimized';
  metrics: ModelMetrics;
  hyperparameters: Record<string, any>;
  isParetoOptimal: boolean;
  paretoRank?: number;
  compositeScore: number;
  featureImportances: { feature: string; importance: number }[];
  validatedAt: string;
}

export interface ObjectiveWeights {
  performance: number; // 0 - 1
  latency: number;     // 0 - 1
  simplicity: number;  // 0 - 1
  robustness: number;  // 0 - 1
}

export interface ParetoPoint {
  modelId: string;
  modelName: string;
  performance: number; // e.g. Accuracy or R^2 (0-1)
  latency: number;     // ms
  complexity: number;  // 1-10
  isParetoOptimal: boolean;
  score: number;
}

export interface EvolutionaryGeneration {
  generation: number;
  bestFitness: number;
  avgFitness: number;
  diversity: number;
  bestCandidateName: string;
  populationSize: number;
  mutationRate: number;
  crossoverRate: number;
}

export interface EvolutionaryOptimizationRun {
  id: string;
  datasetId: string;
  generations: EvolutionaryGeneration[];
  currentGeneration: number;
  totalGenerations: number;
  status: 'idle' | 'running' | 'completed';
  initialFitness: number;
  bestFitness: number;
  finalFitness: number;
  optimalHyperparameters: Record<string, any>;
}

export interface FeatureExplanation {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative' | 'neutral';
  meanValue: number;
  stdDev: number;
}

export interface SinglePredictionExplanation {
  prediction: string | number;
  probability?: number;
  classProbabilities?: { label: string; prob: number }[];
  baseValue: number;
  contributingFactors: {
    feature: string;
    value: any;
    attribution: number;
    impact: 'increases_score' | 'decreases_score';
  }[];
  humanExplanation: string;
  timestamp: string;
}

export interface AnomalyPoint {
  id: string | number;
  recordIndex: number;
  features: Record<string, any>;
  anomalyScore: number; // -1 to 1 (negative = more anomalous)
  isAnomaly: boolean;
  projectedX: number;
  projectedY: number;
  topContributingFactors: { feature: string; deviation: number }[];
}

export interface AnomalySummary {
  totalRecords: number;
  normalCount: number;
  anomalyCount: number;
  anomalyRate: number; // percentage
  threshold: number;
  anomalies: AnomalyPoint[];
}

export interface IntelligenceReport {
  id: string;
  generatedAt: string;
  datasetSummary: {
    name: string;
    rows: number;
    columns: number;
    qualityScore: number;
    detectedProblem: ProblemType;
  };
  dataQualityFindings: string[];
  modelsEvaluated: {
    total: number;
    bestModelName: string;
    bestMetricLabel: string;
    bestMetricValue: number;
  };
  paretoWinners: {
    balanced: string;
    maxPerformance: string;
    ultraFast: string;
  };
  topFeatures: { feature: string; importance: number }[];
  anomalySummary: {
    detected: number;
    rate: number;
  };
  aiNarrative: string;
  recommendations: string[];
}
