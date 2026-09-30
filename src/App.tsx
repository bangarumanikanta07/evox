import React, { useState, useEffect } from 'react';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { LandingHero } from './components/landing/LandingHero';
import { OverviewView } from './components/views/OverviewView';
import { DatasetLabView } from './components/views/DatasetLabView';
import { AutoMLView } from './components/views/AutoMLView';
import { ModelArenaView } from './components/views/ModelArenaView';
import { OptimizationView } from './components/views/OptimizationView';
import { ExplainView } from './components/views/ExplainView';
import { PredictionStudioView } from './components/views/PredictionStudioView';
import { AnomalyRadarView } from './components/views/AnomalyRadarView';
import { ReportView } from './components/views/ReportView';
import { SettingsView } from './components/views/SettingsView';
import { PresentationModeModal } from './components/common/PresentationModeModal';
import {
  Dataset,
  CandidateModel,
  AnomalySummary,
  EvolutionaryOptimizationRun,
  IntelligenceReport,
} from './types';
import { api } from './services/api';

export default function App() {
  const [currentView, setCurrentView] = useState<string>('landing');
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [models, setModels] = useState<CandidateModel[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalySummary | null>(null);
  const [evolutionRun, setEvolutionRun] = useState<EvolutionaryOptimizationRun | null>(null);
  const [report, setReport] = useState<IntelligenceReport | null>(null);
  const [selectedModel, setSelectedModel] = useState<CandidateModel | null>(null);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isPresentationOpen, setIsPresentationOpen] = useState(false);
  const [isDemoLoading, setIsDemoLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState<string>('ONLINE');

  // Health check on boot
  useEffect(() => {
    api.checkHealth().then((res) => {
      setSystemStatus(res.status || 'ONLINE');
    });
  }, []);

  // Demo Workflow (Sections 24 & 25)
  const handleLaunchDemo = async () => {
    setIsDemoLoading(true);
    try {
      // 1. Fetch benchmark dataset
      const demoData = await api.getDemoDataset();
      setDataset(demoData);

      // 2. Train AutoML candidate models
      const trainedModels = await api.trainModels(
        demoData,
        demoData.problemDetection.detectedProblem,
        demoData.targetColumn || 'churn_status'
      );
      setModels(trainedModels);
      setSelectedModel(trainedModels[0] || null);

      // 3. Run evolutionary optimization
      const evo = await api.runOptimization(demoData.id, 10);
      setEvolutionRun(evo);

      // 4. Run anomaly detection
      const anomalySumm = await api.runAnomalyDetection(demoData, 0.05);
      setAnomalies(anomalySumm);

      // 5. Generate intelligence report
      const rep = await api.generateReport(demoData, trainedModels, anomalySumm);
      setReport(rep);

      // Navigate to overview
      setCurrentView('overview');
    } catch (err) {
      console.error('Demo pipeline error:', err);
    } finally {
      setIsDemoLoading(false);
    }
  };

  const handleClearSession = () => {
    setDataset(null);
    setModels([]);
    setAnomalies(null);
    setEvolutionRun(null);
    setReport(null);
    setSelectedModel(null);
    setCurrentView('landing');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Header */}
      <Header
        currentView={currentView}
        onNavigate={(v) => setCurrentView(v)}
        onLaunchDemo={handleLaunchDemo}
        onOpenPresentation={() => setIsPresentationOpen(true)}
        isDemoLoading={isDemoLoading}
        systemStatus={systemStatus}
      />

      {/* Main View Area */}
      {currentView === 'landing' ? (
        <main className="flex-1">
          <LandingHero
            onLaunchLab={() => setCurrentView('dataset')}
            onLaunchDemo={handleLaunchDemo}
            onExploreWorkflow={() => setIsPresentationOpen(true)}
            isDemoLoading={isDemoLoading}
          />
        </main>
      ) : (
        <div className="flex-1 flex overflow-hidden">
          {/* Collapsible Sidebar */}
          <Sidebar
            currentView={currentView}
            onNavigate={(v) => setCurrentView(v)}
            dataset={dataset}
            models={models}
            isCollapsed={isSidebarCollapsed}
            onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          />

          {/* Dynamic View Content */}
          <main className="flex-1 overflow-y-auto bg-slate-950/50">
            {currentView === 'overview' && (
              <OverviewView
                dataset={dataset}
                models={models}
                anomalies={anomalies}
                evolutionRun={evolutionRun}
                onNavigate={(v) => setCurrentView(v)}
                onLaunchDemo={handleLaunchDemo}
                isDemoLoading={isDemoLoading}
              />
            )}

            {currentView === 'dataset' && (
              <DatasetLabView
                dataset={dataset}
                onDatasetLoaded={(ds) => {
                  setDataset(ds);
                  // Reset downstream models
                  setModels([]);
                  setAnomalies(null);
                  setEvolutionRun(null);
                  setReport(null);
                }}
                onNavigate={(v) => setCurrentView(v)}
                onLaunchDemo={handleLaunchDemo}
                isDemoLoading={isDemoLoading}
              />
            )}

            {currentView === 'automl' && (
              <AutoMLView
                dataset={dataset}
                models={models}
                onModelsTrained={(ms) => {
                  setModels(ms);
                  setSelectedModel(ms[0] || null);
                }}
                onNavigate={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'models' && (
              <ModelArenaView
                models={models}
                dataset={dataset}
                onSelectModelForPrediction={(m) => setSelectedModel(m)}
                onNavigate={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'optimization' && (
              <OptimizationView
                dataset={dataset}
                models={models}
                evolutionRun={evolutionRun}
                onUpdateEvolutionRun={(run) => setEvolutionRun(run)}
                onNavigate={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'explain' && (
              <ExplainView
                dataset={dataset}
                models={models}
                onNavigate={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'predictions' && (
              <PredictionStudioView
                dataset={dataset}
                models={models}
                selectedModel={selectedModel}
                onNavigate={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'anomalies' && (
              <AnomalyRadarView
                dataset={dataset}
                anomalies={anomalies}
                onUpdateAnomalies={(anom) => setAnomalies(anom)}
                onNavigate={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'reports' && (
              <ReportView
                dataset={dataset}
                models={models}
                anomalies={anomalies}
                report={report}
                onUpdateReport={(r) => setReport(r)}
                onNavigate={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'settings' && (
              <SettingsView onClearSession={handleClearSession} />
            )}
          </main>
        </div>
      )}

      {/* Presentation Mode Modal (Section 41) */}
      <PresentationModeModal
        isOpen={isPresentationOpen}
        onClose={() => setIsPresentationOpen(false)}
        onNavigateToView={(v) => setCurrentView(v)}
        dataset={dataset}
        models={models}
        anomalies={anomalies}
        report={report}
      />
    </div>
  );
}
