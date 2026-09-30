import React from 'react';
import {
  LayoutDashboard,
  Database,
  Cpu,
  GitFork,
  Trophy,
  BrainCircuit,
  Sliders,
  Radio,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { Dataset, CandidateModel } from '../../types';

interface SidebarProps {
  currentView: string;
  onNavigate: (view: string) => void;
  dataset: Dataset | null;
  models: CandidateModel[];
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onNavigate,
  dataset,
  models,
  isCollapsed,
  onToggleCollapse,
}) => {
  const menuItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'dataset', label: 'Dataset Lab', icon: Database, badge: dataset ? `${dataset.rowCount} rows` : undefined },
    { id: 'automl', label: 'AutoML', icon: Cpu },
    { id: 'optimization', label: 'Optimization', icon: GitFork },
    { id: 'models', label: 'Model Arena', icon: Trophy, badge: models.length > 0 ? `${models.length} models` : undefined },
    { id: 'explain', label: 'Explain AI', icon: BrainCircuit },
    { id: 'predictions', label: 'Predictions', icon: Sliders },
    { id: 'anomalies', label: 'Anomaly Radar', icon: Radio },
    { id: 'reports', label: 'Intelligence Report', icon: FileText },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside
      className={`relative z-30 flex flex-col bg-slate-950/90 border-r border-slate-800/80 transition-all duration-300 ease-in-out shrink-0 ${
        isCollapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Platform subtitle & toggle button */}
      <div className="flex items-center justify-between p-4 border-b border-slate-800/80">
        {!isCollapsed && (
          <div className="truncate">
            <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
              Adaptive ML Engine
            </span>
            <p className="text-[11px] text-slate-400 truncate">
              {dataset ? dataset.name : 'No dataset loaded'}
            </p>
          </div>
        )}
        <button
          onClick={onToggleCollapse}
          className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors mx-auto"
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation items */}
      <div className="flex-1 py-3 px-2 space-y-1 overflow-y-auto">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              title={isCollapsed ? item.label : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                isActive
                  ? 'bg-slate-800/90 text-cyan-300 border border-cyan-500/20 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
              {!isCollapsed && (
                <div className="flex-1 flex items-center justify-between text-left truncate">
                  <span className="truncate">{item.label}</span>
                  {item.badge && (
                    <span className="text-[10px] text-slate-400 tabular-nums">
                      {item.badge}
                    </span>
                  )}
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Quick context info at sidebar footer (Zero pill, clean unboxed text) */}
      {!isCollapsed && (
        <div className="p-3 border-t border-slate-800/80 text-[11px] text-slate-400 space-y-1">
          <div className="flex items-center justify-between">
            <span>Engine</span>
            <span className="text-slate-300">FastAPI & Pure JS</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Optimization</span>
            <span className="text-slate-300">Pareto & Genetic</span>
          </div>
        </div>
      )}
    </aside>
  );
};
