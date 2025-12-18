'use client';

import { useState, useEffect } from 'react';
import { roadmapSteps, Step, ChecklistItem } from '@/data/roadmap';

interface ProgressState {
  [stepId: number]: {
    [checklistId: string]: boolean;
  };
}

export default function Roadmap() {
  const [activeStep, setActiveStep] = useState<number>(1);
  const [progress, setProgress] = useState<ProgressState>({});

  useEffect(() => {
    const saved = localStorage.getItem('roadmap-progress');
    if (saved) {
      setProgress(JSON.parse(saved));
    }
  }, []);

  const saveProgress = (newProgress: ProgressState) => {
    setProgress(newProgress);
    localStorage.setItem('roadmap-progress', JSON.stringify(newProgress));
  };

  const toggleChecklist = (stepId: number, checklistId: string) => {
    const newProgress = {
      ...progress,
      [stepId]: {
        ...progress[stepId],
        [checklistId]: !progress[stepId]?.[checklistId],
      },
    };
    saveProgress(newProgress);
  };

  const getStepProgress = (step: Step) => {
    const stepProgress = progress[step.id] || {};
    const completed = step.checklist.filter((item) => stepProgress[item.id]).length;
    return Math.round((completed / step.checklist.length) * 100);
  };

  const getTotalProgress = () => {
    let totalItems = 0;
    let completedItems = 0;
    roadmapSteps.forEach((step) => {
      totalItems += step.checklist.length;
      const stepProgress = progress[step.id] || {};
      completedItems += step.checklist.filter((item) => stepProgress[item.id]).length;
    });
    return totalItems > 0 ? Math.round((completedItems / totalItems) * 100) : 0;
  };

  return (
    <section id="roadmap" className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold text-center mb-4">
          <span className="text-gradient">AI/ML Engineer 로드맵</span>
        </h2>
        <p className="text-slate-400 text-center mb-8">18개월 성장 계획</p>

        {/* Total Progress */}
        <div className="card-gradient border-gradient rounded-xl p-6 mb-12">
          <div className="flex justify-between items-center mb-3">
            <span className="text-lg font-medium">전체 진행률</span>
            <span className="text-2xl font-bold text-primary-400">{getTotalProgress()}%</span>
          </div>
          <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-cyan-500 transition-all duration-500"
              style={{ width: `${getTotalProgress()}%` }}
            />
          </div>
        </div>

        {/* Step Navigation */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {roadmapSteps.map((step) => (
            <button
              key={step.id}
              onClick={() => setActiveStep(step.id)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeStep === step.id
                  ? 'bg-primary-500 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              Step {step.id}
              {getStepProgress(step) === 100 && (
                <span className="ml-2 text-emerald-400">✓</span>
              )}
            </button>
          ))}
        </div>

        {/* Active Step Detail */}
        {roadmapSteps.map((step) => (
          <div
            key={step.id}
            className={`${activeStep === step.id ? 'block' : 'hidden'}`}
          >
            <div className="card-gradient border-gradient rounded-2xl p-8">
              <div className="flex flex-wrap items-start justify-between gap-4 mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-white mb-2">
                    Step {step.id}: {step.title}
                  </h3>
                  <p className="text-slate-400">{step.description}</p>
                </div>
                <div className="flex items-center gap-4">
                  <span className="px-3 py-1 rounded-lg bg-cyan-500/20 text-cyan-400">
                    {step.duration}
                  </span>
                  <span className="text-2xl font-bold text-primary-400">
                    {getStepProgress(step)}%
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="h-2 bg-slate-700 rounded-full overflow-hidden mb-8">
                <div
                  className="h-full bg-gradient-to-r from-primary-500 to-emerald-500 transition-all duration-500"
                  style={{ width: `${getStepProgress(step)}%` }}
                />
              </div>

              <div className="grid md:grid-cols-2 gap-8">
                {/* Checklist */}
                <div>
                  <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    학습 체크리스트
                  </h4>
                  <div className="space-y-3">
                    {step.checklist.map((item) => (
                      <label
                        key={item.id}
                        className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 cursor-pointer transition-colors"
                      >
                        <input
                          type="checkbox"
                          checked={progress[step.id]?.[item.id] || false}
                          onChange={() => toggleChecklist(step.id, item.id)}
                          className="w-5 h-5 rounded border-slate-600 text-primary-500 focus:ring-primary-500 bg-slate-700"
                        />
                        <span className={`${progress[step.id]?.[item.id] ? 'text-slate-500 line-through' : 'text-slate-300'}`}>
                          {item.title}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Skills & Resources */}
                <div className="space-y-6">
                  {/* Skills */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      습득 기술
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {step.skills.map((skill) => (
                        <span
                          key={skill}
                          className="px-3 py-1 rounded-lg bg-cyan-500/20 text-cyan-400 text-sm"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Resources */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                      </svg>
                      학습 리소스
                    </h4>
                    <div className="space-y-2">
                      {step.resources.map((resource) => (
                        <a
                          key={resource.name}
                          href={resource.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors"
                        >
                          <span className="text-slate-300">{resource.name}</span>
                          <span className={`px-2 py-0.5 rounded text-xs ${
                            resource.type === 'free'
                              ? 'bg-emerald-500/20 text-emerald-400'
                              : 'bg-amber-500/20 text-amber-400'
                          }`}>
                            {resource.type === 'free' ? '무료' : '유료'}
                          </span>
                        </a>
                      ))}
                    </div>
                  </div>

                  {/* Projects */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                      실습 프로젝트
                    </h4>
                    <div className="space-y-2">
                      {step.projects.map((project) => (
                        <div
                          key={project}
                          className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-200"
                        >
                          {project}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
