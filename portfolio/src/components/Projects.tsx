'use client';

import { portfolioProjects } from '@/data/roadmap';

const statusColors = {
  planned: { bg: 'bg-amber-500/20', text: 'text-amber-400', label: '계획됨' },
  'in-progress': { bg: 'bg-cyan-500/20', text: 'text-cyan-400', label: '진행중' },
  completed: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', label: '완료' },
};

export default function Projects() {
  return (
    <section id="projects" className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold text-center mb-4">
          <span className="text-gradient">포트폴리오 프로젝트</span>
        </h2>
        <p className="text-slate-400 text-center mb-12">AI/ML 역량을 증명할 핵심 프로젝트</p>

        <div className="grid md:grid-cols-2 gap-6">
          {portfolioProjects.map((project, index) => {
            const status = statusColors[project.status as keyof typeof statusColors];
            return (
              <div
                key={project.id}
                className="card-gradient border-gradient rounded-2xl p-6 hover:border-primary-500/50 transition-all group"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <span className="w-10 h-10 rounded-lg bg-primary-500/20 flex items-center justify-center text-primary-400 font-bold">
                      #{project.id}
                    </span>
                    <h3 className="text-xl font-bold text-white group-hover:text-primary-400 transition-colors">
                      {project.title}
                    </h3>
                  </div>
                  <span className={`px-3 py-1 rounded-lg ${status.bg} ${status.text} text-sm`}>
                    {status.label}
                  </span>
                </div>

                <p className="text-slate-400 mb-4">{project.description}</p>

                {/* Tech Stack */}
                <div className="mb-4">
                  <div className="flex flex-wrap gap-2">
                    {project.tech.map((tech) => (
                      <span
                        key={tech}
                        className="px-2 py-1 rounded bg-slate-800 text-cyan-400 text-xs"
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Features */}
                <div className="border-t border-slate-700/50 pt-4">
                  <h4 className="text-sm font-medium text-slate-500 mb-2">주요 기능</h4>
                  <div className="flex flex-wrap gap-2">
                    {project.features.map((feature) => (
                      <span
                        key={feature}
                        className="px-2 py-1 rounded bg-emerald-500/10 text-emerald-400 text-xs"
                      >
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
