'use client';

import { careers } from '@/data/roadmap';

export default function Career() {
  return (
    <section id="career" className="py-20 px-4 bg-slate-900/50">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-4xl font-bold text-center mb-4">
          <span className="text-gradient">경력 사항</span>
        </h2>
        <p className="text-slate-400 text-center mb-12">7년+ 백엔드 개발 경험</p>

        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-0 md:left-1/2 transform md:-translate-x-px top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary-500 via-cyan-500 to-emerald-500" />

          {careers.map((career, index) => (
            <div
              key={career.company}
              className={`relative flex items-center mb-12 ${
                index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'
              }`}
            >
              {/* Timeline dot */}
              <div className="absolute left-0 md:left-1/2 transform -translate-x-1/2 w-4 h-4 rounded-full bg-primary-500 border-4 border-slate-900 z-10" />

              {/* Content */}
              <div className={`ml-8 md:ml-0 md:w-1/2 ${index % 2 === 0 ? 'md:pr-12' : 'md:pl-12'}`}>
                <div className="card-gradient border-gradient rounded-xl p-6 hover:border-primary-500/50 transition-colors">
                  <div className="flex flex-wrap items-start justify-between gap-2 mb-3">
                    <h3 className="text-xl font-bold text-white">{career.company}</h3>
                    <span className="px-3 py-1 rounded-lg bg-primary-500/20 text-primary-400 text-sm">
                      {career.role}
                    </span>
                  </div>
                  <p className="text-slate-500 text-sm mb-3">{career.period}</p>
                  <p className="text-slate-300 mb-4">{career.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {career.skills.map((skill) => (
                      <span
                        key={skill}
                        className="px-2 py-1 rounded bg-slate-800 text-slate-400 text-xs"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
