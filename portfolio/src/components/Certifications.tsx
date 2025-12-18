'use client';

import { certifications } from '@/data/roadmap';

const difficultyColors = {
  '하': { bg: 'bg-emerald-500/20', text: 'text-emerald-400' },
  '중': { bg: 'bg-cyan-500/20', text: 'text-cyan-400' },
  '중상': { bg: 'bg-amber-500/20', text: 'text-amber-400' },
  '상': { bg: 'bg-red-500/20', text: 'text-red-400' },
};

export default function Certifications() {
  return (
    <section id="certifications" className="py-20 px-4 bg-slate-900/50">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-4xl font-bold text-center mb-4">
          <span className="text-gradient">자격증 로드맵</span>
        </h2>
        <p className="text-slate-400 text-center mb-12">커리어 성장을 위한 인증 계획</p>

        <div className="space-y-4">
          {certifications.map((cert, index) => {
            const difficulty = difficultyColors[cert.difficulty as keyof typeof difficultyColors];
            return (
              <div
                key={cert.name}
                className="card-gradient border-gradient rounded-xl p-6 hover:border-primary-500/50 transition-colors"
              >
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <span className="w-8 h-8 rounded-full bg-primary-500/20 flex items-center justify-center text-primary-400 font-bold text-sm">
                      {index + 1}
                    </span>
                    <div>
                      <h3 className="text-lg font-bold text-white">{cert.name}</h3>
                      <p className="text-slate-500 text-sm">목표: {cert.target}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1 rounded-lg ${difficulty.bg} ${difficulty.text} text-sm`}>
                      난이도: {cert.difficulty}
                    </span>
                    <span className="px-3 py-1 rounded-lg bg-slate-800 text-slate-400 text-sm">
                      {cert.cost}
                    </span>
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
