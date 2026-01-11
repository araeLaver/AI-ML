'use client';

import { profile } from '@/data/roadmap';

export default function Hero() {
  return (
    <section className="min-h-screen flex items-center justify-center px-4 py-20">
      <div className="max-w-4xl mx-auto text-center">
        <div className="mb-6">
          <span className="inline-block px-4 py-2 rounded-full bg-primary-500/20 text-primary-400 text-sm font-medium">
            Backend Developer → AI Platform Engineer
          </span>
        </div>

        <h1 className="text-5xl md:text-7xl font-bold mb-6">
          <span className="text-gradient">{profile.name}</span>
        </h1>

        <p className="text-xl md:text-2xl text-slate-300 mb-8">
          {profile.goal}
        </p>

        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {profile.currentSkills.map((skill) => (
            <span
              key={skill}
              className="px-3 py-1 rounded-lg bg-slate-800 text-slate-300 text-sm"
            >
              {skill}
            </span>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          <div className="card-gradient border-gradient rounded-xl p-6">
            <div className="text-3xl font-bold text-primary-400 mb-2">7+</div>
            <div className="text-slate-400">년 개발 경력</div>
          </div>
          <div className="card-gradient border-gradient rounded-xl p-6">
            <div className="text-3xl font-bold text-cyan-400 mb-2">18개월</div>
            <div className="text-slate-400">목표 타임라인</div>
          </div>
          <div className="card-gradient border-gradient rounded-xl p-6">
            <div className="text-3xl font-bold text-emerald-400 mb-2">2억+</div>
            <div className="text-slate-400">목표 연봉</div>
          </div>
        </div>

        <div className="mt-12">
          <a
            href="#roadmap"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-primary-500 hover:bg-primary-600 text-white font-medium transition-colors"
          >
            로드맵 보기
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </a>
        </div>
      </div>
    </section>
  );
}
