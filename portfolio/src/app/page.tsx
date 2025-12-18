import {
  Navigation,
  Hero,
  Roadmap,
  Career,
  Projects,
  Certifications,
  Footer,
} from '@/components';

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navigation />
      <Hero />
      <Roadmap />
      <Career />
      <Projects />
      <Certifications />
      <Footer />
    </main>
  );
}
