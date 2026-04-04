'use client';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { Cpu, Database, Cloud, Lock, Layers, GitBranch } from 'lucide-react';

const stack = [
  {
    layer: 'AI / ML Layer',
    color: '#E87722',
    icon: Cpu,
    items: ['XGBoost (R²=0.9998)', 'Random Forest (n=500)', 'LSTM 64→32→Dense', 'Isolation Forest (FPR<4%)', 'RL-PPO BESS Agent', 'XGBoost Classifier (AUC=0.87)'],
  },
  {
    layer: 'Physics Engine',
    color: '#00C4FF',
    icon: Layers,
    items: ['Spencer Solar Position', 'Bird Clear-Sky Model', 'NOCT Thermal Correction', 'Hay-Davies Diffuse Irradiance', 'Perez Transposition', 'CERC Grid Model'],
  },
  {
    layer: 'Data & Integration',
    color: '#00E5A0',
    icon: Database,
    items: ['Real-time SCADA Ingest', 'InfluxDB Time-series', 'IMD Weather API', 'IEX Market Feed', 'DISCOM Tariff DB', 'REST + WebSocket APIs'],
  },
  {
    layer: 'Platform & Infra',
    color: '#A78BFA',
    icon: Cloud,
    items: ['Streamlit Dashboard', 'FastAPI (14 endpoints)', 'Docker Orchestration', 'GitHub Actions CI/CD', 'Grafana Monitoring', 'Multi-tenant Auth'],
  },
];

const principles = [
  { icon: Lock, title: 'Enterprise Security', desc: 'SOC2-aligned, role-based access, encrypted data at rest and in transit.', color: '#E87722' },
  { icon: GitBranch, title: 'CI/CD Pipeline', desc: 'Automated testing, linting, and deployment on every commit to main.', color: '#00C4FF' },
  { icon: Cloud, title: 'Cloud Native', desc: 'Deploy anywhere — Render, AWS, Azure, or on-premise SCADA integration.', color: '#00E5A0' },
  { icon: Cpu, title: 'Edge Ready', desc: 'Lightweight ML inference deployable on-site for sub-100ms response times.', color: '#A78BFA' },
];

export default function Technology() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section id="technology" ref={ref} className="relative py-28 overflow-hidden">
      <div className="glow-orb-blue" style={{ width: '500px', height: '500px', top: '0', left: '-150px', opacity: 0.25 }} />
      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><Cpu size={14} /> Technology</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            Built on a <span className="gradient-text-blue">World-Class</span><br />AI Architecture
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            Seven layers of intelligence — from physics equations to production ML models — engineered for India's renewable energy grid.
          </p>
        </motion.div>

        {/* Stack grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16">
          {stack.map((s, i) => (
            <motion.div key={s.layer} initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: i * 0.1 }}
              className="glass-card p-7"
              style={{ borderTop: `2px solid ${s.color}` }}>
              <div className="flex items-center gap-3 mb-5">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ background: `${s.color}18` }}>
                  <s.icon size={20} style={{ color: s.color }} />
                </div>
                <h3 className="text-lg font-bold" style={{ fontFamily: 'Space Grotesk', color: s.color }}>{s.layer}</h3>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {s.items.map(item => (
                  <div key={item} className="flex items-center gap-2 p-2.5 rounded-lg"
                    style={{ background: `${s.color}08`, border: `1px solid ${s.color}15` }}>
                    <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: s.color }} />
                    <span className="text-xs font-medium" style={{ color: '#CBD5E1' }}>{item}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Architecture principles */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          {principles.map((p, i) => (
            <motion.div key={p.title} initial={{ opacity: 0, y: 20 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.5 + i * 0.08 }}
              className="glass-card-hover p-6 text-center">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4"
                style={{ background: `${p.color}15`, border: `1px solid ${p.color}30` }}>
                <p.icon size={22} style={{ color: p.color }} />
              </div>
              <h4 className="font-bold mb-2" style={{ fontFamily: 'Space Grotesk' }}>{p.title}</h4>
              <p className="text-xs leading-relaxed" style={{ color: '#64748B' }}>{p.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
