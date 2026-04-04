'use client';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { Sun, Wind, Battery, Cpu, Award, Users } from 'lucide-react';

const pillars = [
  { icon: Sun, title: 'Bhāskara', sub: 'The Sun', desc: 'Solar intelligence that harnesses every photon. Physics-driven models aligned with India\'s diverse irradiance zones.', color: '#E87722' },
  { icon: Wind, title: 'Akṣaya', sub: 'The Inexhaustible', desc: 'Boundless renewable potential realized through ML-powered wind forecasting and grid balancing algorithms.', color: '#00C4FF' },
  { icon: Battery, title: 'Śakti', sub: 'Energy & Power', desc: 'BESS dispatch optimization using Reinforcement Learning — store when cheap, sell when valuable.', color: '#00E5A0' },
  { icon: Cpu, title: 'Kriyā', sub: 'Action & Work', desc: 'Real-time SCADA-integrated AI actions: auto fault detection, cleaning alerts, and grid compliance triggers.', color: '#A78BFA' },
];

const awards = [
  { icon: Award, label: 'BCG Smart Energy Partner 2026' },
  { icon: Users, label: '50+ Utility Clients Across India' },
  { icon: Award, label: 'MNRE Certified Platform' },
  { icon: Award, label: 'Adani Energy Technology Alliance' },
];

export default function About() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });

  return (
    <section id="about" ref={ref} className="relative py-28 overflow-hidden">
      <div className="glow-orb-orange" style={{ width: '400px', height: '400px', top: '10%', right: '-150px', opacity: 0.3 }} />

      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><Sun size={14} /> About BASKER Energy</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-6" style={{ fontFamily: 'Space Grotesk' }}>
            The Name Behind the <span className="gradient-text">Mission</span>
          </h2>
          <p className="text-lg max-w-3xl mx-auto" style={{ color: '#94A3B8' }}>
            BASKER is more than a platform — it's a philosophy. Each letter embodies a principle of renewable energy transformation that drives India toward a net-zero future.
          </p>
        </motion.div>

        {/* Acronym pillars */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-20">
          {pillars.map((p, i) => (
            <motion.div key={p.title} initial={{ opacity: 0, y: 40 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: i * 0.12 }}
              className="glass-card-hover card-shine p-7 text-center">
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-5"
                style={{ background: `${p.color}18`, border: `1px solid ${p.color}35` }}>
                <p.icon size={26} style={{ color: p.color }} />
              </div>
              <h3 className="text-xl font-bold mb-1" style={{ fontFamily: 'Space Grotesk', color: p.color }}>{p.title}</h3>
              <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: '#64748B' }}>{p.sub}</p>
              <p className="text-sm leading-relaxed" style={{ color: '#94A3B8' }}>{p.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* Two-col layout: story + visual */}
        <div className="grid lg:grid-cols-2 gap-14 items-center">
          <motion.div initial={{ opacity: 0, x: -40 }} animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.7, delay: 0.3 }}>
            <div className="section-tag mb-6"><Cpu size={14} /> Our Story</div>
            <h3 className="text-3xl md:text-4xl font-bold mb-6" style={{ fontFamily: 'Space Grotesk' }}>
              Built by Practitioners,<br /><span className="gradient-text">for India's Grid</span>
            </h3>
            <p className="text-base mb-4 leading-relaxed" style={{ color: '#94A3B8' }}>
              BASKER Energy was born from a BCG Smart Energy engagement with Adani Energy — to solve real problems that India's renewable sector faces: poor performance ratios, reactive maintenance, and missed trading opportunities.
            </p>
            <p className="text-base mb-8 leading-relaxed" style={{ color: '#94A3B8' }}>
              Our platform is built on 15+ ML models trained on 100,000+ hours of Indian solar and wind data, aligned with CERC regulations, and tested across Gujarat, Rajasthan, Karnataka, and Andhra Pradesh.
            </p>
            <div className="grid grid-cols-2 gap-4">
              {awards.map((a, i) => (
                <div key={i} className="flex items-center gap-3 p-4 rounded-xl"
                  style={{ background: 'rgba(232,119,34,0.06)', border: '1px solid rgba(232,119,34,0.15)' }}>
                  <a.icon size={16} style={{ color: '#E87722', flexShrink: 0 }} />
                  <span className="text-xs font-medium" style={{ color: '#CBD5E1' }}>{a.label}</span>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, x: 40 }} animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="relative">
            <div className="glass-card p-8 relative overflow-hidden"
              style={{ background: 'linear-gradient(135deg, rgba(10,22,40,0.9), rgba(6,11,20,0.9))' }}>
              <div className="glow-orb-orange" style={{ width: '300px', height: '300px', top: '-50%', right: '-30%', opacity: 0.3 }} />
              <h4 className="text-xl font-bold mb-6 gradient-text" style={{ fontFamily: 'Space Grotesk' }}>Platform at a Glance</h4>
              {[
                { label: 'ML Model Accuracy (R²)', val: 99.9, color: '#E87722' },
                { label: 'Performance Ratio Improvement', val: 78, color: '#00C4FF' },
                { label: 'Fault Detection AUC', val: 87, color: '#00E5A0' },
                { label: 'Platform Uptime SLA', val: 99.8, color: '#A78BFA' },
              ].map(m => (
                <div key={m.label} className="mb-5">
                  <div className="flex justify-between text-sm mb-1.5">
                    <span style={{ color: '#94A3B8' }}>{m.label}</span>
                    <span className="font-semibold" style={{ color: m.color }}>{m.val}%</span>
                  </div>
                  <div className="progress-bar">
                    <motion.div className="progress-fill" style={{ background: `linear-gradient(90deg, ${m.color}, ${m.color}99)` }}
                      initial={{ width: 0 }} animate={inView ? { width: `${m.val}%` } : {}}
                      transition={{ duration: 1.5, delay: 0.6 }} />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
