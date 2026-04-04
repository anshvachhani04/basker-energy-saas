'use client';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { CheckCircle, XCircle, Star } from 'lucide-react';

const comparisons = [
  { feature: 'Physics-AI Hybrid Models', basker: true, legacy: false, saas: false },
  { feature: 'India-specific CERC Compliance', basker: true, legacy: false, saas: false },
  { feature: 'Real-time SCADA Integration', basker: true, legacy: true, saas: false },
  { feature: 'BESS Dispatch (RL-PPO)', basker: true, legacy: false, saas: false },
  { feature: 'ESG & REC Auto-reporting', basker: true, legacy: false, saas: true },
  { feature: 'Multi-Asset Fleet Dashboard', basker: true, legacy: true, saas: false },
  { feature: 'Energy Trading Signals', basker: true, legacy: false, saas: false },
  { feature: 'Sub-5min Update Latency', basker: true, legacy: false, saas: true },
];

const advantages = [
  { num: '7+', label: 'Specialized AI Models', sub: 'vs 1-2 in competitors', color: '#E87722' },
  { num: '40%', label: 'Faster Fault Response', sub: 'Predictive vs Reactive', color: '#00C4FF' },
  { num: '₹340Cr', label: 'Client Revenue Uplift', sub: 'Across pilot deployments', color: '#00E5A0' },
  { num: '3x', label: 'ROI vs Traditional SCADA', sub: 'Within 18 months', color: '#A78BFA' },
];

export default function WhyBasker() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });

  return (
    <section id="why-basker" ref={ref} className="relative py-28 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><Star size={14} /> Why BASKER</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            The <span className="gradient-text">Unfair Advantage</span><br />Every Plant Operator Needs
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            We built what no traditional SCADA or generic SaaS could — a platform engineered for India's unique energy grid, climate, and regulatory landscape.
          </p>
        </motion.div>

        {/* Advantages grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-16">
          {advantages.map((a, i) => (
            <motion.div key={a.label} initial={{ opacity: 0, scale: 0.9 }} animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ delay: i * 0.1 }}
              className="glass-card-hover text-center p-7">
              <div className="text-4xl font-black mb-2" style={{ color: a.color, fontFamily: 'Space Grotesk' }}>{a.num}</div>
              <div className="text-base font-semibold mb-1" style={{ color: '#E2E8F0' }}>{a.label}</div>
              <div className="text-xs" style={{ color: '#64748B' }}>{a.sub}</div>
            </motion.div>
          ))}
        </div>

        {/* Comparison table */}
        <motion.div initial={{ opacity: 0, y: 40 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="glass-card overflow-hidden">
          <div className="grid grid-cols-4 p-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
            <div className="text-sm font-semibold" style={{ color: '#64748B' }}>Feature</div>
            <div className="text-center">
              <span className="px-4 py-1.5 rounded-full text-sm font-bold"
                style={{ background: 'rgba(232,119,34,0.15)', color: '#E87722', border: '1px solid rgba(232,119,34,0.3)' }}>
                BASKER Energy
              </span>
            </div>
            <div className="text-center text-sm font-medium" style={{ color: '#64748B' }}>Legacy SCADA</div>
            <div className="text-center text-sm font-medium" style={{ color: '#64748B' }}>Generic SaaS</div>
          </div>
          {comparisons.map((c, i) => (
            <motion.div key={c.feature}
              initial={{ opacity: 0, x: -20 }} animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ delay: 0.3 + i * 0.05 }}
              className="grid grid-cols-4 px-5 py-4 border-b transition-all"
              style={{ borderColor: 'rgba(255,255,255,0.04)' }}
              onMouseEnter={e => (e.currentTarget.style.background = 'rgba(232,119,34,0.04)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
              <div className="text-sm font-medium" style={{ color: '#CBD5E1' }}>{c.feature}</div>
              <div className="flex justify-center">
                {c.basker ? <CheckCircle size={20} style={{ color: '#00E5A0' }} /> : <XCircle size={20} style={{ color: '#EF4444' }} />}
              </div>
              <div className="flex justify-center">
                {c.legacy ? <CheckCircle size={20} style={{ color: '#00E5A0' }} /> : <XCircle size={20} style={{ color: '#EF4444' }} />}
              </div>
              <div className="flex justify-center">
                {c.saas ? <CheckCircle size={20} style={{ color: '#00E5A0' }} /> : <XCircle size={20} style={{ color: '#EF4444' }} />}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
