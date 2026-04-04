'use client';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { MapPin, TrendingUp, Zap, ArrowRight, ChevronLeft, ChevronRight } from 'lucide-react';

const cases = [
  {
    id: 1,
    plant: 'Bhadla Solar Park',
    location: 'Rajasthan',
    capacity: '10 MW',
    client: 'Adani Energy',
    challenge: 'Performance Ratio declining to 67% due to undetected soiling and inverter faults across 40,000 panels in extreme desert conditions.',
    solution: 'BASKER deployed M001 (RF Power Model) + M006 (Soiling AI) + M004 (Fault Detection). Integrated with SCADA for 5-min data ingestion.',
    results: [
      { label: 'Performance Ratio', before: '67%', after: '84%', improvement: '+25%' },
      { label: 'Annual Revenue', before: '₹1.8Cr', after: '₹2.4Cr', improvement: '+₹60L' },
      { label: 'MTTR (hours)', before: '48h', after: '6h', improvement: '-87%' },
      { label: 'Cleaning Cost', before: '₹12L/yr', after: '₹7L/yr', improvement: '-42%' },
    ],
    color: '#E87722',
    tag: 'Solar + Fault Detection',
  },
  {
    id: 2,
    plant: 'Pavagada Solar Complex',
    location: 'Karnataka',
    capacity: '8.5 MW',
    client: 'ReNew Power',
    challenge: 'BESS underutilized — only 40% of storage capacity used effectively. Missing peak price windows for grid export revenue.',
    solution: 'BASKER M007 (RL-PPO Dispatch Agent) trained on 18 months of IEX price data, DISCOM tariff schedules, and weather forecasts.',
    results: [
      { label: 'BESS Utilization', before: '40%', after: '89%', improvement: '+122%' },
      { label: 'Trading Revenue', before: '₹45L/yr', after: '₹87L/yr', improvement: '+₹42L' },
      { label: 'Grid Deviation Penalty', before: '₹8L/yr', after: '₹1.2L/yr', improvement: '-85%' },
      { label: 'Peak Export Capture', before: '55%', after: '91%', improvement: '+65%' },
    ],
    color: '#00C4FF',
    tag: 'BESS + Trading AI',
  },
  {
    id: 3,
    plant: 'Rewa Ultra Mega Solar',
    location: 'Madhya Pradesh',
    capacity: '7.5 MW',
    client: 'NTPC Renewable',
    challenge: 'ESG reporting done manually — 200+ hours annually, errors in REC generation claims, and no carbon credit monetization.',
    solution: 'BASKER ESG Intelligence module with automated CERC-aligned CO₂ calculations, REC tracker, and investor-ready report generation.',
    results: [
      { label: 'Reporting Time', before: '200h/yr', after: '2h/yr', improvement: '-99%' },
      { label: 'REC Revenue', before: '₹18L/yr', after: '₹26L/yr', improvement: '+₹8L' },
      { label: 'ESG Score', before: '61/100', after: '89/100', improvement: '+46%' },
      { label: 'Carbon Credits', before: '₹0', after: '₹12L/yr', improvement: 'New Revenue' },
    ],
    color: '#00E5A0',
    tag: 'ESG + Carbon Intelligence',
  },
];

export default function CaseStudies() {
  const [active, setActive] = useState(0);
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });
  const c = cases[active];

  return (
    <section id="case-studies" ref={ref} className="relative py-28 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-14">
          <div className="section-tag"><TrendingUp size={14} /> Case Studies</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            Proven Results <span className="gradient-text">Across India</span>
          </h2>
        </motion.div>

        {/* Case selector */}
        <div className="flex gap-3 mb-8 justify-center flex-wrap">
          {cases.map((c2, i) => (
            <button key={c2.id} onClick={() => setActive(i)}
              className="px-5 py-2.5 rounded-full text-sm font-semibold transition-all"
              style={{
                background: active === i ? `${c2.color}20` : 'rgba(255,255,255,0.04)',
                border: `1px solid ${active === i ? c2.color : 'rgba(255,255,255,0.1)'}`,
                color: active === i ? c2.color : '#94A3B8',
              }}>
              {c2.plant.split(' ')[0]} {c2.location && `· ${c2.location}`}
            </button>
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.div key={active} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
            className="glass-card p-8 lg:p-12"
            style={{ border: `1px solid ${c.color}25` }}>
            <div className="grid lg:grid-cols-2 gap-10">
              <div>
                <div className="flex items-start gap-4 mb-6">
                  <div>
                    <span className="text-xs font-semibold px-3 py-1 rounded-full mb-2 inline-block"
                      style={{ background: `${c.color}15`, color: c.color, border: `1px solid ${c.color}30` }}>
                      {c.tag}
                    </span>
                    <h3 className="text-2xl font-bold" style={{ fontFamily: 'Space Grotesk' }}>{c.plant}</h3>
                    <div className="flex items-center gap-4 mt-1">
                      <span className="flex items-center gap-1 text-sm" style={{ color: '#64748B' }}>
                        <MapPin size={13} /> {c.location}
                      </span>
                      <span className="flex items-center gap-1 text-sm" style={{ color: '#64748B' }}>
                        <Zap size={13} /> {c.capacity}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="mb-5">
                  <h4 className="text-sm font-semibold uppercase tracking-wider mb-2" style={{ color: '#64748B' }}>The Challenge</h4>
                  <p className="text-sm leading-relaxed" style={{ color: '#94A3B8' }}>{c.challenge}</p>
                </div>
                <div>
                  <h4 className="text-sm font-semibold uppercase tracking-wider mb-2" style={{ color: '#64748B' }}>Our Solution</h4>
                  <p className="text-sm leading-relaxed" style={{ color: '#94A3B8' }}>{c.solution}</p>
                </div>
              </div>
              <div>
                <h4 className="text-sm font-semibold uppercase tracking-wider mb-4" style={{ color: '#64748B' }}>Results Achieved</h4>
                <div className="space-y-4">
                  {c.results.map(r => (
                    <div key={r.label} className="p-4 rounded-xl"
                      style={{ background: `${c.color}08`, border: `1px solid ${c.color}18` }}>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium" style={{ color: '#CBD5E1' }}>{r.label}</span>
                        <span className="text-sm font-bold px-2 py-0.5 rounded-lg"
                          style={{ background: `${c.color}20`, color: c.color }}>
                          {r.improvement}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs line-through" style={{ color: '#4A5568' }}>{r.before}</span>
                        <ArrowRight size={12} style={{ color: c.color }} />
                        <span className="text-sm font-bold" style={{ color: c.color }}>{r.after}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        <div className="flex justify-center gap-4 mt-6">
          <button onClick={() => setActive(Math.max(0, active - 1))}
            className="w-10 h-10 rounded-full flex items-center justify-center border transition-all"
            style={{ borderColor: active > 0 ? '#E87722' : 'rgba(255,255,255,0.1)', color: active > 0 ? '#E87722' : '#4A5568' }}>
            <ChevronLeft size={18} />
          </button>
          {cases.map((_, i) => (
            <button key={i} onClick={() => setActive(i)}
              className="w-2.5 h-2.5 rounded-full transition-all"
              style={{ background: active === i ? '#E87722' : 'rgba(255,255,255,0.2)' }} />
          ))}
          <button onClick={() => setActive(Math.min(cases.length - 1, active + 1))}
            className="w-10 h-10 rounded-full flex items-center justify-center border transition-all"
            style={{ borderColor: active < cases.length - 1 ? '#E87722' : 'rgba(255,255,255,0.1)', color: active < cases.length - 1 ? '#E87722' : '#4A5568' }}>
            <ChevronRight size={18} />
          </button>
        </div>
      </div>
    </section>
  );
}
