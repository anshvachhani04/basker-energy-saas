'use client';
import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import CountUp from 'react-countup';
import { useInView as useIOInView } from 'react-intersection-observer';
import { Leaf, Zap, TrendingUp, Users, Globe, Award } from 'lucide-react';

const metrics = [
  { value: 4700, suffix: ' MW', label: 'Renewable Capacity Monitored', icon: Zap, color: '#E87722', decimals: 0 },
  { value: 1.2, suffix: 'M tCO₂', label: 'CO₂ Avoided per Year', icon: Leaf, color: '#00E5A0', decimals: 1 },
  { value: 340, suffix: ' Cr+', label: 'Client Revenue Uplift (₹)', icon: TrendingUp, color: '#00C4FF', decimals: 0 },
  { value: 50, suffix: '+', label: 'Utility Clients Across India', icon: Users, color: '#A78BFA', decimals: 0 },
  { value: 99.8, suffix: '%', label: 'Platform Uptime SLA', icon: Award, color: '#F59E0B', decimals: 1 },
  { value: 12, suffix: ' States', label: 'Pan-India Coverage', icon: Globe, color: '#6EE7B7', decimals: 0 },
];

const sdgs = [
  { num: 7, title: 'Affordable & Clean Energy', color: '#FCA326' },
  { num: 9, title: 'Industry Innovation', color: '#FF821E' },
  { num: 11, title: 'Sustainable Cities', color: '#FD9D24' },
  { num: 13, title: 'Climate Action', color: '#3F7E44' },
  { num: 17, title: 'Partnerships for Goals', color: '#19486A' },
];

export default function Impact() {
  const [ref, inView] = useIOInView({ triggerOnce: true, threshold: 0.1 });

  return (
    <section id="impact" ref={ref} className="relative py-28 overflow-hidden">
      <div className="glow-orb-orange" style={{ width: '400px', height: '400px', bottom: '0', left: '-100px', opacity: 0.25 }} />

      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><Leaf size={14} /> Our Impact</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            Numbers That <span className="gradient-text-green">Drive Change</span>
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            Powering India's energy transition — one megawatt, one tree, one rupee at a time.
          </p>
        </motion.div>

        {/* Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6 mb-16">
          {metrics.map((m, i) => (
            <motion.div key={m.label} initial={{ opacity: 0, scale: 0.9 }} animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ delay: i * 0.1 }}
              className="glass-card-hover p-7 text-center"
              style={{ border: `1px solid ${m.color}20` }}>
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4"
                style={{ background: `${m.color}15`, border: `1px solid ${m.color}30` }}>
                <m.icon size={22} style={{ color: m.color }} />
              </div>
              <div className="text-3xl md:text-4xl font-black mb-2" style={{ color: m.color, fontFamily: 'Space Grotesk' }}>
                {inView ? (
                  <CountUp start={0} end={m.value} duration={2.5} delay={i * 0.1}
                    decimals={m.decimals} suffix={m.suffix} />
                ) : `0${m.suffix}`}
              </div>
              <p className="text-sm font-medium" style={{ color: '#94A3B8' }}>{m.label}</p>
            </motion.div>
          ))}
        </div>

        {/* SDG alignment */}
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.5 }}
          className="glass-card p-8 text-center">
          <h3 className="text-2xl font-bold mb-4" style={{ fontFamily: 'Space Grotesk' }}>
            Aligned with UN Sustainable Development Goals
          </h3>
          <p className="text-sm mb-8" style={{ color: '#94A3B8' }}>
            BASKER Energy's operations directly contribute to 5 SDGs, supporting India's commitment to net-zero by 2070.
          </p>
          <div className="flex flex-wrap gap-4 justify-center">
            {sdgs.map(sdg => (
              <div key={sdg.num} className="flex items-center gap-3 px-5 py-3 rounded-xl"
                style={{ background: `${sdg.color}15`, border: `1px solid ${sdg.color}40` }}>
                <span className="text-2xl font-black" style={{ color: sdg.color, fontFamily: 'Space Grotesk' }}>
                  {sdg.num}
                </span>
                <span className="text-sm font-medium" style={{ color: '#CBD5E1' }}>{sdg.title}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
