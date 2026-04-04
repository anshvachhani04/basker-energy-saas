'use client';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { Sun, Wind, Battery, BarChart3, AlertTriangle, Droplets, TrendingUp, Globe, ArrowRight } from 'lucide-react';

const solutions = [
  {
    id: 'solar',
    icon: Sun,
    color: '#E87722',
    title: 'Solar Intelligence Suite',
    tag: 'M001–M003',
    headline: 'Physics-AI Hybrid Forecasting',
    desc: 'Combines Spencer solar position model, Bird clear-sky irradiance, and NOCT thermal correction with XGBoost (R²=0.9998) for minute-level AC power prediction.',
    features: ['24–168h Generation Forecasts', 'Performance Ratio Tracking', 'Irradiance Decomposition', 'String-Level Monitoring', 'Shadow Analysis'],
    kpi: { label: 'Forecast Accuracy', value: '99.9% R²' },
  },
  {
    id: 'fault',
    icon: AlertTriangle,
    color: '#FF6B6B',
    title: 'Predictive Fault Detection',
    tag: 'M004–M005',
    headline: 'Zero Unplanned Downtime',
    desc: 'XGBoost Classifier (AUC=0.87) + Isolation Forest anomaly detector identifies inverter trips, string faults, tracker failures, and degradation patterns 48h before failure.',
    features: ['Inverter Health Scoring', 'String Fault Isolation', 'Degradation Trend Analysis', 'Maintenance Work Orders', 'RCA Auto-Reporting'],
    kpi: { label: 'Fault Detection AUC', value: '0.87' },
  },
  {
    id: 'cleaning',
    icon: Droplets,
    color: '#00C4FF',
    title: 'Soiling & Cleaning AI',
    tag: 'M006',
    headline: 'Optimize O&M Spend',
    desc: 'XGBoost Regressor (MAE=0.18%) estimates soiling loss in real-time using weather, dust, humidity models. Triggers ROI-based cleaning alerts to maximize cleaning cycles.',
    features: ['Soiling Loss Estimation', 'Cleaning ROI Calculator', 'Regional Dust Forecasts', 'Washing Schedule Optimizer', 'Water Usage Minimization'],
    kpi: { label: 'Soiling MAE', value: '0.18%' },
  },
  {
    id: 'bess',
    icon: Battery,
    color: '#00E5A0',
    title: 'BESS Dispatch & Trading',
    tag: 'M007 (RL-PPO)',
    headline: 'Maximize Every Stored kWh',
    desc: 'Reinforcement Learning (PPO agent) optimizes BESS charge/discharge cycles against real-time grid prices, demand forecasts, and CERC guidelines — delivering +₹4.2L/MW/year.',
    features: ['Real-time Price Signals', 'Charge/Discharge Scheduling', 'IEX Market Integration', 'DISCOM Tariff Optimization', 'Revenue Forecasting'],
    kpi: { label: 'Revenue Uplift', value: '₹4.2L/MW/yr' },
  },
  {
    id: 'esg',
    icon: Globe,
    color: '#A78BFA',
    title: 'ESG & Carbon Intelligence',
    tag: 'CERC-Aligned',
    headline: 'India\'s Leading ESG Engine',
    desc: 'CERC-aligned CO₂ avoidance calculations, REC generation tracking, carbon credit monetization, and automated ESG reports for regulatory filing and investor reporting.',
    features: ['CO₂ Avoidance Certificates', 'REC Generation Tracking', 'Carbon Credit Monetization', 'ESG Score Dashboard', 'Investor Report Generation'],
    kpi: { label: 'CO₂ Tracked', value: '1.2M+ tCO₂/yr' },
  },
  {
    id: 'trading',
    icon: TrendingUp,
    color: '#F59E0B',
    title: 'Energy Trading Signals',
    tag: 'Real-time',
    headline: 'Trade Smarter, Earn More',
    desc: 'AI-driven buy/sell/hold signals integrated with IEX spot markets and PPA management. Automated day-ahead scheduling and deviation settlement minimization.',
    features: ['IEX Spot Market Signals', 'PPA Management', 'Deviation Settlement', 'Day-ahead Scheduling', 'Revenue Analytics'],
    kpi: { label: 'Trading Signal Accuracy', value: '91.3%' },
  },
];

export default function Solutions() {
  const [active, setActive] = useState(solutions[0].id);
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });
  const sol = solutions.find(s => s.id === active)!;

  return (
    <section id="solutions" ref={ref} className="relative py-28 overflow-hidden">
      <div className="glow-orb-blue" style={{ width: '500px', height: '500px', bottom: '0', left: '-150px', opacity: 0.3 }} />
      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-14">
          <div className="section-tag"><BarChart3 size={14} /> Our Solutions</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            AI-Powered <span className="gradient-text">Energy Suite</span>
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            Seven specialized AI modules covering every dimension of renewable energy management.
          </p>
        </motion.div>

        {/* Tab buttons */}
        <div className="flex flex-wrap gap-3 justify-center mb-12">
          {solutions.map((s, i) => (
            <motion.button key={s.id} initial={{ opacity: 0, y: 20 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: i * 0.07 }}
              onClick={() => setActive(s.id)}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-semibold transition-all"
              style={{
                background: active === s.id ? `${s.color}20` : 'rgba(255,255,255,0.04)',
                border: `1px solid ${active === s.id ? s.color : 'rgba(255,255,255,0.1)'}`,
                color: active === s.id ? s.color : '#94A3B8',
              }}>
              <s.icon size={15} />
              {s.title.split(' ')[0]} {s.title.split(' ')[1]}
            </motion.button>
          ))}
        </div>

        {/* Detail panel */}
        <AnimatePresence mode="wait">
          <motion.div key={active} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
            className="glass-card p-8 lg:p-12 grid lg:grid-cols-2 gap-10"
            style={{ border: `1px solid ${sol.color}25` }}>
            <div>
              <div className="flex items-center gap-4 mb-6">
                <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
                  style={{ background: `${sol.color}20`, border: `1px solid ${sol.color}40` }}>
                  <sol.icon size={26} style={{ color: sol.color }} />
                </div>
                <div>
                  <span className="tech-badge" style={{ background: `${sol.color}15`, borderColor: `${sol.color}40`, color: sol.color }}>{sol.tag}</span>
                  <h3 className="text-2xl font-bold mt-1" style={{ fontFamily: 'Space Grotesk' }}>{sol.title}</h3>
                </div>
              </div>
              <h4 className="text-xl font-semibold mb-4" style={{ color: sol.color }}>{sol.headline}</h4>
              <p className="text-base leading-relaxed mb-8" style={{ color: '#94A3B8' }}>{sol.desc}</p>
              <a href="#contact" className="btn-primary inline-flex" style={{ background: `linear-gradient(135deg, ${sol.color}, ${sol.color}99)`, padding: '13px 28px' }}>
                Explore Solution <ArrowRight size={16} />
              </a>
            </div>
            <div>
              <div className="glass-card p-6 mb-6" style={{ background: `${sol.color}08`, border: `1px solid ${sol.color}20` }}>
                <p className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: '#64748B' }}>Key Performance Indicator</p>
                <p className="text-3xl font-bold gradient-text" style={{ fontFamily: 'Space Grotesk', backgroundImage: `linear-gradient(135deg, ${sol.color}, ${sol.color}99)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  {sol.kpi.value}
                </p>
                <p className="text-sm mt-1" style={{ color: '#94A3B8' }}>{sol.kpi.label}</p>
              </div>
              <ul className="space-y-3">
                {sol.features.map(f => (
                  <li key={f} className="flex items-center gap-3">
                    <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: sol.color }} />
                    <span className="text-sm" style={{ color: '#CBD5E1' }}>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}
