'use client';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { Zap, Factory, Home, Building2, Truck, TreePine } from 'lucide-react';

const industries = [
  {
    icon: Zap,
    title: 'Utility Scale',
    sub: '>1 MW Solar & Wind',
    desc: 'Full-stack monitoring for IPP and PSU utilities. Grid compliance, deviation settlement, and revenue optimization at scale.',
    tier: 'T1 — ₹1.5–5L/MW/yr',
    color: '#E87722',
    clients: ['Adani Energy', 'ReNew Power', 'NTPC Renewable', 'Green Infra Wind'],
    features: ['Fleet Portfolio Dashboard', 'DISCOM Interface', 'Regulatory Reporting', 'PPA Optimization'],
  },
  {
    icon: Factory,
    title: 'MSMEs & Industry',
    sub: '100kW – 1MW',
    desc: 'Factories and industrial parks reduce energy costs by ₹54L/year with smart solar + storage management and demand-side optimization.',
    tier: 'T2 — ₹15K–50K/site/mo',
    color: '#00C4FF',
    clients: ['Tata Steel Plants', 'Mahindra Auto', 'JSW Group Sites', 'MIDC Industrial Parks'],
    features: ['Demand Charge Reduction', 'Export Revenue Tracking', 'Equipment Health Monitoring', 'Carbon Footprint Reports'],
  },
  {
    icon: Home,
    title: 'Residential Premium',
    sub: '3kW – 10kW Rooftop',
    desc: 'Smart home energy dashboards with AI-powered consumption insights, net metering optimization, and battery storage management.',
    tier: 'T3 — ₹2K–10K/home/mo',
    color: '#00E5A0',
    clients: ['Premium Housing Societies', 'Villa Communities', 'Smart City Projects'],
    features: ['Mobile App Dashboard', 'Net Meter Tracking', 'Bill Prediction', 'Neighbor Benchmarking'],
  },
  {
    icon: Building2,
    title: 'Commercial Buildings',
    sub: '50kW – 500kW',
    desc: 'Office parks, malls, and commercial complexes optimize rooftop solar yield and integrate with building management systems.',
    tier: 'T2 Plus — Custom',
    color: '#A78BFA',
    clients: ['Embassy Office Parks', 'DLF Commercial', 'Brigade Group'],
    features: ['BMS Integration', 'Tenant Energy Billing', 'Green Building Certification', 'Load Forecasting'],
  },
  {
    icon: Truck,
    title: 'EV Charging Hubs',
    sub: 'Solar + BESS + EV',
    desc: 'Solar-powered EV charging stations with BESS dispatch intelligence — maximize green charging revenue while minimizing grid import costs.',
    tier: 'T2 Plus — ₹25K/site/mo',
    color: '#F59E0B',
    clients: ['ChargeZone', 'Ather Energy', 'CESL Highway Hubs'],
    features: ['Charging Load Prediction', 'Solar+BESS Coordination', 'V2G Readiness', 'Revenue Analytics'],
  },
  {
    icon: TreePine,
    title: 'Carbon Markets',
    sub: 'ESG & REC Trading',
    desc: 'End-to-end carbon credit origination, REC tracking, and ESG reporting for organizations monetizing their renewable energy investments.',
    tier: 'Custom — % of credits',
    color: '#6EE7B7',
    clients: ['Climate Neutral IPPs', 'Corporate RPO Buyers', 'ESG Funds'],
    features: ['REC Issuance Tracking', 'Carbon Credit Registry', 'Scope 2 Reporting', 'Investor ESG Dashboards'],
  },
];

export default function Industries() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section id="industries" ref={ref} className="relative py-28 overflow-hidden">
      <div className="glow-orb-orange" style={{ width: '600px', height: '600px', top: '30%', right: '-200px', opacity: 0.2 }} />
      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><Factory size={14} /> Industries Served</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            Every Scale, <span className="gradient-text">Every Sector</span>
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            From India's largest utility IPPs to premium residential solar — BASKER scales across every renewable energy segment.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {industries.map((ind, i) => (
            <motion.div key={ind.title} initial={{ opacity: 0, y: 40 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: i * 0.08 }}
              className="glass-card-hover card-shine p-7 flex flex-col"
              style={{ borderTop: `3px solid ${ind.color}` }}>
              <div className="flex items-start justify-between mb-5">
                <div className="w-12 h-12 rounded-xl flex items-center justify-center"
                  style={{ background: `${ind.color}18`, border: `1px solid ${ind.color}30` }}>
                  <ind.icon size={22} style={{ color: ind.color }} />
                </div>
                <span className="text-xs font-semibold px-3 py-1 rounded-full"
                  style={{ background: `${ind.color}12`, color: ind.color, border: `1px solid ${ind.color}30` }}>
                  {ind.tier}
                </span>
              </div>
              <h3 className="text-xl font-bold mb-1" style={{ fontFamily: 'Space Grotesk' }}>{ind.title}</h3>
              <p className="text-xs font-medium mb-3" style={{ color: '#64748B' }}>{ind.sub}</p>
              <p className="text-sm leading-relaxed mb-5 flex-grow" style={{ color: '#94A3B8' }}>{ind.desc}</p>
              <div className="flex flex-wrap gap-1.5 mb-5">
                {ind.features.map(f => (
                  <span key={f} className="text-xs px-2.5 py-1 rounded-lg"
                    style={{ background: 'rgba(255,255,255,0.04)', color: '#94A3B8', border: '1px solid rgba(255,255,255,0.06)' }}>
                    {f}
                  </span>
                ))}
              </div>
              <p className="text-xs" style={{ color: '#4A5568' }}>
                Clients: {ind.clients.slice(0, 2).join(', ')}{ind.clients.length > 2 ? ` +${ind.clients.length - 2} more` : ''}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
