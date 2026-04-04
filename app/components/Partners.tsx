'use client';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { Handshake } from 'lucide-react';

const partners = [
  { name: 'Adani Energy', role: 'Strategic Energy Partner', color: '#E87722' },
  { name: 'BCG Smart Energy', role: 'Technology Consultant', color: '#00C4FF' },
  { name: 'NTPC Renewable', role: 'Utility Partner', color: '#00E5A0' },
  { name: 'ReNew Power', role: 'IPP Partner', color: '#A78BFA' },
  { name: 'Tata Power Solar', role: 'EPC Partner', color: '#F59E0B' },
  { name: 'IMD Weather', role: 'Meteorological Data', color: '#6EE7B7' },
  { name: 'IEX', role: 'Energy Exchange Partner', color: '#FB923C' },
  { name: 'CERC', role: 'Regulatory Alignment', color: '#60A5FA' },
  { name: 'Green Infra Wind', role: 'Wind Energy Partner', color: '#34D399' },
  { name: 'Waaree Energies', role: 'Module Partner', color: '#F472B6' },
];

const certs = [
  'MNRE Empanelled Platform',
  'ISO 27001 Security Aligned',
  'CERC Schedule IV Compliant',
  'BEE Star Rating Methodology',
  'IEC 61724-1 Performance Standard',
  'BCG Smart Energy Certified',
];

export default function Partners() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });

  return (
    <section id="partners" ref={ref} className="relative py-28 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><Handshake size={14} /> Partners & Certifications</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            Trusted by <span className="gradient-text">India's Best</span>
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            Our network of strategic partners and regulatory certifications ensures the highest standards across every deployment.
          </p>
        </motion.div>

        {/* Partner logos grid */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-16">
          {partners.map((p, i) => (
            <motion.div key={p.name} initial={{ opacity: 0, y: 20 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: i * 0.05 }}
              className="glass-card-hover p-5 text-center">
              <div className="w-10 h-10 rounded-xl mx-auto mb-3 flex items-center justify-center"
                style={{ background: `${p.color}18`, border: `1px solid ${p.color}30` }}>
                <span className="text-sm font-black" style={{ color: p.color }}>{p.name.charAt(0)}</span>
              </div>
              <p className="text-sm font-bold mb-1" style={{ color: '#E2E8F0' }}>{p.name}</p>
              <p className="text-xs" style={{ color: '#64748B' }}>{p.role}</p>
            </motion.div>
          ))}
        </div>

        {/* Certifications */}
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.5 }}
          className="glass-card p-8">
          <h3 className="text-xl font-bold text-center mb-8" style={{ fontFamily: 'Space Grotesk' }}>
            Certifications & Compliance
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {certs.map((c, i) => (
              <div key={c} className="flex items-center gap-3 p-4 rounded-xl"
                style={{ background: 'rgba(232,119,34,0.06)', border: '1px solid rgba(232,119,34,0.15)' }}>
                <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: '#E87722' }} />
                <span className="text-sm font-medium" style={{ color: '#CBD5E1' }}>{c}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
