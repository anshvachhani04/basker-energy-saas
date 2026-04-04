'use client';
import { useState } from 'react';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { Mail, Phone, MapPin, Send, Zap } from 'lucide-react';

const contactInfo = [
  { icon: Mail, label: 'Email', value: 'hello@baskerenergy.ai', href: 'mailto:hello@baskerenergy.ai', color: '#E87722' },
  { icon: Phone, label: 'Phone', value: '+91 98765 43210', href: 'tel:+919876543210', color: '#00C4FF' },
  { icon: MapPin, label: 'Head Office', value: 'WeWork Galaxy, Bengaluru 560001', href: '#', color: '#00E5A0' },
];

const tiers = [
  { label: 'Utility Scale (>1 MW)', value: 'utility' },
  { label: 'MSME / Factory (100kW–1MW)', value: 'msme' },
  { label: 'Commercial Building', value: 'commercial' },
  { label: 'Residential Premium', value: 'residential' },
  { label: 'ESG / Carbon Markets', value: 'esg' },
];

export default function Contact() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });
  const [form, setForm] = useState({ name: '', email: '', company: '', capacity: '', segment: '', message: '' });
  const [sent, setSent] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSent(true);
  };

  return (
    <section id="contact" ref={ref} className="relative py-28 overflow-hidden">
      <div className="glow-orb-orange" style={{ width: '500px', height: '500px', bottom: '-100px', left: '-100px', opacity: 0.25 }} />
      <div className="glow-orb-blue" style={{ width: '400px', height: '400px', top: '0', right: '-100px', opacity: 0.2 }} />

      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><Zap size={14} /> Get Started</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            Let's <span className="gradient-text">Power Your Portfolio</span>
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            Book a free 30-minute demo tailored to your assets. Our team will show you exactly how BASKER impacts your specific portfolio.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-5 gap-10">
          {/* Contact info */}
          <motion.div initial={{ opacity: 0, x: -30 }} animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: 0.2 }} className="lg:col-span-2 flex flex-col gap-6">
            {contactInfo.map(c => (
              <a key={c.label} href={c.href}
                className="glass-card-hover p-6 flex items-start gap-4">
                <div className="w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0"
                  style={{ background: `${c.color}18`, border: `1px solid ${c.color}30` }}>
                  <c.icon size={20} style={{ color: c.color }} />
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: '#64748B' }}>{c.label}</p>
                  <p className="text-sm font-medium" style={{ color: '#E2E8F0' }}>{c.value}</p>
                </div>
              </a>
            ))}

            <div className="glass-card p-6 mt-2"
              style={{ background: 'linear-gradient(135deg, rgba(232,119,34,0.1), rgba(0,196,255,0.05))' }}>
              <h4 className="font-bold mb-3" style={{ fontFamily: 'Space Grotesk' }}>Book a Demo</h4>
              <p className="text-sm mb-4" style={{ color: '#94A3B8' }}>
                Free 30-min personalized demo. No sales pitch — just results.
              </p>
              <ul className="space-y-2">
                {['Live platform walkthrough', 'Your asset data analysis', 'ROI projection report', 'Integration feasibility'].map(item => (
                  <li key={item} className="flex items-center gap-2 text-sm" style={{ color: '#CBD5E1' }}>
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#E87722' }} />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>

          {/* Form */}
          <motion.div initial={{ opacity: 0, x: 30 }} animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: 0.3 }} className="lg:col-span-3">
            {sent ? (
              <div className="glass-card p-12 text-center h-full flex flex-col items-center justify-center"
                style={{ border: '1px solid rgba(0,229,160,0.3)' }}>
                <div className="w-16 h-16 rounded-full flex items-center justify-center mb-6"
                  style={{ background: 'rgba(0,229,160,0.15)', border: '2px solid #00E5A0' }}>
                  <Send size={28} style={{ color: '#00E5A0' }} />
                </div>
                <h3 className="text-2xl font-bold mb-3 gradient-text-green" style={{ fontFamily: 'Space Grotesk' }}>
                  Message Received!
                </h3>
                <p style={{ color: '#94A3B8' }}>Our team will reach out within 24 hours to schedule your personalized demo.</p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="glass-card p-8 space-y-5">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold mb-1.5" style={{ color: '#94A3B8' }}>Full Name *</label>
                    <input className="input-field" placeholder="Arjun Sharma" required
                      value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold mb-1.5" style={{ color: '#94A3B8' }}>Work Email *</label>
                    <input type="email" className="input-field" placeholder="arjun@adani.com" required
                      value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold mb-1.5" style={{ color: '#94A3B8' }}>Company</label>
                    <input className="input-field" placeholder="Adani Energy" 
                      value={form.company} onChange={e => setForm({ ...form, company: e.target.value })} />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold mb-1.5" style={{ color: '#94A3B8' }}>Plant Capacity (MW)</label>
                    <input className="input-field" placeholder="e.g. 50 MW"
                      value={form.capacity} onChange={e => setForm({ ...form, capacity: e.target.value })} />
                  </div>
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1.5" style={{ color: '#94A3B8' }}>Segment</label>
                  <select className="input-field" value={form.segment} onChange={e => setForm({ ...form, segment: e.target.value })}
                    style={{ appearance: 'none', background: 'rgba(10,22,40,0.8)' }}>
                    <option value="">Select your segment...</option>
                    {tiers.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1.5" style={{ color: '#94A3B8' }}>Message</label>
                  <textarea className="input-field" rows={4} placeholder="Tell us about your assets and challenges..."
                    value={form.message} onChange={e => setForm({ ...form, message: e.target.value })}
                    style={{ resize: 'none' }} />
                </div>
                <button type="submit" className="btn-primary w-full justify-center text-base" style={{ padding: '16px' }}>
                  Request Free Demo <Send size={16} />
                </button>
                <p className="text-xs text-center" style={{ color: '#4A5568' }}>
                  No spam. Demo within 24h. Cancel anytime.
                </p>
              </form>
            )}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
