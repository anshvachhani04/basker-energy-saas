'use client';
import { motion } from 'framer-motion';
import { Zap, ExternalLink, Share2, Code2, Play, Mail, ArrowRight } from 'lucide-react';

const footerLinks = {
  'Platform': ['Solar Intelligence', 'Wind Analytics', 'BESS Dispatch', 'Fault Detection', 'ESG Intelligence', 'Energy Trading'],
  'Company': ['About Us', 'Careers', 'Press & Media', 'Partners', 'Blog & Insights', 'Contact'],
  'Resources': ['API Documentation', 'System Status', 'Security', 'Privacy Policy', 'Terms of Service', 'CERC Compliance'],
  'Industries': ['Utility Scale', 'MSMEs & Factories', 'Commercial Buildings', 'Residential Premium', 'EV Charging Hubs', 'Carbon Markets'],
};

const socials = [
  { icon: ExternalLink, href: '#', label: 'LinkedIn' },
  { icon: Share2, href: '#', label: 'Twitter' },
  { icon: Code2, href: 'https://github.com/anshvachhani04', label: 'GitHub' },
  { icon: Play, href: '#', label: 'YouTube' },
];

export default function Footer() {
  return (
    <footer className="relative pt-20 pb-8 overflow-hidden" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
      <div className="max-w-7xl mx-auto px-6">
        {/* CTA banner */}
        <div className="rounded-3xl p-10 mb-16 text-center relative overflow-hidden"
          style={{ background: 'linear-gradient(135deg, rgba(232,119,34,0.15), rgba(0,196,255,0.08))', border: '1px solid rgba(232,119,34,0.25)' }}>
          <div className="glow-orb-orange" style={{ width: '400px', height: '300px', top: '-50%', left: '50%', transform: 'translateX(-50%)', opacity: 0.4 }} />
          <h2 className="text-3xl md:text-4xl font-bold mb-4 relative z-10" style={{ fontFamily: 'Space Grotesk' }}>
            Ready to Transform Your <span className="gradient-text">Energy Assets?</span>
          </h2>
          <p className="text-base mb-8 relative z-10 max-w-xl mx-auto" style={{ color: '#94A3B8' }}>
            Join 50+ utilities and IPPs already maximizing returns with BASKER Energy's AI platform.
          </p>
          <div className="flex gap-4 justify-center relative z-10">
            <a href="#contact" className="btn-primary">Get Free Demo <ArrowRight size={16} /></a>
            <a href="mailto:hello@baskerenergy.ai" className="btn-secondary flex items-center gap-2">
              <Mail size={15} /> Email Us
            </a>
          </div>
        </div>

        {/* Links grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-10 mb-12">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg, #E87722, #C45F0A)' }}>
                <Zap size={18} color="white" fill="white" />
              </div>
              <span className="text-xl font-bold" style={{ fontFamily: 'Space Grotesk' }}>
                <span className="gradient-text">BASKER</span> Energy
              </span>
            </div>
            <p className="text-sm leading-relaxed mb-5" style={{ color: '#64748B' }}>
              AI-Powered Renewable Energy Intelligence for India's sustainable future.
            </p>
            <div className="flex gap-3">
              {socials.map(s => (
                <a key={s.label} href={s.href} target="_blank" rel="noopener noreferrer"
                  className="w-9 h-9 rounded-lg flex items-center justify-center transition-all"
                  style={{ background: 'rgba(255,255,255,0.05)', color: '#64748B', border: '1px solid rgba(255,255,255,0.08)' }}
                  onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = '#E87722'; (e.currentTarget as HTMLElement).style.borderColor = 'rgba(232,119,34,0.4)'; }}
                  onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = '#64748B'; (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.08)'; }}>
                  <s.icon size={15} />
                </a>
              ))}
            </div>
          </div>

          {Object.entries(footerLinks).map(([section, links]) => (
            <div key={section}>
              <h4 className="text-sm font-bold mb-5" style={{ color: '#E2E8F0' }}>{section}</h4>
              <ul className="space-y-2.5">
                {links.map(link => (
                  <li key={link}>
                    <a href="#" className="text-sm transition-colors"
                      style={{ color: '#64748B' }}
                      onMouseEnter={e => ((e.target as HTMLElement).style.color = '#E87722')}
                      onMouseLeave={e => ((e.target as HTMLElement).style.color = '#64748B')}>
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col md:flex-row items-center justify-between pt-6 gap-4"
          style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <p className="text-xs" style={{ color: '#4A5568' }}>
            © 2026 BASKER Energy. Built by BCG Smart Energy Solutions for Basker Energy / Adani Energy. All rights reserved.
          </p>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: '#00E5A0' }} />
            <span className="text-xs" style={{ color: '#4A5568' }}>All systems operational</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
