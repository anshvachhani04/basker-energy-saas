'use client';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { Briefcase, MapPin, Clock, ArrowRight, Users } from 'lucide-react';

const roles = [
  { title: 'Senior ML Engineer', dept: 'AI/ML', loc: 'Bengaluru / Remote', type: 'Full-time', level: 'Senior', color: '#E87722', hot: true },
  { title: 'Full Stack Developer', dept: 'Engineering', loc: 'Mumbai / Hybrid', type: 'Full-time', level: 'Mid-Senior', color: '#00C4FF', hot: true },
  { title: 'Energy Data Scientist', dept: 'Analytics', loc: 'Hyderabad / Remote', type: 'Full-time', level: 'Mid', color: '#00E5A0', hot: false },
  { title: 'SCADA Integration Engineer', dept: 'Platform', loc: 'Pune / On-site', type: 'Full-time', level: 'Senior', color: '#A78BFA', hot: false },
  { title: 'Business Development Manager', dept: 'Sales', loc: 'Delhi / Mumbai', type: 'Full-time', level: 'Senior', color: '#F59E0B', hot: true },
  { title: 'Product Manager — Energy SaaS', dept: 'Product', loc: 'Bengaluru', type: 'Full-time', level: 'Senior', color: '#6EE7B7', hot: false },
];

const perks = [
  { label: 'Work on India\'s Most Impactful Clean Energy Platform' },
  { label: 'Competitive Compensation + ESOP for Senior Hires' },
  { label: 'Remote-First Culture with Quarterly Off-sites' },
  { label: 'Learning Budget ₹1L/year + Conference Sponsorship' },
  { label: 'Access to BCG Smart Energy Advisory Network' },
  { label: 'Direct Impact on India\'s Net-Zero 2070 Mission' },
];

export default function Careers() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });

  return (
    <section id="careers" ref={ref} className="relative py-28 overflow-hidden">
      <div className="glow-orb-orange" style={{ width: '400px', height: '400px', top: '10%', right: '-100px', opacity: 0.2 }} />
      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><Users size={14} /> Careers</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            Build the Future of <span className="gradient-text">India's Energy Grid</span>
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            Join a team of 60+ engineers, data scientists, and energy experts redefining how India manages its renewable future.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-5 mb-14">
          {roles.map((r, i) => (
            <motion.div key={r.title} initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: i * 0.08 }}
              className="glass-card-hover p-6 group cursor-pointer">
              <div className="flex items-start justify-between mb-4">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: `${r.color}18`, border: `1px solid ${r.color}30` }}>
                  <Briefcase size={18} style={{ color: r.color }} />
                </div>
                <div className="flex gap-2">
                  {r.hot && (
                    <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                      style={{ background: 'rgba(232,119,34,0.15)', color: '#E87722', border: '1px solid rgba(232,119,34,0.3)' }}>
                      Hot
                    </span>
                  )}
                  <span className="text-xs px-2 py-0.5 rounded-full"
                    style={{ background: 'rgba(255,255,255,0.04)', color: '#64748B', border: '1px solid rgba(255,255,255,0.08)' }}>
                    {r.level}
                  </span>
                </div>
              </div>
              <h3 className="text-base font-bold mb-1" style={{ fontFamily: 'Space Grotesk' }}>{r.title}</h3>
              <p className="text-xs font-medium mb-3" style={{ color: r.color }}>{r.dept}</p>
              <div className="flex items-center gap-4 text-xs" style={{ color: '#64748B' }}>
                <span className="flex items-center gap-1"><MapPin size={11} /> {r.loc}</span>
                <span className="flex items-center gap-1"><Clock size={11} /> {r.type}</span>
              </div>
              <div className="mt-4 flex items-center gap-1 text-xs font-semibold opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ color: r.color }}>
                Apply Now <ArrowRight size={12} />
              </div>
            </motion.div>
          ))}
        </div>

        {/* Perks */}
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.5 }}
          className="glass-card p-8">
          <h3 className="text-2xl font-bold text-center mb-8" style={{ fontFamily: 'Space Grotesk' }}>Why Work at BASKER Energy?</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {perks.map((p, i) => (
              <div key={i} className="flex items-center gap-3 p-4 rounded-xl"
                style={{ background: 'rgba(0,229,160,0.05)', border: '1px solid rgba(0,229,160,0.1)' }}>
                <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{ background: 'rgba(0,229,160,0.15)' }}>
                  <div className="w-2 h-2 rounded-full" style={{ background: '#00E5A0' }} />
                </div>
                <span className="text-sm" style={{ color: '#CBD5E1' }}>{p.label}</span>
              </div>
            ))}
          </div>
          <div className="text-center mt-8">
            <a href="mailto:careers@baskerenergy.ai" className="btn-primary">
              View All Openings <ArrowRight size={16} />
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
