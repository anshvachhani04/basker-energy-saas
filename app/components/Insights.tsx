'use client';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { BookOpen, Calendar, Clock, ArrowRight, Tag } from 'lucide-react';

const posts = [
  {
    tag: 'Technical Deep Dive',
    tagColor: '#E87722',
    title: 'How BASKER\'s Physics-AI Hybrid Engine Achieves R²=0.9998 Solar Forecasting',
    excerpt: 'We break down the Spencer solar position model, Bird clear-sky irradiance correction, and NOCT thermal adjustment that underpins our world-class prediction accuracy.',
    date: 'March 2026',
    readTime: '8 min read',
    author: 'BASKER AI Team',
  },
  {
    tag: 'Case Study',
    tagColor: '#00C4FF',
    title: 'Bhadla Solar Park: From 67% to 84% Performance Ratio in 90 Days',
    excerpt: 'A detailed breakdown of how BASKER\'s ML-powered fault detection and soiling optimization transformed India\'s largest solar park\'s O&M operations.',
    date: 'February 2026',
    readTime: '6 min read',
    author: 'BASKER Analytics',
  },
  {
    tag: 'Industry Insight',
    tagColor: '#00E5A0',
    title: 'India\'s BESS Revolution: Why RL-Powered Dispatch Changes Everything',
    excerpt: 'Reinforcement learning is no longer just for games. Here\'s how BASKER\'s PPO-based BESS dispatch agent is generating ₹4.2L/MW/year for Indian IPPs.',
    date: 'January 2026',
    readTime: '5 min read',
    author: 'BASKER Research',
  },
  {
    tag: 'Regulation',
    tagColor: '#A78BFA',
    title: 'CERC\'s New Grid Compliance Rules: What Every Solar IPP Must Know in 2026',
    excerpt: 'Navigating Schedule IV deviations, RPO compliance, and the new CERC deviation settlement mechanism — with practical guidance for O&M teams.',
    date: 'December 2025',
    readTime: '7 min read',
    author: 'BASKER Policy Team',
  },
  {
    tag: 'ESG',
    tagColor: '#F59E0B',
    title: 'How to Monetize Your Solar REC Portfolio in India\'s Voluntary Carbon Market',
    excerpt: 'Step-by-step guide to REC issuance, carbon credit origination, and finding voluntary market buyers — with BASKER\'s automated ESG intelligence tools.',
    date: 'November 2025',
    readTime: '6 min read',
    author: 'BASKER ESG Team',
  },
  {
    tag: 'Product Update',
    tagColor: '#6EE7B7',
    title: 'BASKER v2.0: 10 Modules, Real-time Fleet Intelligence, and New ESG Dashboard',
    excerpt: 'Everything that\'s new in BASKER Energy SaaS v2.0 — enhanced ML models, multi-plant fleet analytics, and a redesigned investor-facing ESG reporting module.',
    date: 'October 2025',
    readTime: '4 min read',
    author: 'BASKER Product',
  },
];

export default function Insights() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section id="insights" ref={ref} className="relative py-28 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }} className="text-center mb-16">
          <div className="section-tag"><BookOpen size={14} /> Insights & Blog</div>
          <h2 className="text-4xl md:text-6xl font-bold mb-5" style={{ fontFamily: 'Space Grotesk' }}>
            Energy Intelligence, <span className="gradient-text">Decoded</span>
          </h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: '#94A3B8' }}>
            Technical deep dives, case studies, and policy updates from India's leading renewable energy AI platform.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {posts.map((p, i) => (
            <motion.article key={p.title} initial={{ opacity: 0, y: 40 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: i * 0.08 }}
              className="glass-card-hover p-6 flex flex-col cursor-pointer group">
              <div className="flex items-center gap-2 mb-4">
                <Tag size={12} style={{ color: p.tagColor }} />
                <span className="text-xs font-semibold" style={{ color: p.tagColor }}>{p.tag}</span>
              </div>
              <h3 className="text-base font-bold mb-3 leading-snug group-hover:text-orange-400 transition-colors"
                style={{ fontFamily: 'Space Grotesk', color: '#E2E8F0' }}>
                {p.title}
              </h3>
              <p className="text-sm leading-relaxed flex-grow mb-5" style={{ color: '#64748B' }}>{p.excerpt}</p>
              <div className="flex items-center justify-between pt-4 border-t"
                style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
                <div className="flex items-center gap-3 text-xs" style={{ color: '#4A5568' }}>
                  <span className="flex items-center gap-1"><Calendar size={11} /> {p.date}</span>
                  <span className="flex items-center gap-1"><Clock size={11} /> {p.readTime}</span>
                </div>
                <ArrowRight size={14} className="transition-transform group-hover:translate-x-1"
                  style={{ color: p.tagColor }} />
              </div>
            </motion.article>
          ))}
        </div>

        <motion.div initial={{ opacity: 0 }} animate={inView ? { opacity: 1 } : {}} transition={{ delay: 0.6 }}
          className="text-center mt-12">
          <a href="#" className="btn-secondary">View All Articles <ArrowRight size={16} /></a>
        </motion.div>
      </div>
    </section>
  );
}
