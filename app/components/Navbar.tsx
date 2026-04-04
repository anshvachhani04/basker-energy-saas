'use client';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Menu, X, ChevronDown } from 'lucide-react';

const navLinks = [
  { label: 'Solutions', href: '#solutions', sub: ['Solar Intelligence', 'Wind Analytics', 'BESS Dispatch', 'Grid Management'] },
  { label: 'Industries', href: '#industries' },
  { label: 'Technology', href: '#technology' },
  { label: 'Impact', href: '#impact' },
  { label: 'Case Studies', href: '#case-studies' },
  { label: 'About', href: '#about' },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 30);
    window.addEventListener('scroll', handler);
    return () => window.removeEventListener('scroll', handler);
  }, []);

  return (
    <>
      <motion.nav
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'nav-blur' : 'bg-transparent'}`}
      >
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between h-18" style={{ height: '72px' }}>
          {/* Logo */}
          <a href="#" className="flex items-center gap-3 group">
            <div className="relative w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, #E87722, #C45F0A)', boxShadow: '0 0 20px rgba(232,119,34,0.4)' }}>
              <Zap size={20} color="white" fill="white" />
            </div>
            <div>
              <span className="text-xl font-bold" style={{ fontFamily: 'Space Grotesk' }}>
                <span className="gradient-text">BASKER</span>
                <span className="text-white"> Energy</span>
              </span>
            </div>
          </a>

          {/* Desktop Nav */}
          <div className="hidden lg:flex items-center gap-1">
            {navLinks.map(link => (
              <div key={link.label} className="relative"
                onMouseEnter={() => link.sub && setActiveDropdown(link.label)}
                onMouseLeave={() => setActiveDropdown(null)}>
                <a href={link.href}
                  className="flex items-center gap-1 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                  style={{ color: '#94A3B8' }}
                  onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                  onMouseLeave={e => (e.currentTarget.style.color = '#94A3B8')}>
                  {link.label}
                  {link.sub && <ChevronDown size={14} />}
                </a>
                {link.sub && activeDropdown === link.label && (
                  <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                    className="absolute top-full left-0 mt-2 w-52 glass-card p-2 rounded-xl"
                    style={{ background: 'rgba(6,11,20,0.95)', border: '1px solid rgba(232,119,34,0.2)' }}>
                    {link.sub.map(s => (
                      <a key={s} href="#solutions"
                        className="block px-4 py-2.5 rounded-lg text-sm transition-all"
                        style={{ color: '#94A3B8' }}
                        onMouseEnter={e => { e.currentTarget.style.color = '#E87722'; e.currentTarget.style.background = 'rgba(232,119,34,0.08)'; }}
                        onMouseLeave={e => { e.currentTarget.style.color = '#94A3B8'; e.currentTarget.style.background = 'transparent'; }}>
                        {s}
                      </a>
                    ))}
                  </motion.div>
                )}
              </div>
            ))}
          </div>

          {/* CTA */}
          <div className="hidden lg:flex items-center gap-3">
            <a href="#contact" className="btn-secondary" style={{ padding: '10px 22px', fontSize: '14px' }}>Contact</a>
            <a href="#contact" className="btn-primary" style={{ padding: '10px 22px', fontSize: '14px' }}>
              Get Demo <Zap size={14} />
            </a>
          </div>

          {/* Mobile toggle */}
          <button onClick={() => setMobileOpen(!mobileOpen)}
            className="lg:hidden p-2 rounded-lg" style={{ color: 'white', background: 'rgba(255,255,255,0.05)' }}>
            {mobileOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </motion.nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div initial={{ opacity: 0, x: '100%' }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: '100%' }}
            transition={{ type: 'tween', duration: 0.3 }}
            className="fixed inset-0 z-40 lg:hidden"
            style={{ background: 'rgba(2,4,8,0.97)', backdropFilter: 'blur(20px)', paddingTop: '80px' }}>
            <div className="px-6 py-4 flex flex-col gap-2">
              {navLinks.map(link => (
                <a key={link.label} href={link.href}
                  onClick={() => setMobileOpen(false)}
                  className="py-4 text-lg font-medium border-b"
                  style={{ color: '#E2E8F0', borderColor: 'rgba(255,255,255,0.06)' }}>
                  {link.label}
                </a>
              ))}
              <div className="flex flex-col gap-3 mt-6">
                <a href="#contact" onClick={() => setMobileOpen(false)} className="btn-primary text-center justify-center">Get Demo <Zap size={16} /></a>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
