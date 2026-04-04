'use client';
import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Play, Zap, TrendingUp, Shield, Globe } from 'lucide-react';

const particles = Array.from({ length: 60 }, (_, i) => ({
  id: i,
  x: Math.random() * 100,
  y: Math.random() * 100,
  size: Math.random() * 2 + 0.5,
  duration: Math.random() * 8 + 4,
  delay: Math.random() * 5,
  opacity: Math.random() * 0.6 + 0.1,
}));

const stats = [
  { value: '4.7 GW', label: 'Capacity Monitored', icon: Zap, color: '#E87722' },
  { value: '₹340Cr+', label: 'Revenue Uplift', icon: TrendingUp, color: '#00C4FF' },
  { value: '99.8%', label: 'Platform Uptime', icon: Shield, color: '#00E5A0' },
  { value: '12 States', label: 'Pan-India Coverage', icon: Globe, color: '#A78BFA' },
];

const badges = [
  { label: 'AI-Powered Forecasting', color: '#E87722' },
  { label: 'Real-time SCADA Analytics', color: '#00C4FF' },
  { label: 'ESG Intelligence', color: '#00E5A0' },
  { label: 'CERC Grid Compliance', color: '#A78BFA' },
];

export default function Hero() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const pts = Array.from({ length: 80 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 1.5 + 0.5,
    }));

    let raf: number;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      pts.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(232,119,34,0.5)';
        ctx.fill();
      });
      pts.forEach((a, i) => {
        pts.slice(i + 1).forEach(b => {
          const d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < 120) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(232,119,34,${0.12 * (1 - d / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        });
      });
      raf = requestAnimationFrame(draw);
    };
    draw();
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    window.addEventListener('resize', resize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
  }, []);

  return (
    <section className="relative min-h-screen flex flex-col justify-center overflow-hidden grid-bg" id="hero">
      {/* Canvas particles */}
      <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" style={{ opacity: 0.7 }} />

      {/* Glow orbs */}
      <div className="glow-orb-orange" style={{ width: '700px', height: '700px', top: '-200px', left: '-200px', opacity: 0.6 }} />
      <div className="glow-orb-blue" style={{ width: '500px', height: '500px', bottom: '0', right: '-100px', opacity: 0.5 }} />

      {/* Hero gradient */}
      <div className="absolute inset-0 hero-gradient pointer-events-none" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-28 pb-16 w-full">
        {/* Badges */}
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
          className="flex flex-wrap gap-2 mb-8 justify-center">
          {badges.map((b, i) => (
            <motion.span key={b.label} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 * i }}
              className="px-4 py-1.5 rounded-full text-xs font-semibold border"
              style={{ background: `${b.color}15`, borderColor: `${b.color}40`, color: b.color }}>
              {b.label}
            </motion.span>
          ))}
        </motion.div>

        {/* Main heading */}
        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.2 }}
          className="text-center mb-8">
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold leading-tight mb-6" style={{ fontFamily: 'Space Grotesk', letterSpacing: '-0.02em' }}>
            Power India's
            <br />
            <span className="gradient-text">Renewable Future</span>
            <br />
            <span className="text-white">with AI Intelligence</span>
          </h1>
          <p className="text-lg md:text-xl max-w-3xl mx-auto" style={{ color: '#94A3B8', lineHeight: '1.8' }}>
            BASKER Energy's world-class SaaS platform delivers real-time AI analytics, predictive maintenance,
            and ESG compliance for utility-scale solar, wind, and BESS assets across India.
          </p>
        </motion.div>

        {/* CTA buttons */}
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.4 }}
          className="flex flex-wrap gap-4 justify-center mb-16">
          <a href="#contact" className="btn-primary text-base" style={{ padding: '16px 36px' }}>
            Request Demo <ArrowRight size={18} />
          </a>
          <a href="#solutions" className="btn-secondary text-base flex items-center gap-2" style={{ padding: '16px 36px' }}>
            <Play size={16} /> Watch Platform Tour
          </a>
        </motion.div>

        {/* Stats row */}
        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.6 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto">
          {stats.map((s, i) => (
            <motion.div key={s.label} whileHover={{ y: -4 }}
              className="stat-card text-center"
              style={{ border: `1px solid ${s.color}25` }}>
              <div className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-3"
                style={{ background: `${s.color}15` }}>
                <s.icon size={20} style={{ color: s.color }} />
              </div>
              <div className="text-2xl md:text-3xl font-bold mb-1" style={{ color: s.color, fontFamily: 'Space Grotesk' }}>
                {s.value}
              </div>
              <div className="text-xs font-medium" style={{ color: '#64748B' }}>{s.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Scroll indicator */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }}
          className="flex justify-center mt-12">
          <div className="flex flex-col items-center gap-2" style={{ color: '#4A5568' }}>
            <span className="text-xs font-medium tracking-widest uppercase">Scroll to explore</span>
            <div className="w-5 h-8 border-2 rounded-full flex justify-center pt-1.5"
              style={{ borderColor: 'rgba(255,255,255,0.15)' }}>
              <motion.div animate={{ y: [0, 10, 0] }} transition={{ duration: 1.5, repeat: Infinity }}
                className="w-1 h-2 rounded-full" style={{ background: '#E87722' }} />
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
