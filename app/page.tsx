import Navbar from './components/Navbar';
import Hero from './components/Hero';
import About from './components/About';
import Solutions from './components/Solutions';
import WhyBasker from './components/WhyBasker';
import Industries from './components/Industries';
import Technology from './components/Technology';
import Impact from './components/Impact';
import CaseStudies from './components/CaseStudies';
import Partners from './components/Partners';
import Careers from './components/Careers';
import Insights from './components/Insights';
import Contact from './components/Contact';
import Footer from './components/Footer';

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <div className="divider-orange" />
        <About />
        <div className="divider-orange" />
        <Solutions />
        <div className="divider-orange" />
        <WhyBasker />
        <div className="divider-orange" />
        <Industries />
        <div className="divider-orange" />
        <Technology />
        <div className="divider-orange" />
        <Impact />
        <div className="divider-orange" />
        <CaseStudies />
        <div className="divider-orange" />
        <Partners />
        <div className="divider-orange" />
        <Careers />
        <div className="divider-orange" />
        <Insights />
        <div className="divider-orange" />
        <Contact />
      </main>
      <Footer />
    </>
  );
}
