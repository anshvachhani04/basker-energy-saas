import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BASKER Energy — AI-Powered Renewable Energy Intelligence",
  description: "World-class SaaS platform for solar, wind, and BESS asset management. Real-time AI analytics, predictive maintenance, ESG compliance, and energy trading intelligence for India's renewable energy sector.",
  keywords: "renewable energy, solar analytics, wind energy, BESS dispatch, AI energy, ESG, India solar, performance ratio, SCADA analytics",
  authors: [{ name: "BASKER Energy", url: "https://baskerenergy.ai" }],
  creator: "BCG Smart Energy Solutions",
  openGraph: {
    title: "BASKER Energy — AI-Powered Renewable Energy Intelligence",
    description: "World-class AI platform for India's renewable energy sector. Monitor 4.7 GW, prevent faults, optimize BESS, and maximize ESG value.",
    type: "website",
    locale: "en_IN",
  },
  twitter: {
    card: "summary_large_image",
    title: "BASKER Energy — AI-Powered Renewable Energy Intelligence",
    description: "World-class AI platform for India's renewable energy sector.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body style={{ background: '#020408', color: '#ffffff', overflowX: 'hidden' }}>
        {children}
      </body>
    </html>
  );
}
