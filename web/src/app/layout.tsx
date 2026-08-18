import type { Metadata, Viewport } from "next";
import { Archivo } from "next/font/google";

import "@/styles/modernist.css";
import "@/styles/app.css";

/* Modernist sets everything in Archivo. Loading it through next/font
 * self-hosts the file and removes the render-blocking round trip to Google. */
const archivo = Archivo({
  subsets: ["latin"],
  weight: ["400", "600", "800"],
  variable: "--font-archivo",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Pearls AQI Predictor",
  description:
    "A three-day air quality forecast for Islamabad, produced by a serverless machine learning pipeline and published with the reasoning behind every number.",
  applicationName: "Pearls AQI Predictor",
  authors: [{ name: "Muhammad Aitazaz Ahsan" }],
  openGraph: {
    title: "Pearls AQI Predictor",
    description: "Three-day air quality forecasts for Islamabad, with the model reasoning shown.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#f3f2f2",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={archivo.variable}>
      <body>{children}</body>
    </html>
  );
}
