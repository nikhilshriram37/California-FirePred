import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "California Wildfire Risk — 5-Day Forecast",
  description:
    "5-day forward-looking wildfire ignition-risk forecast for California, scored daily by ML and overlaid with live satellite fire detections.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
