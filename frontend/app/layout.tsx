import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Variant Risk Explainer",
  description: "AI-powered genomic sequence analysis for variant risk interpretation"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
