import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Variant Risk Explainer",
  description: "Research-only GRCh38 variant risk explanation demo"
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
