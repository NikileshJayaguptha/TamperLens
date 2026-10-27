import type { Metadata } from "next";
import Link from "next/link";
import ApiStatus from "@/components/ApiStatus";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Edit Detector",
  description: "Find AI-edited regions in photos",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" />
      </head>
      <body>
        <header className="topbar">
          <Link href="/" className="brand">AI Edit Detector</Link>
          <ApiStatus />
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
