import type { Metadata } from "next";
import { Geist, Geist_Mono, Exo_2 } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const exo = Exo_2({
  subsets: ["latin"],
  variable: "--font-exo",
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Picks",
  description:
    "Discover your next favorite movie or TV show with personalized recommendations based on your tastes and preferences.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${exo.className} antialiased`}>
        {children}
        <Toaster />
      </body>
    </html>
  );
}
