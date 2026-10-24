import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// Inter 4 (OFL-1.1, see public/fonts/OFL.txt): the variable Latin subset Google Fonts serves, taken from
// @fontsource-variable/inter 5.3.0. A committed file, so building never fetches Google Fonts (blocked in some regions).
const inter = localFont({
  src: "../../public/fonts/InterVariable-latin.woff2",
  weight: "100 900",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Prism - Paper to Manga",
  description: "Transform academic papers into adorable manga with AI",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  );
}
