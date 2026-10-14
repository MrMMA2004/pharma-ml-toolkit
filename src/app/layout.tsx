import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "next-themes";
import { Toaster } from "@/components/ui/sonner";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PharmaML Toolkit — ML · Pharma · Docking",
  description:
    "Typed, tested Python toolkit for QSAR machine learning, pharmaceutical PK/PD calculations, and molecular docking workflows — with an interactive code explorer, calculators playground, and download portal.",
  keywords: [
    "PharmaML",
    "QSAR",
    "machine learning",
    "pharmacokinetics",
    "molecular docking",
    "cheminformatics",
    "Python toolkit",
  ],
  authors: [{ name: "PharmaML Toolkit" }],
  icons: {
    icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
  },
  openGraph: {
    title: "PharmaML Toolkit",
    description: "One toolkit. Three scientific domains — ML, pharma PK/PD, molecular docking.",
    siteName: "PharmaML Toolkit",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <Toaster position="bottom-right" richColors closeButton />
        </ThemeProvider>
      </body>
    </html>
  );
}
