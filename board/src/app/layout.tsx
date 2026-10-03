import type { Metadata } from "next";
import type { ReactNode } from "react";
import { identity } from "@moe/contracts";
import "@livekit/components-styles";
import "./globals.css";

export const metadata: Metadata = { title: identity.product_name };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
