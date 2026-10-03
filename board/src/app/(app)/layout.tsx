import type { ReactNode } from "react";
import { AppHeader } from "@/components/AppHeader";

// Pages with the app header. The meeting room renders full-screen without it.
export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <AppHeader />
      <main>{children}</main>
    </>
  );
}
