import type { ReactNode } from "react";
import { AppHeader } from "@/components/AppHeader";
import { AuthGate } from "@/components/AuthProvider";

// Signed-in pages with the app header. The lobby and meeting room render full-screen without it.
export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <AuthGate>
      <div className="app">
        <AppHeader />
        {children}
      </div>
    </AuthGate>
  );
}
