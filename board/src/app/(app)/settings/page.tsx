"use client";

import { useRouter } from "next/navigation";
import { TeamSettingsForm } from "@/components/settings/TeamSettingsForm";

export default function SettingsPage() {
  const router = useRouter();
  return <TeamSettingsForm onClose={() => router.push("/")} />;
}
