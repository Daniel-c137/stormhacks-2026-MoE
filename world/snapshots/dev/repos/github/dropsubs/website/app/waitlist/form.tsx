"use client";

import { useState } from "react";

/** Email and first name; the API sends the confirmation email and returns the new count. */
export function WaitlistForm({ cta }: { cta: string }) {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [count, setCount] = useState<number | null>(null);

  const join = async () => {
    const response = await fetch("/api/waitlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, name }),
    });
    if (response.ok) setCount((await response.json()).count);
  };

  return (
    <form onSubmit={(e) => (e.preventDefault(), void join())}>
      <input value={name} onChange={(e) => setName(e.target.value)} placeholder="First name" />
      <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" required />
      <button type="submit">{cta}</button>
      {count !== null && <p>You're on the list, with {count.toLocaleString()} others.</p>}
    </form>
  );
}
