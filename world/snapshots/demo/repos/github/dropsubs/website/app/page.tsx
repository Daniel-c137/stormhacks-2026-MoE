import Image from "next/image";
import Link from "next/link";
import { SUCCESS_FEE_PERCENT } from "@/lib/pricing";
import { WaitlistForm } from "./waitlist/form";

const FAQ = [
  {
    q: "Which emails do you read?",
    a: "Only receipts, renewal notices and cancellation confirmations. Nothing else is stored.",
  },
  {
    q: "What does it cost?",
    a: `${SUCCESS_FEE_PERCENT}% of what we save you in the first year. If we save you nothing, you pay nothing.`,
  },
  {
    q: "Can it cancel without asking me?",
    a: "No. Every cancellation waits for you to tap Approve.",
  },
];

export default function Home() {
  return (
    <main>
      <section className="hero">
        <h1>Find every subscription. Cancel the ones you forgot.</h1>
        <p>DropSubs reads your receipts, finds what you pay for, and cancels it when you say so.</p>
        <WaitlistForm cta="Join the waitlist" />
        <Image src="/screens/iphone-list.png" alt="Your subscriptions on iPhone" width={320} height={640} />
      </section>
      <section id="pricing">
        <h2>Pricing</h2>
        <p>
          We charge {SUCCESS_FEE_PERCENT}% of what we save you in the first year. Nothing if we save
          you nothing.
        </p>
      </section>
      <section id="faq">
        {FAQ.map(({ q, a }) => (
          <details key={q}>
            <summary>{q}</summary>
            <p>{a}</p>
          </details>
        ))}
      </section>
      <footer>
        <Link href="/privacy">Privacy</Link>
      </footer>
    </main>
  );
}
