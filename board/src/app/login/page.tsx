import { Manrope } from "next/font/google";
import { LoginForm } from "@/components/LoginForm";

const manrope = Manrope({ subsets: ["latin"], weight: ["400", "500", "600", "700", "800"], variable: "--font-manrope" });

export default function LoginPage() {
  return (
    <div className={manrope.variable}>
      <LoginForm />
    </div>
  );
}
