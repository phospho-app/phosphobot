"use client";

import Link from "next/link";
import { Button } from "./ui/button";
import { useAuth } from "@phosphobot/shared-auth";
import { LogoutButton } from "./logout-button";

export function AuthButton() {
  const { session } = useAuth();

  return session ? (
    <div className="flex items-center gap-4">
      <LogoutButton />
    </div>
  ) : (
    <div className="flex gap-2">
      <Button asChild size="sm" variant={"outline"}>
        <Link href="/auth/login">Sign in</Link>
      </Button>
      <Button asChild size="sm" variant={"default"}>
        <Link href="/auth/sign-up">Sign up</Link>
      </Button>
    </div>
  );
}
