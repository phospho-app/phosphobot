"use client";

import { useAuth } from "@phosphobot/shared-auth";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

export function LogoutButton() {
  const router = useRouter();
  const { logout } = useAuth();

  const handleLogout = async () => {
    await logout();
    router.push("/auth/login");
  };

  return <Button onClick={handleLogout}>Logout</Button>;
}
