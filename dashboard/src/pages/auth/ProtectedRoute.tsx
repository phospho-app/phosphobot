import { useAuth } from "@/context/AuthContext";
import { ReactNode } from "react";
import { Navigate } from "react-router-dom";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { session } = useAuth();

  if (!session) {
    return <Navigate to="/sign-up" />;
  }

  return <>{children}</>;
}
