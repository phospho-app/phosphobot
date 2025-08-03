import { useAuth } from "@/context/AuthContext";
import { ReactNode } from "react";
import { Navigate } from "react-router-dom";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { session, isLoading } = useAuth();

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (!session) {
    return <Navigate to="/auth" />;
  }

  return <>{children}</>;
}
