import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/context/AuthContext";
import { fetchWithBaseUrl } from "@/lib/utils";
import { Session } from "@/types";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

export function ConfirmEmail() {
  const { login } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const confirmEmail = async () => {
      const hash = window.location.hash.substring(1);
      const params = new URLSearchParams(hash);
      const accessToken = params.get("access_token");
      const refreshToken = params.get("refresh_token");

      if (!accessToken || !refreshToken) {
        toast.error("Missing access_token or refresh_token in URL");
        navigate("/");
        return;
      }

      const data: { message: string; session: Session } =
        await fetchWithBaseUrl("/auth/confirm", "POST", {
          access_token: accessToken,
          refresh_token: refreshToken,
        });

      if (data) {
        localStorage.setItem("session", JSON.stringify(data.session));
        login(data.session.user_email, "", data.session);
        toast.success("Email confirmed successfully!");
        navigate("/");
      }
    };

    confirmEmail();
  }, [login, navigate]);

  return (
    <div className="flex items-center justify-center bg-muted">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold text-center">
            Email Confirmation
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <p className="text-center text-muted-foreground">
            Confirming your email...
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
