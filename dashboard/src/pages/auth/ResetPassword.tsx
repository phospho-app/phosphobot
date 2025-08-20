import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { fetchWithBaseUrl } from "@/lib/utils";
import { Loader2 } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

export function ResetPassword() {
  const [newPassword, setNewPassword] = useState<string>("");
  const [confirmPassword, setConfirmPassword] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [tokens, setTokens] = useState<{
    accessToken: string | null;
    refreshToken: string | null;
  }>({
    accessToken: null,
    refreshToken: null,
  });
  const navigate = useNavigate();

  useEffect(() => {
    const hash = window.location.hash.substring(1);
    const params = new URLSearchParams(hash);
    const accessToken = params.get("access_token");
    const refreshToken = params.get("refresh_token");
    const type = params.get("type");

    if (type !== "recovery" || !accessToken || !refreshToken) {
      toast.error(
        "Invalid or missing reset link. Please request a new password reset.",
      );
      return;
    }

    setTokens({ accessToken, refreshToken });
  }, []);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!tokens.accessToken || !tokens.refreshToken) {
      toast.error("No valid reset tokens available.");
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error("Passwords do not match.");
      return;
    }

    setIsLoading(true);

    const response = await fetchWithBaseUrl("/auth/reset-password", "POST", {
      access_token: tokens.accessToken,
      refresh_token: tokens.refreshToken,
      new_password: newPassword,
    });

    if (response) {
      toast.success("Password reset successfully! Redirecting to login...");
      setTimeout(() => navigate("/sign-in"), 2000);
    }

    setIsLoading(false);
  };

  return (
    <div className="flex items-center justify-center bg-muted">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold text-center">
            Reset Your Password
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <p className="text-muted-foreground text-center">
            Enter a new password below to reset your account.
          </p>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="New Password"
              className="border p-2 rounded"
              required
              disabled={isLoading || !tokens.accessToken}
            />
            <Input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Confirm New Password"
              className="border p-2 rounded"
              required
              disabled={isLoading || !tokens.accessToken}
            />
            <Button
              type="submit"
              variant="outline"
              className="w-full"
              disabled={isLoading || !tokens.accessToken}
            >
              {isLoading ? (
                <>
                  <Loader2 className="size-5 mr-2 animate-spin" />
                  Resetting...
                </>
              ) : (
                "Reset Password"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
