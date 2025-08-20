import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/context/AuthContext";
import { Loader2 } from "lucide-react";
import { type FormEvent, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { toast } from "sonner";

export function AuthForm() {
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const { login, signup } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || "/";

  // Determine the title based on the current path
  const getTitle = () => {
    if (location.pathname.includes("sign-in")) {
      return "Sign in to your phosphobot account";
    }
    if (location.pathname.includes("sign-up")) {
      return "Create a new phosphobot account";
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    if (!email || !password) {
      toast.error("Email and password are required.");
      setIsLoading(false);
      return;
    }

    if (location.pathname.includes("sign-up")) {
      try {
        await signup(email, password);
        toast.success(
          "Account created! Please check your email for email validation code.",
        );
        navigate(`/sign-up/confirm?email=${encodeURIComponent(email)}`, {
          replace: true,
        });
      } catch (signupErr) {
        console.error(signupErr);
      }
      setIsLoading(false);
      return;
    }

    if (location.pathname.includes("sign-in")) {
      try {
        await login(email, password);
        toast.success("Logged in successfully!");
        navigate(from, { replace: true });
      } catch (loginErr) {
        console.error(loginErr);
      }
      setIsLoading(false);
      return;
    }
  };

  return (
    <div className="flex items-center justify-center bg-muted">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold text-center">
            {getTitle()}
          </CardTitle>
          <p className="text-sm text-muted-foreground text-center">
            {location.pathname.includes("sign-up") &&
              "Create an account to enable the AI features in phosphobot."}
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              key="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email"
              className="border p-2 rounded"
              required
              disabled={isLoading}
            />
            <Input
              key="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              className="border p-2 rounded"
              required
              disabled={isLoading}
            />
            <Button
              type="submit"
              variant="outline"
              className="w-full"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="size-5 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                "Continue"
              )}
            </Button>
          </form>
          <p className="text-sm text-muted-foreground text-center">
            <a
              href="/auth/forgot-password"
              className="underline cursor-pointer"
            >
              Forgot password?
            </a>
          </p>
          <p className="text-sm text-muted-foreground text-center">
            {location.pathname.includes("sign-in")
              ? "Don't have an account?"
              : "Already have an account?"}{" "}
            <a
              href={
                location.pathname.includes("sign-in") ? "/sign-up" : "/sign-in"
              }
              className="underline cursor-pointer"
            >
              {location.pathname.includes("sign-in") ? "Sign Up" : "Sign In"}
            </a>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
