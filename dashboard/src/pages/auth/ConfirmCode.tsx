"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/context/AuthContext";
import { Loader2, Mail } from "lucide-react";
import type React from "react";
import { type FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

export function ConfirmCode() {
  const [code, setCode] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const { verifyEmailCode } = useAuth();
  const navigate = useNavigate();

  // get the email from the URL query parameters
  const searchParams = new URLSearchParams(window.location.search);
  const email = searchParams.get("email");

  // If email is not provided, redirect to sign-in
  if (!email) {
    toast.error("Email is required to verify your account.");
    //navigate("/sign-up", { replace: true });
    //return;
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    // Trim the code
    const trimmedCode = code.trim();

    if (!trimmedCode || trimmedCode.length !== 6) {
      toast.error("Please enter a valid 6-digit verification code.");
      setIsLoading(false);
      return;
    }

    if (!email) {
      toast.error("Email is required to verify your account.");
      setIsLoading(false);
      return;
    }

    try {
      await verifyEmailCode(email, trimmedCode);
      toast.success("Email verified successfully!");
      navigate("/", { replace: true });
    } catch (error) {
      console.error(error);
    }

    setIsLoading(false);
  };

  const handleCodeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.replace(/\D/g, "").slice(0, 6);
    setCode(value);
  };

  return (
    <div className="flex items-center justify-center bg-muted">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
            <Mail className="h-6 w-6 text-primary" />
          </div>
          <CardTitle className="text-2xl font-semibold text-center">
            Verify Your Email
          </CardTitle>
          <p className="text-sm text-muted-foreground mt-2">
            We've sent a 6-digit verification code to confirm your email
            address. Please enter it below to confirm your account.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col space-y-2">
              <label htmlFor="code" className="text-sm font-medium">
                Verification Code
              </label>
              <Input
                id="code"
                type="text"
                value={code}
                onChange={handleCodeChange}
                placeholder="Enter 6-digit code"
                className="text-center text-lg tracking-widest font-mono"
                maxLength={6}
                required
                disabled={isLoading}
              />
            </div>

            <Button
              type="submit"
              className="w-full"
              disabled={isLoading || code.length !== 6}
            >
              {isLoading ? (
                <>
                  <Loader2 className="size-4 mr-2 animate-spin" />
                  Verifying...
                </>
              ) : (
                "Verify Email"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
