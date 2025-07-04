import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fetchWithBaseUrl } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertCircle,
  CheckCircle2,
  HelpCircle,
  LoaderCircle,
  Save,
} from "lucide-react";
import type React from "react";
import { useState } from "react";

export function WandBKeyInput() {
  const [token, setToken] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Form submission
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    // Reset states
    setError(null);
    setSuccess(false);

    // Validate token
    if (!token.trim()) {
      setError("WandB Token is required");
      return;
    }

    // No specific patern for WandB token
    // So no pattern check here

    setIsLoading(true);

    try {
      const result = await fetchWithBaseUrl("/admin/wandb", "POST", { token });

      if (result && result.status == "success") {
        setSuccess(true);
      } else {
        setError(result?.message || "Failed to save token");
      }
    } catch (error) {
      console.error("Error saving token:", error);
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setIsLoading(false);

      // Auto-hide success message after 5 seconds
      if (success) {
        setTimeout(() => {
          setSuccess(false);
        }, 5000);
      }
    }
  };

  return (
    <div className="space-y-2">
      <form onSubmit={handleSubmit} className="space-y-2">
        <div className="space-y-2">
          <Label htmlFor="token">
            Weights and Biases token{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Your token is securely stored. It will be used to report
                    training metrics to Weights and Biases.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </Label>
          <div className="text-sm text-muted-foreground">
            <p>
              Go to{" "}
              <a
                href="https://wandb.ai/authorize"
                target="_blank"
                rel="noopener noreferrer"
                className="underline text-primary hover:text-primary/80"
              >
                WandB authorization page
              </a>{" "}
              and get your token.
            </p>
          </div>
          <div className="flex gap-x-2">
            <Input
              id="token"
              type="password"
              placeholder="•••••••••••••••••••••••••••••••••"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              className={
                error ? "border-red-500 focus-visible:ring-red-500" : ""
              }
              disabled={isLoading}
              autoComplete="off"
            />
            <Button
              type="submit"
              disabled={isLoading}
              className="cursor-pointer"
            >
              {isLoading ? (
                <span className="flex items-center">
                  <LoaderCircle className="animate-spin size-5" />
                  Saving...
                </span>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save token
                </>
              )}
            </Button>
          </div>
        </div>
      </form>

      {error && (
        <Alert variant="destructive" className="mt-2">
          <AlertCircle className="h-4 w-4" />
          <div className="ml-2">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </div>
        </Alert>
      )}

      {success && (
        <Alert className="mt-2 bg-green-50 text-green-800 border-green-200">
          <CheckCircle2 className="h-4 w-4 text-green-600" />
          <div className="ml-2">
            <AlertTitle>Success</AlertTitle>
            <AlertDescription>
              WandB token has been saved successfully.
            </AlertDescription>
          </div>
        </Alert>
      )}
    </div>
  );
}
