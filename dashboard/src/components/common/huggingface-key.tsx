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

export function HuggingFaceKeyInput() {
  const [token, setToken] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Handle Hugging Face form submission
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    // Reset states
    setError(null);
    setSuccess(false);

    // Validate token
    if (!token.trim()) {
      setError("Hugging Face Token is required");
      return;
    }

    if (!token.startsWith("hf_")) {
      setError("Token should start with 'hf_'");
      return;
    }

    setIsLoading(true);

    try {
      const result = await fetchWithBaseUrl("/admin/huggingface", "POST", { token });

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
            Hugging Face token{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Your token is securely stored. It will be used to sync
                    datasets and models to the Hugging Face hub.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </Label>
          <div className="text-sm text-muted-foreground">
            <p>
              Go to{" "}
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noopener noreferrer"
                className="underline text-primary hover:text-primary/80"
              >
                Hugging Face settings page
              </a>{" "}
              and create a token with{" "}
              <span className="font-semibold">
                Write access to content/settings
              </span>{" "}
              for syncing datasets and models.
            </p>
          </div>
          <div className="flex gap-x-2">
            <Input
              id="token"
              type="password"
              placeholder="hf_••••••••••••••••••••••••••••••"
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
              Hugging Face token has been saved successfully.
            </AlertDescription>
          </div>
        </Alert>
      )}
    </div>
  );
}
