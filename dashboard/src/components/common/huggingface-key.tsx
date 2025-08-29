import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetchWithBaseUrl } from "@/lib/utils";
import { CheckCircle2, HelpCircle, LoaderCircle, Save } from "lucide-react";
import type React from "react";
import { useState } from "react";
import { toast } from "sonner";

export function HuggingFaceKeyInput() {
  const [token, setToken] = useState("");
  const [isError, setIsError] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Handle Hugging Face form submission
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    setIsError(false);
    setIsSuccess(false);

    if (token != "" && !token.startsWith("hf_")) {
      toast.error("Token should start with 'hf_'");
      setIsError(true);
      return;
    }

    setIsLoading(true);
    fetchWithBaseUrl("/admin/huggingface", "POST", {
      token,
    }).then((response) => {
      setIsLoading(false);
      if (response.status == "ok") {
        toast.success("Hugging Face token saved successfully");
        setIsSuccess(true);
        // auto hide success message after 5 seconds
        setTimeout(() => {
          setIsSuccess(false);
        }, 5000);
      } else {
        toast.error(response.message || "Failed to save WandB token");
        setIsError(true);
      }
    });
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
                isError ? "border-red-500 focus-visible:ring-red-500" : ""
              }
              disabled={isLoading}
              autoComplete="off"
            />
            <Button type="submit" disabled={isLoading}>
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

      {isSuccess && (
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
