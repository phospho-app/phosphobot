import { CodeSnippet } from "@/components/common/code-snippet";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn, fetcher } from "@/lib/utils";
import type { ServerStatus } from "@/types";
import { Download, ExternalLink } from "lucide-react";
import { useMemo, useState } from "react";
import useSWR from "swr";

function DiscordIcon({ className }: { className?: string } = {}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 640 512"
      fill="currentColor"
      className={cn("size-4", className)}
    >
      <path d="M524.5 69.8a1.5 1.5 0 0 0 -.8-.7A485.1 485.1 0 0 0 404.1 32a1.8 1.8 0 0 0 -1.9 .9 337.5 337.5 0 0 0 -14.9 30.6 447.8 447.8 0 0 0 -134.4 0 309.5 309.5 0 0 0 -15.1-30.6 1.9 1.9 0 0 0 -1.9-.9A483.7 483.7 0 0 0 116.1 69.1a1.7 1.7 0 0 0 -.8 .7C39.1 183.7 18.2 294.7 28.4 404.4a2 2 0 0 0 .8 1.4A487.7 487.7 0 0 0 176 479.9a1.9 1.9 0 0 0 2.1-.7A348.2 348.2 0 0 0 208.1 430.4a1.9 1.9 0 0 0 -1-2.6 321.2 321.2 0 0 1 -45.9-21.9 1.9 1.9 0 0 1 -.2-3.1c3.1-2.3 6.2-4.7 9.1-7.1a1.8 1.8 0 0 1 1.9-.3c96.2 43.9 200.4 43.9 295.5 0a1.8 1.8 0 0 1 1.9 .2c2.9 2.4 6 4.9 9.1 7.2a1.9 1.9 0 0 1 -.2 3.1 301.4 301.4 0 0 1 -45.9 21.8 1.9 1.9 0 0 0 -1 2.6 391.1 391.1 0 0 0 30 48.8 1.9 1.9 0 0 0 2.1 .7A486 486 0 0 0 610.7 405.7a1.9 1.9 0 0 0 .8-1.4C623.7 277.6 590.9 167.5 524.5 69.8zM222.5 337.6c-29 0-52.8-26.6-52.8-59.2S193.1 219.1 222.5 219.1c29.7 0 53.3 26.8 52.8 59.2C275.3 311 251.9 337.6 222.5 337.6zm195.4 0c-29 0-52.8-26.6-52.8-59.2S388.4 219.1 417.9 219.1c29.7 0 53.3 26.8 52.8 59.2C470.7 311 447.5 337.6 417.9 337.6z" />
    </svg>
  );
}

function GithubIcon({ className }: { className?: string } = {}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 496 512"
      fill="currentColor"
      className={cn("size-4", className)}
    >
      <path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3 .3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5 .3-6.2 2.3zm44.2-1.7c-2.9 .7-4.9 2.6-4.6 4.9 .3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3 .7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3 .3 2.9 2.3 3.9 1.6 1 3.6 .7 4.3-.7 .7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3 .7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3 .7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z" />
    </svg>
  );
}

export function Footer() {
  const [isUpdating, setIsUpdating] = useState(false);

  const { data: serverStatus } = useSWR<ServerStatus>(
    ["/status"],
    ([url]) => fetcher(url),
    {
      refreshInterval: 5000,
    },
  );

  const { data: updateVersion } = useSWR<{ version?: string; error: string }>(
    ["/update/version"],
    ([url]) => fetcher(url, "POST"),
    {
      refreshInterval: 60000,
    },
  );

  // Compare the current version with the latest version
  // Version are string in the format "x.x.x"
  const isLatest = useMemo(() => {
    if (!serverStatus?.version_id || !updateVersion?.version) return true;

    const current = serverStatus.version_id.split(".").map(Number);
    const latest = updateVersion.version.split(".").map(Number);

    return current.every((v, i) => v >= (latest[i] || 0));
  }, [serverStatus?.version_id, updateVersion?.version]);

  // Return whether we're on linux, macos, or windows
  const operatingSystem = useMemo(() => {
    // don't use platform (deprecated)
    const platform = navigator.userAgent.toLowerCase();
    if (platform.includes("win")) return "windows";
    if (platform.includes("mac")) return "macos";
    return "linux";
  }, []);

  return (
    <footer className="fixed bottom-0 left-0 right-0 z-50 bg-background border-t p-4">
      <div className="flex justify-start items-center gap-x-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => {
            window.open("https://discord.gg/cbkggY6NSK", "_blank");
          }}
        >
          <DiscordIcon className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => {
            window.open("https://github.com/phospho-app/phosphobot", "_blank");
          }}
        >
          <GithubIcon className="size-4" />
        </Button>
        <Badge variant="outline" className="text-xs">
          {`${serverStatus?.version_id}`}{" "}
          {isLatest
            ? "(latest)"
            : `(update available: ${updateVersion?.version})`}
        </Badge>
        {!isLatest && operatingSystem == "linux" && (
          <Button
            onClick={async () => {
              // Sends a command to update the software. Disables the update button while updating.

              try {
                setIsUpdating(true);
                const res = await fetch("/update/upgrade-to-latest-version");
                if (!res.ok) throw new Error("Update failed");
                const data = await res.json();
                alert(data.status);
                setTimeout(() => {
                  window.location.reload();
                }, 2000);
              } catch (error) {
                console.error("Error updating software:", error);
                setTimeout(() => {
                  setIsUpdating(false);
                }, 3000);
              }
            }}
            size="sm"
            className="text-xs h-6 px-2 py-0 bg-green-500 hover:bg-green-600"
            disabled={isUpdating}
          >
            {isUpdating ? (
              "Updating..."
            ) : (
              <>
                <Download className="h-3 w-3 mr-1" /> Update
              </>
            )}
          </Button>
        )}
        {!isLatest && operatingSystem == "macos" && (
          // display a popover with instructions on how to update the software (code snippet)
          <Popover>
            <PopoverTrigger>
              <Button
                size="sm"
                className="text-xs h-6 px-2 py-0 bg-green-500 hover:bg-green-600"
              >
                Update
              </Button>
            </PopoverTrigger>
            <PopoverContent className="min-w-[30rem] p-4 flex flex-col gap-2 text-muted-foreground">
              Run this command in a terminal to update the software:
              <CodeSnippet
                title="Update phosphobot"
                code={`brew update && brew upgrade phosphobot
# Check version
phosphobot --version`}
                language="bash"
                showLineNumbers={false}
              />
              If updating fails, try to reinstall the software.
              <CodeSnippet
                title="Reinstall phosphobot"
                code={`brew uninstall phosphobot && brew install phosphobot`}
                language="bash"
                showLineNumbers={false}
              />
            </PopoverContent>
          </Popover>
        )}
        {/* Windows update instructions */}
        {!isLatest && operatingSystem == "windows" && (
          <Popover>
            <PopoverTrigger>
              <Button
                size="sm"
                className="text-xs h-6 px-2 py-0 bg-green-500 hover:bg-green-600"
              >
                Update
              </Button>
            </PopoverTrigger>
            <PopoverContent className="min-w-[30rem] p-4 flex flex-col gap-2 text-muted-foreground">
              <p>
                Use this command in a PowerShell terminal to update phosphobot:
              </p>
              <CodeSnippet
                title="Update phosphobot"
                code={`powershell -ExecutionPolicy ByPass -Command "irm https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.ps1 | iex"`}
                language="powershell"
                showLineNumbers={false}
              />
              <p>
                Alternatively, replace your phosphobot.exe file with the latest
                one
              </p>
              <div className="flex justify-center mt-2">
                <Button
                  onClick={() =>
                    window.open(
                      "https://github.com/phospho-app/homebrew-phosphobot/releases/latest",
                      "_blank",
                    )
                  }
                  className="flex items-center gap-2"
                >
                  <GithubIcon className="size-3" />
                  Download on Github <ExternalLink className="h-3 w-3" />
                </Button>
              </div>
            </PopoverContent>
          </Popover>
        )}
      </div>
    </footer>
  );
}
