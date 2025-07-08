import { CodeSnippet } from "@/components/common/code-snippet";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { fetcher, fetchWithBaseUrl } from "@/lib/utils";
import type { ServerStatus } from "@/types";
import { Download, ExternalLink, Github } from "lucide-react";
import { useMemo, useState } from "react";
import useSWR from "swr";

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
    ([url]) => fetchWithBaseUrl(url, "POST"),
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
                const data = await fetchWithBaseUrl("/update/upgrade-to-latest-version", "POST");
                if (!data) throw new Error("Update failed");
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
            className="text-xs h-6 px-2 py-0 bg-green-500 hover:bg-green-600 cursor-pointer"
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
                className="text-xs h-6 px-2 py-0 bg-green-500 hover:bg-green-600 cursor-pointer"
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
                className="text-xs h-6 px-2 py-0 bg-green-500 hover:bg-green-600 cursor-pointer"
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
                  <Github className="h-3 w-3" />
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
