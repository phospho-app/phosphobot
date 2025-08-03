import { PhosphoProCallout } from "@/components/callout/phospho-pro";
import { AIControlDisclaimer } from "@/components/common/ai-control-disclaimer";
import { HuggingFaceKeyInput } from "@/components/common/huggingface-key";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuth } from "@/context/AuthContext";
import { fetcher } from "@/lib/utils";
import { AdminTokenSettings, ServerStatus } from "@/types";
import {
  AlertTriangle,
  Bot,
  BrainCircuit,
  Camera,
  Code,
  Dumbbell,
  FileCog,
  FolderOpen,
  LoaderCircle,
  Network,
  Play,
  Settings,
  Sliders,
} from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import useSWR from "swr";

function RobotStatusAlert({
  serverStatus,
  isLoading,
  robotConnected,
}: {
  serverStatus?: ServerStatus;
  isLoading: boolean;
  robotConnected: boolean;
}) {
  if (isLoading) {
    return (
      <Alert>
        <AlertTitle className="flex flex-row gap-1 items-center">
          <LoaderCircle className="animate-spin size-5 mr-1" />
          Status: Loading
        </AlertTitle>
        <AlertDescription>Loading robot status...</AlertDescription>
      </Alert>
    );
  }

  if (!serverStatus) {
    return (
      <Alert>
        <AlertTitle className="flex flex-row gap-1 items-center">
          <span className="size-2 rounded-full bg-red-500" />
          <Bot className="size-5 mr-1" />
          Status: Communication Error
        </AlertTitle>
        <AlertDescription>
          Error fetching robot status. Please check the server connection.
        </AlertDescription>
      </Alert>
    );
  }

  if (robotConnected) {
    return (
      <Alert>
        <AlertTitle className="flex flex-row gap-1 items-center">
          <span className="size-2 rounded-full bg-green-500" />
          <Bot className="size-5 mr-1" />
          Status: Connected
        </AlertTitle>
        <AlertDescription>
          Robot is connected and ready to control.
        </AlertDescription>
      </Alert>
    );
  } else {
    return (
      <Alert>
        <AlertTitle className="flex flex-row gap-1 items-center">
          <span className="size-2 rounded-full bg-red-500" />
          <Bot className="size-5 mr-1" />
          Status: Disconnected
        </AlertTitle>
        <AlertDescription>
          Check the robot is plugged to your computer and powered on. Unplug and
          plug cables again if needed.
        </AlertDescription>
      </Alert>
    );
  }
}

function AIModelsCard() {
  const [showWarning, setShowWarning] = useState(false);

  const navigate = useNavigate();

  const { data: adminSettingsTokens, isLoading } = useSWR<AdminTokenSettings>(
    ["/admin/settings/tokens"],
    ([url]) => fetcher(url, "POST"),
  );

  const handleControlByAI = () => {
    if (localStorage.getItem("disclaimer_accepted") === "true") {
      navigate(`/inference`);
      return;
    }
    // Otherwise display the warning dialog
    setShowWarning(true);
  };

  const onProceed = () => {
    setShowWarning(false);
    localStorage.setItem("disclaimer_accepted", true.toString());
    navigate(`/inference`);
  };

  return (
    <>
      <Card className="flex justify-between md:min-h-[25vh]">
        <CardContent className="flex flex-col md:flex-row py-6">
          <div className="flex-1 md:flex-1/3 mb-4 md:mb-0">
            <div className="flex items-center gap-2 text-xl font-semibold mb-2">
              <BrainCircuit className="text-green-500" />
              AI Training and Control
            </div>
            <div className="text-xs text-muted-foreground">
              Teach your robot new skills. Control your robot with Artificial
              Intelligence.
            </div>
          </div>
          <div className="flex-1 md:flex-2/3">
            <div className="flex flex-col gap-y-4">
              {!isLoading && !adminSettingsTokens?.huggingface && (
                <div className="mb-4">
                  <HuggingFaceKeyInput />
                </div>
              )}

              <div className="flex flex-col md:flex-row gap-2">
                <Tooltip>
                  <TooltipTrigger className="flex-1/2" asChild>
                    <Button
                      variant="outline"
                      onClick={() => {
                        navigate("/train");
                      }}
                      disabled={!adminSettingsTokens?.huggingface}
                    >
                      <Dumbbell className="size-5" />
                      Train an AI Model
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <div>
                      Once you have recorded a dataset, you can train an AI
                      model. Make sure you have a HuggingFace account and a
                      valid API key.
                    </div>
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger className="flex-1/2" asChild>
                    <Button
                      onClick={handleControlByAI}
                      disabled={!adminSettingsTokens?.huggingface}
                    >
                      <BrainCircuit className="size-5" />
                      Go to AI Control
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <div>
                      After training your AI model, let your AI model control
                      the robot.
                    </div>
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      <Dialog open={showWarning} onOpenChange={setShowWarning}>
        <DialogContent className="sm:max-w-md border-amber-300 border">
          <DialogHeader className="bg-amber-50 dark:bg-amber-950/20 p-4 -m-4 rounded-t-lg">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="size-16 text-red-500 mr-2" />
              <DialogTitle className="text-bold font-bold tracking-tight">
                You are about to surrender control to an artificial intelligence
                system.
              </DialogTitle>
            </div>
          </DialogHeader>

          <AIControlDisclaimer />

          <DialogFooter className="gap-x-2 mt-2">
            <Button
              variant="outline"
              onClick={() => setShowWarning(false)}
              className="border-gray-200 hover:bg-gray-50"
            >
              Cancel
            </Button>
            <Button variant="default" onClick={onProceed}>
              I Understand the Risks
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();
  const { proUser } = useAuth();
  const { data: serverStatus, isLoading } = useSWR<ServerStatus>(
    ["/status"],
    ([url]) => fetcher(url),
    {
      refreshInterval: 5000,
    },
  );
  const robotConnected =
    serverStatus !== undefined &&
    serverStatus.robots &&
    serverStatus.robots.length > 0;

  return (
    <div className="flex flex-col gap-4">
      {/* Phospho Pro Callout */}
      {!proUser && <PhosphoProCallout />}
      {/* Control */}
      <Card className="flex justify-between md:min-h-[25vh]">
        <CardContent className="w-full flex flex-row gap-4">
          <div className="flex-1/3">
            <div className="flex items-center gap-2 text-xl font-semibold mb-2">
              <Play className="text-green-500" />
              Control and Record
            </div>
            <div className="text-xs text-muted-foreground">
              Control the robot with your keyboard, a leader arm, or a VR
              headset. Record and replay movements. Record datasets.
            </div>
          </div>

          <div className="flex-2/3">
            <div className="mb-2 flex flex-col md:flex-row gap-2">
              <div className="flex-1/2">
                <RobotStatusAlert
                  serverStatus={serverStatus}
                  isLoading={isLoading}
                  robotConnected={robotConnected}
                />
              </div>

              <div className="flex-1/2">
                <Button
                  variant="default"
                  className="w-full h-full"
                  disabled={!robotConnected}
                  onClick={() => {
                    if (!robotConnected) return;
                    navigate("/control");
                  }}
                >
                  <div className="flex items-center gap-2">
                    <Play className="text-green-500" />
                    Control
                  </div>
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <Button asChild variant="outline">
                <a href="/browse">
                  <FolderOpen className="size-5" />
                  Browse your Datasets
                </a>
              </Button>
              <Button asChild variant="outline">
                <a href="/calibration">
                  <Sliders className="size-5" />
                  Calibration
                </a>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Models */}
      <AIModelsCard />

      {/* Advanced Settings */}
      <Card className="flex justify-between md:min-h-[25vh]">
        <CardContent className="flex justify-between">
          <div className="flex-1/3">
            <div className="flex items-center gap-2 text-xl font-semibold mb-2">
              <Settings className="text-green-500" />
              Advanced Settings
            </div>
            <div className="text-xs text-muted-foreground">
              Configure the server and the robot settings.
            </div>
          </div>
          <div className="flex-2/3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <Button asChild variant="secondary">
                <a href="/admin">
                  <FileCog className="size-5" />
                  Admin Configuration
                </a>
              </Button>
              <Button asChild variant="secondary">
                <a href="/docs">
                  <Code className="size-5" />
                  API Documentation
                </a>
              </Button>

              <Button asChild variant="outline">
                <a href="/viz">
                  <Camera className="size-5" />
                  Camera Overview
                </a>
              </Button>
              <Button asChild variant="outline">
                <a href="/network">
                  <Network className="size-5" />
                  Network Management
                </a>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
