import { AIControlDisclaimer } from "@/components/common/ai-control-disclaimer";
import { AutoComplete, type Option } from "@/components/common/autocomplete";
import CameraKeyMapper from "@/components/common/camera-mapping-selector";
import CameraSelector from "@/components/common/camera-selector";
import { SpeedSelect } from "@/components/common/speed-select";
import supabase from "@/components/common/supabase-db";
import Feedback from "@/components/custom/Feedback";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuth } from "@/context/AuthContext";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { AIStatusResponse, ServerStatus, TrainingConfig } from "@/types";
import {
  CameraIcon,
  CameraOff,
  ExternalLink,
  HelpCircle,
  Pause,
  Play,
  Square,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

type ModelVideoKeys = {
  video_keys: string[];
};

export default function AIControlPage() {
  const [prompt, setPrompt] = useState("");
  const modelId = useGlobalStore((state) => state.modelId);
  const setModelId = useGlobalStore((state) => state.setModelId);

  const [showCassette, setShowCassette] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const location = useLocation();
  const { session } = useAuth();
  const leaderArmSerialIds = useGlobalStore(
    (state) => state.leaderArmSerialIds,
  );
  const showCamera = useGlobalStore((state) => state.showCamera);
  const setShowCamera = useGlobalStore((state) => state.setShowCamera);
  const cameraKeysMapping = useGlobalStore((state) => state.cameraKeysMapping);

  const modelsThatRequirePrompt = ["gr00t", "ACT_BBOX", "smolvla"];
  const selectedModelType = useGlobalStore((state) => state.selectedModelType);
  const setSelectedModelType = useGlobalStore(
    (state) => state.setSelectedModelType,
  );
  const selectedCameraId = useGlobalStore((state) => state.selectedCameraId);
  const setSelectedCameraId = useGlobalStore(
    (state) => state.setSelectedCameraId,
  );

  const { data: modelVideoKeys } = useSWR<ModelVideoKeys>(
    modelId ? ["/model/video-keys", modelId, selectedModelType] : null,
    ([url]) =>
      fetcher(url, "POST", {
        model_id: modelId,
        model_type: selectedModelType,
      }),
  );
  const { data: trainedModels } = useSWR<TrainingConfig>(
    ["/training/models/read"],
    ([endpoint]) => fetcher(endpoint, "POST"),
  );

  supabase.auth.setSession({
    access_token: session?.access_token || "",
    refresh_token: session?.refresh_token || "",
  });

  const { data: serverStatus, mutate: mutateServerStatus } =
    useSWR<ServerStatus>(["/status"], fetcher);
  const { data: aiStatus, mutate: mutateAIStatus } = useSWR<AIStatusResponse>(
    session ? ["/ai-control/status"] : null,
    ([arg]) =>
      fetcher(arg, "POST", { user_id: session?.user_id }).then((data) => {
        console.log("AI status data:", data);
        return data;
      }),
  );

  useEffect(() => {
    if (aiStatus !== undefined && aiStatus?.status !== "stopped") {
      setShowCassette(true);
    }
  }, [aiStatus, aiStatus?.status]);

  useEffect(() => {
    const initialPrompt = new URLSearchParams(location.search).get("prompt");
    if (initialPrompt) {
      setPrompt(initialPrompt);
    }
  }, [location.search]);

  useEffect(() => {
    // if no robots are connected, display toast message
    if (serverStatus?.robots.length === 0) {
      toast.warning("No robots are connected. AI control will not work.");
    }
  }, [serverStatus]);

  useEffect(() => {
    if (aiStatus === undefined) return;
    if (!aiStatus.id || !session) {
      if (!session) {
        toast.error("Please log in to access AI control sessions");
      }
      return;
    }

    console.log("Subscribing to AI control session:", aiStatus.id);

    const subscription = supabase
      .channel(`ai_control_sessions:${aiStatus.id}`)
      .on(
        "postgres_changes",
        {
          event: "UPDATE",
          schema: "public",
          table: "ai_control_sessions",
          filter: `id=eq.${aiStatus.id}`,
        },
        (payload) => {
          if (payload.new && "status" in payload.new) {
            const newStatus = payload.new.status;
            if (
              newStatus === null ||
              newStatus === "" ||
              newStatus === undefined
            ) {
              console.log("New status : ", newStatus);
              return;
            }

            console.log("AI control status updated:", newStatus);
            mutateAIStatus({
              ...aiStatus,
              status: newStatus,
            });
          }
        },
      )
      .subscribe((status) => {
        if (status === "SUBSCRIBED") {
          supabase
            .from("ai_control_sessions")
            .select("status, user_id")
            .eq("id", aiStatus.id)
            .maybeSingle()
            .then(({ data, error }) => {
              if (error) {
                console.error("Error fetching AI control status:", error);
                mutateAIStatus({
                  ...aiStatus,
                  status: "stopped",
                });
              } else if (data) {
                if (data.user_id !== session.user_id) {
                  toast.error("Access denied: Session belongs to another user");
                }
                console.log("AI control status:", data.status);
                mutateAIStatus({
                  ...aiStatus,
                  status: data.status,
                });
              }
            });
        }
      });

    return () => {
      supabase.removeChannel(subscription);
    };
  }, [aiStatus?.id, session, aiStatus, mutateAIStatus]);

  const startControlByAI = async () => {
    if (
      serverStatus?.robot_status?.length === 1 &&
      serverStatus.robot_status[0].device_name &&
      leaderArmSerialIds.includes(serverStatus.robot_status[0].device_name)
    ) {
      toast.warning(
        "Remove the leader arm mark on your robot to control it with AI",
      );
      return;
    }

    if (!modelId.trim()) {
      toast.error("Model ID cannot be empty");
      return;
    }
    if (!prompt.trim() && modelsThatRequirePrompt.includes(selectedModelType)) {
      toast.error("Prompt cannot be empty");
      return;
    }
    mutateAIStatus({
      ...aiStatus,
      status: "waiting",
    });
    setShowCassette(true);
    const robot_serials_to_ignore = leaderArmSerialIds ?? null;

    try {
      const response = await fetchWithBaseUrl("/ai-control/start", "POST", {
        prompt,
        model_id: modelId,
        speed,
        robot_serials_to_ignore,
        cameras_keys_mapping: cameraKeysMapping,
        model_type: selectedModelType,
        selected_camera_id: selectedCameraId,
      });

      if (!response) {
        setShowCassette(false);
        mutateAIStatus({
          ...aiStatus,
          status: "stopped",
        });
        return;
      }

      if (response.status === "error") {
        // We receive an error message if the control loop is already running
        setShowCassette(true);
        mutateAIStatus({
          ...aiStatus,
          id: response.ai_control_signal_id,
          status: response.ai_control_signal_status,
        });
        return;
      }

      mutateAIStatus({
        ...aiStatus,
        id: response.ai_control_signal_id,
        status: response.ai_control_signal_status,
      });
      console.log("AI control started successfully with id:", aiStatus?.id);
      mutateServerStatus();
      toast.success("Halfway there, we have started a GPU...");
      setTimeout(() => {
        toast.success("We are fetching your model, please wait...");
      }, 5000);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Network or server error";
      toast.error(`AI start failed: ${errorMessage}`);
      console.error("Start AI control error:", error);
      mutateAIStatus({
        ...aiStatus,
        status: "stopped",
      });
      setShowCassette(false);
    }
  };

  const stopControl = async () => {
    try {
      const response = await fetchWithBaseUrl("/ai-control/stop", "POST");

      if (!response) return;

      mutateAIStatus({
        ...aiStatus,
        status: "stopped",
      });
      mutateServerStatus();
      toast.success("AI control stopped successfully");
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Network or server error";
      toast.error(`AI stop failed: ${errorMessage}`);
      console.error("Stop AI control error:", error);
    }
  };

  const pauseControl = async () => {
    try {
      const response = await fetchWithBaseUrl("/ai-control/pause", "POST");

      if (!response) return;

      mutateAIStatus({
        ...aiStatus,
        status: "paused",
      });
      mutateServerStatus();
      toast.success("AI control paused successfully");
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Network or server error";
      toast.error(`AI pause failed: ${errorMessage}`);
      console.error("Pause AI control error:", error);
    }
  };

  const resumeControl = async () => {
    try {
      const response = await fetchWithBaseUrl("/ai-control/resume", "POST");

      if (!response) return;

      mutateAIStatus({
        ...aiStatus,
        status: "running",
      });
      mutateServerStatus();
      toast.success("AI control resumed successfully");
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Network or server error";
      toast.error(`AI continue failed: ${errorMessage}`);
      console.error("Continue AI control error:", error);
    }
  };

  return (
    <div className="container mx-auto py-8 max-w-4xl">
      <Card>
        <CardContent className="space-y-6 pt-6">
          <div className="flex flex-col gap-y-2">
            <div className="text-xs text-muted-foreground">
              Select model type
            </div>
            <ToggleGroup
              type="single"
              value={selectedModelType}
              onValueChange={setSelectedModelType}
            >
              <ToggleGroupItem value="ACT_BBOX">BB-ACT</ToggleGroupItem>
              <ToggleGroupItem value="gr00t">gr00t</ToggleGroupItem>
              <ToggleGroupItem value="ACT">ACT</ToggleGroupItem>
              <ToggleGroupItem value="smolvla">SmolVLA</ToggleGroupItem>
            </ToggleGroup>
          </div>

          {selectedModelType && (
            <>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label htmlFor="modelId">Model ID</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          Enter the Hugging Face model ID of your model. It
                          should be public.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="flex flex-col md:flex-row gap-2">
                  <AutoComplete
                    options={
                      // Filter out duplicate model names and sort by requested_at
                      trainedModels?.models
                        .filter(
                          (model) =>
                            model.model_type === selectedModelType &&
                            model.status === "succeeded",
                        )
                        .sort(
                          (a, b) =>
                            -a.requested_at.localeCompare(b.requested_at),
                        )
                        .filter(
                          (model, index, self) =>
                            index ===
                            self.findIndex(
                              (m) => m.model_name === model.model_name,
                            ),
                        )
                        .map((model) => ({
                          value: model.model_name,
                          label: model.model_name,
                        })) ?? []
                    }
                    value={{ value: modelId, label: modelId }}
                    onValueChange={(option: Option) => {
                      setModelId(option.value);
                    }}
                    placeholder="nvidia/GR00T-N1-2B"
                    className="w-full"
                    disabled={aiStatus?.status !== "stopped"}
                    emptyMessage="Make sure this is a public model available on Hugging Face."
                  />
                  <Button variant="outline" className="cursor-pointer" asChild>
                    <a
                      href={
                        selectedModelType === "gr00t"
                          ? "https://huggingface.co/models?other=gr00t_n1"
                          : "https://huggingface.co/models?other=act"
                      }
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Browse Models
                      <ExternalLink className="ml-2 h-4 w-4" />
                    </a>
                  </Button>
                </div>
              </div>

              <Accordion
                type="single"
                collapsible
                value={showCamera ? "camera-mapping" : ""}
              >
                <AccordionItem value="camera-mapping">
                  <TooltipProvider>
                    <Tooltip>
                      <AccordionTrigger
                        onClick={() => {
                          setShowCamera(!showCamera);
                        }}
                      >
                        <TooltipTrigger asChild>
                          <div className="cursor-pointer flex items-center gap-2 flex-row">
                            {showCamera ? (
                              <CameraOff className="mr-1 h-4 w-4" />
                            ) : (
                              <CameraIcon className="mr-1 h-4 w-4" />
                            )}
                            {showCamera
                              ? "Hide camera mapping settings"
                              : "Show cameras mapping settings"}
                          </div>
                        </TooltipTrigger>
                      </AccordionTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">The eyes of your robot.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <AccordionContent>
                    {selectedModelType === "ACT_BBOX" ? (
                      <CameraSelector
                        onCameraSelect={(cameraId) => {
                          setSelectedCameraId?.(cameraId);
                        }}
                        selectedCameraId={selectedCameraId}
                      />
                    ) : (
                      <CameraKeyMapper modelKeys={modelVideoKeys?.video_keys} />
                    )}
                  </AccordionContent>
                </AccordionItem>
              </Accordion>

              <div className="space-y-2">
                {selectedModelType == "gr00t" && <Label>Prompt</Label>}
                {selectedModelType === "ACT_BBOX" && (
                  <Label>Object to detect</Label>
                )}
                <div className="flex flex-col md:flex-row gap-2">
                  {modelsThatRequirePrompt.includes(selectedModelType) && (
                    <Input
                      id="prompt"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder={
                        selectedModelType === "gr00t"
                          ? "eg. 'Pick up the red ball and place it in the box.'"
                          : "eg. 'red ball', 'plushy', 'green cube'"
                      }
                      className="w-full"
                      disabled={aiStatus?.status !== "stopped"}
                    />
                  )}
                  <SpeedSelect
                    onChange={setSpeed}
                    defaultValue={1.0}
                    disabled={aiStatus?.status !== "stopped"}
                    title="Step Speed"
                  />
                  <Button
                    onClick={startControlByAI}
                    className="cursor-pointer"
                    disabled={
                      aiStatus?.status !== "stopped" ||
                      !modelId.trim() ||
                      (!prompt.trim() &&
                        modelsThatRequirePrompt.includes(selectedModelType))
                    }
                  >
                    <Play className="size-5 mr-2 text-green-500" />
                    Start AI control
                  </Button>
                </div>
              </div>
            </>
          )}

          {/* Cassette Player Style Control Panel */}
          {showCassette && (
            <div className="bg-gray-100 p-6 rounded-lg shadow-inner">
              <div className="flex flex-col items-center space-y-4">
                {/* Message top of cassette */}
                <div className="text-center mb-2">
                  <Badge variant={"outline"} className="text-sm px-3 py-1">
                    AI state: {aiStatus?.status}
                  </Badge>
                </div>

                <div className="flex justify-center gap-4">
                  <Button
                    size="lg"
                    variant="default"
                    className={`h-16 w-16 rounded-full ${
                      aiStatus?.status === "stopped" ||
                      aiStatus?.status === "paused"
                        ? "bg-green-600 hover:bg-green-700"
                        : "bg-gray-400 cursor-not-allowed"
                    }`}
                    onClick={
                      aiStatus?.status === "stopped"
                        ? startControlByAI
                        : aiStatus?.status === "paused"
                          ? resumeControl
                          : undefined
                    }
                    disabled={
                      (aiStatus?.status === "stopped" &&
                        !prompt.trim() &&
                        modelsThatRequirePrompt.includes(selectedModelType)) ||
                      aiStatus?.status === "running" ||
                      aiStatus?.status === "waiting"
                    }
                    title={
                      aiStatus?.status === "stopped"
                        ? "Start AI control"
                        : aiStatus?.status === "paused"
                          ? "Continue AI control"
                          : ""
                    }
                  >
                    <Play className="h-8 w-8" />
                    <span className="sr-only">
                      {aiStatus?.status === "stopped"
                        ? "Start"
                        : aiStatus?.status === "paused"
                          ? "Continue"
                          : "Play"}
                    </span>
                  </Button>

                  <Button
                    size="lg"
                    variant="default"
                    className={`h-16 w-16 rounded-full ${
                      aiStatus?.status === "running"
                        ? "bg-amber-500 hover:bg-amber-600"
                        : "bg-gray-400 cursor-not-allowed"
                    }`}
                    onClick={pauseControl}
                    disabled={aiStatus?.status !== "running"}
                    title="Pause AI control"
                  >
                    <Pause className="h-8 w-8" />
                    <span className="sr-only">Pause</span>
                  </Button>

                  <Button
                    size="lg"
                    variant="default"
                    className={`h-16 w-16 rounded-full ${
                      aiStatus?.status !== "stopped"
                        ? "bg-red-600 hover:bg-red-700"
                        : "bg-gray-400 cursor-not-allowed"
                    }`}
                    onClick={stopControl}
                    disabled={aiStatus?.status === "stopped"}
                    title="Stop AI control"
                  >
                    <Square className="h-8 w-8" />
                    <span className="sr-only">Stop</span>
                  </Button>
                </div>

                <div className="text-xs text-center text-gray-500 mt-2">
                  {aiStatus?.status === "stopped"
                    ? "Ready to start"
                    : aiStatus?.status === "paused"
                      ? "AI execution paused"
                      : aiStatus?.status === "waiting"
                        ? "AI getting ready, please don't refresh the page, this can take up to a minute..."
                        : "AI actively controlling robot"}
                </div>

                {aiStatus !== undefined &&
                  (aiStatus?.status === "running" ||
                    aiStatus?.status === "paused") && (
                    <div>
                      <div>How is the AI doing?</div>
                      <Feedback aiControlID={aiStatus.id} />
                    </div>
                  )}
              </div>
            </div>
          )}

          <Accordion type="single" collapsible>
            <AccordionItem value="item-1">
              <AccordionTrigger>AI Control Disclaimer</AccordionTrigger>
              <AccordionContent>
                <AIControlDisclaimer />
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>
    </div>
  );
}
