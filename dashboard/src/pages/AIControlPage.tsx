import { AIControlDisclaimer } from "@/components/common/ai-control-disclaimer";
import { AutoComplete, type Option } from "@/components/common/autocomplete";
import CameraKeyMapper from "@/components/common/camera-mapping-selector";
import CameraSelector from "@/components/common/camera-selector";
import { SpeedSelect } from "@/components/common/speed-select";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { AIStatusResponse, ServerStatus, TrainingsList } from "@/types";
import {
  CameraIcon,
  CameraOff,
  ExternalLink,
  HelpCircle,
  LoaderCircle,
  Pause,
  Play,
  Square,
  TestTubeDiagonal,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

type ModelConfiguration = {
  video_keys: string[];
  checkpoints: string[];
};

export function AIControlPage() {
  const [prompt, setPrompt] = useState("");
  const modelId = useGlobalStore((state) => state.modelId);
  const setModelId = useGlobalStore((state) => state.setModelId);

  const [showCassette, setShowCassette] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<number | null>(
    null,
  );
  const location = useLocation();
  const leaderArmSerialIds = useGlobalStore(
    (state) => state.leaderArmSerialIds,
  );
  const showCamera = useGlobalStore((state) => state.showCamera);
  const setShowCamera = useGlobalStore((state) => state.setShowCamera);
  const cameraKeysMapping = useGlobalStore((state) => state.cameraKeysMapping);

  const modelsThatRequirePrompt = ["gr00t", "ACT_BBOX", "pi0.5", "smolvla"];
  const selectedModelType = useGlobalStore((state) => state.selectedModelType);
  const setSelectedModelType = useGlobalStore(
    (state) => state.setSelectedModelType,
  );
  const selectedAngleFormat = useGlobalStore(
    (state) => state.selectedAngleFormat,
  );
  const setSelectedAngleFormat = useGlobalStore(
    (state) => state.setSelectedAngleFormat,
  );
  const minAngle = useGlobalStore((state) => state.minAngle);
  const setMinAngle = useGlobalStore((state) => state.setMinAngle);
  const maxAngle = useGlobalStore((state) => state.maxAngle);
  const setMaxAngle = useGlobalStore((state) => state.setMaxAngle);
  const selectedCameraId = useGlobalStore((state) => state.selectedCameraId);
  const setSelectedCameraId = useGlobalStore(
    (state) => state.setSelectedCameraId,
  );

  const { data: modelConfiguration } = useSWR<ModelConfiguration>(
    modelId ? ["/model/configuration", modelId, selectedModelType] : null,
    ([url]) =>
      fetcher(url, "POST", {
        model_id: modelId,
        model_type: selectedModelType,
      }),
  );
  const { data: trainedModels } = useSWR<TrainingsList>(
    ["/training/models/read"],
    ([endpoint]) => fetcher(endpoint, "POST"),
  );

  const { data: serverStatus, mutate: mutateServerStatus } =
    useSWR<ServerStatus>(["/status"], fetcher);
  const { data: aiStatus, mutate: mutateAIStatus } = useSWR<AIStatusResponse>(
    ["/ai-control/status"],
    ([arg]) => fetcher(arg, "POST"),
    { refreshInterval: 1000 },
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
    setModelId("");
    setSelectedCheckpoint(null);
  }, [selectedModelType, setModelId, setSelectedCheckpoint]);

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

    const response = await fetchWithBaseUrl("/ai-control/start", "POST", {
      prompt,
      model_id: modelId,
      speed,
      robot_serials_to_ignore,
      cameras_keys_mapping: cameraKeysMapping,
      model_type: selectedModelType,
      selected_camera_id: selectedCameraId,
      checkpoint: selectedCheckpoint,
      angle_format: selectedAngleFormat,
      min_angle: selectedAngleFormat === "other" ? minAngle : undefined,
      max_angle: selectedAngleFormat === "other" ? maxAngle : undefined,
    });

    if (!response) {
      setShowCassette(false);
      mutateAIStatus();
      // Call the /ai-control/stop endpoint to reset the AI control status
      await fetchWithBaseUrl("/ai-control/stop", "POST");
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
    mutateServerStatus();
  };

  const stopControl = async () => {
    const data = await fetchWithBaseUrl("/ai-control/stop", "POST");

    if (!data) return;

    mutateAIStatus({
      ...aiStatus,
      status: "stopped",
    });
    mutateServerStatus();
    toast.success("AI control stopped successfully");
  };

  const pauseControl = async () => {
    const data = await fetchWithBaseUrl("/ai-control/pause", "POST");

    if (!data) return;

    mutateAIStatus({
      ...aiStatus,
      status: "paused",
    });
    mutateServerStatus();
    toast.success("AI control paused successfully");
  };

  const resumeControl = async () => {
    const data = await fetchWithBaseUrl("/ai-control/resume", "POST");

    if (!data) return;

    mutateAIStatus({
      ...aiStatus,
      status: "running",
    });
    mutateServerStatus();
    toast.success("AI control resumed successfully");
  };

  return (
    <div className="container mx-auto py-8 max-w-4xl">
      <Card>
        <CardContent className="space-y-4 pt-6">
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
              <ToggleGroupItem value="pi0.5">pi0.5</ToggleGroupItem>
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
                    key={selectedModelType}
                    placeholder="nvidia/GR00T-N1.5-3B"
                    className="w-full"
                    disabled={aiStatus?.status !== "stopped"}
                    emptyMessage="Make sure this is a public model available on Hugging Face."
                  />
                  {modelConfiguration?.checkpoints && (
                    <Select
                      value={
                        selectedCheckpoint !== null
                          ? selectedCheckpoint.toString()
                          : "main"
                      }
                      onValueChange={(value) => {
                        if (value === "main") {
                          setSelectedCheckpoint(null);
                        } else {
                          setSelectedCheckpoint(parseInt(value, 10));
                        }
                        console.log("Selected checkpoint:", value);
                      }}
                      disabled={aiStatus?.status !== "stopped"}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select checkpoint" />
                      </SelectTrigger>
                      <SelectContent>
                        {modelConfiguration?.checkpoints?.map((checkpoint) => (
                          <SelectItem key={checkpoint} value={checkpoint}>
                            Checkpoint {checkpoint}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                  <Button variant="outline" asChild>
                    <a
                      href={
                        selectedModelType === "gr00t"
                          ? "https://huggingface.co/models?other=gr00t_n1"
                          : selectedModelType === "smolvla"
                          ? "https://huggingface.co/models?other=smolvla"
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

              {/* The user should select the format of the angles he recorded: degrees, radians or custom */}
              <div className="flex flex-col gap-y-2">
                <div className="text-xs text-muted-foreground">
                  Select angle units in the original dataset
                </div>
                <div className="flex items-center gap-2">
                  <ToggleGroup
                    type="single"
                    value={selectedAngleFormat}
                    onValueChange={setSelectedAngleFormat}
                  >
                    <ToggleGroupItem value="rad" className="flex-none">
                      <div className="flex items-center w-full">
                        <TestTubeDiagonal className="mr-1 h-4 w-4 text-green-500" />
                        Radians (default)
                      </div>
                    </ToggleGroupItem>
                    <ToggleGroupItem value="degrees">Degrees</ToggleGroupItem>
                    <ToggleGroupItem value="other">Other</ToggleGroupItem>
                  </ToggleGroup>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          <span className="font-semibold">
                            phosphobot records data in radians.{" "}
                          </span>
                          <span>
                            Use Radians if you recorded your dataset with
                            phosphobot.
                          </span>
                        </p>
                        <p>
                          LeRobot records data either between [-100, 100],
                          either in degrees.
                        </p>
                        <p>
                          Unsure about the units of your dataset? Use{" "}
                          <a
                            href="https://lerobot-visualize-dataset.hf.space/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
                          >
                            the dataset visualizer.
                          </a>
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                {/* If other is selected, we show a min and max value for denormalization */}
                {selectedAngleFormat === "other" && (
                  <div className="flex flex-col gap-y-2">
                    <div className="text-xs text-muted-foreground">
                      Select min and max values for denormalization
                    </div>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        placeholder="Min"
                        value={minAngle}
                        onChange={(e) => {
                          const value = e.target.value;
                          if (value === "" || value === "-") {
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            setMinAngle(value as any);
                          } else {
                            const numValue = Number(value);
                            if (!isNaN(numValue)) {
                              setMinAngle(numValue);
                            }
                          }
                        }}
                      />
                      <Input
                        type="number"
                        placeholder="Max"
                        value={maxAngle}
                        onChange={(e) => {
                          const value = e.target.value;
                          if (value === "" || value === "-") {
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            setMaxAngle(value as any);
                          } else {
                            const numValue = Number(value);
                            if (!isNaN(numValue)) {
                              setMaxAngle(numValue);
                            }
                          }
                        }}
                      />
                    </div>
                  </div>
                )}
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
                          <div className="flex items-center gap-2 flex-row">
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
                      <CameraKeyMapper
                        modelKeys={modelConfiguration?.video_keys}
                      />
                    )}
                  </AccordionContent>
                </AccordionItem>
              </Accordion>

              <div className="space-y-2 mt-2">
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
                        selectedModelType === "ACT_BBOX"
                          ? "eg. 'red ball', 'plushy', 'green cube'"
                          : "eg. 'Pick up the red ball and place it in the box.'"
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
                    disabled={
                      aiStatus?.status !== "stopped" ||
                      !modelId.trim() ||
                      !modelConfiguration ||
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
            <div className="bg-muted p-6 rounded-lg">
              <div className="flex flex-col items-center space-y-4">
                {/* Message top of cassette */}
                <div className="text-center mb-2">
                  <Badge variant={"outline"} className="text-sm px-3 py-1">
                    AI state: {aiStatus?.status}
                    {aiStatus?.status === "waiting" && (
                      // add spinner
                      <LoaderCircle className="inline-block size-4 animate-spin" />
                    )}
                  </Badge>
                </div>

                <div className="flex justify-center gap-4">
                  <Button
                    size="lg"
                    variant="default"
                    className={`h-16 w-16 rounded-full ${
                      aiStatus?.status === "stopped" ||
                      aiStatus?.status === "paused"
                        ? "bg-green-500 hover:bg-green-600"
                        : "bg-muted-foreground cursor-not-allowed"
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
                        : "bg-muted-foreground cursor-not-allowed"
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
                        ? "bg-red-500 hover:bg-red-600"
                        : "bg-muted-foreground cursor-not-allowed"
                    }`}
                    onClick={stopControl}
                    disabled={aiStatus?.status === "stopped"}
                    title="Stop AI control"
                  >
                    <Square className="h-8 w-8" />
                    <span className="sr-only">Stop</span>
                  </Button>
                </div>

                <div className="text-xs text-center mt-2 text-muted-foreground">
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
