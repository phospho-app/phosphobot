import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore } from "@/lib/hooks";
import { fetcher } from "@/lib/utils";
import { fetchWithBaseUrl } from "@/lib/utils";
import type { AdminSettings, ServerStatus } from "@/types";
import {
  AlertCircle,
  Camera,
  CameraOff,
  CheckCircle2,
  Database,
  Ellipsis,
  HelpCircle,
  Repeat,
  Save,
  Settings,
  Square,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

interface RecordingSettings {
  dataset_name: string;
  task_instruction: string;
}

export function Recorder({
  showCamera,
  setShowCamera,
}: {
  showCamera: boolean;
  setShowCamera: (showCamera: boolean) => void;
}) {
  const navigate = useNavigate();
  const leaderArmSerialIds = useGlobalStore(
    (state) => state.leaderArmSerialIds,
  );
  const { data: serverStatus, mutate: refreshStatus } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 2000, // Poll every 5 seconds
      revalidateOnFocus: true,
    },
  );

  // Set the recording state based on server status
  const isRecording = serverStatus?.is_recording || false;
  const [isPlaying, setIsPlaying] = useState(false);

  // Recording settings state
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const [formSubmitted, setFormSubmitted] = useState(false);
  const [formError, setFormError] = useState("");
  const [validationErrors, setValidationErrors] = useState<
    Record<string, string>
  >({});

  // Fetch recording settings
  const { data: adminSettings, mutate: mutateSettings } = useSWR<AdminSettings>(
    "/admin/settings",
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  // Local state for form values
  const [localSettings, setLocalSettings] = useState<RecordingSettings>({
    dataset_name: "",
    task_instruction: "",
  });

  // Update local state when settings are fetched
  useEffect(() => {
    if (adminSettings) {
      setLocalSettings({
        dataset_name: adminSettings.dataset_name,
        task_instruction: adminSettings.task_instruction,
      });
    }
  }, [adminSettings]);

  // Validation functions
  const validateDatasetName = (value: string) => {
    const validDatasetPattern = /^[a-zA-Z0-9._-]+$/;
    if (!value) {
      return "Dataset name is required";
    }
    if (!validDatasetPattern.test(value)) {
      return "Dataset name can only contain letters, numbers, ., _, -";
    }
    return "";
  };

  // Handle settings change
  const handleSettingChange = (key: keyof RecordingSettings, value: string) => {
    // Validate based on field
    let error = "";
    if (key === "dataset_name") {
      error = validateDatasetName(value);
    }

    setValidationErrors((prev) => ({
      ...prev,
      [key]: error,
    }));

    setLocalSettings((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  // Handle form submission
  const handleSaveSettings = async () => {
    // Check if there are any validation errors
    const datasetNameError = validateDatasetName(localSettings.dataset_name);
    setValidationErrors({
      dataset_name: datasetNameError,
    });

    if (datasetNameError) {
      setFormError("Please fix validation errors before saving");
      return;
    }

    try {
      const response = await fetch("/admin/form/usersettings", {
        method: "POST",
        body: JSON.stringify({
          dataset_name: localSettings.dataset_name,
          task_instruction: localSettings.task_instruction,
          // Keep other settings unchanged
          ...(adminSettings && {
            freq: adminSettings.freq,
            episode_format: adminSettings.episode_format,
            video_codec: adminSettings.video_codec,
            video_size: adminSettings.video_size,
            cameras_to_record: adminSettings.cameras_to_record,
          }),
        }),
        headers: { "Content-Type": "application/json" },
      });

      if (response.ok) {
        mutateSettings();
        setFormSubmitted(true);
        setFormError("");
        setTimeout(() => {
          setFormSubmitted(false);
          setIsPopoverOpen(false);
        }, 2000);
      } else {
        const error = await response.json();
        setFormError(error.message || "Error saving settings.");
      }
    } catch (error) {
      console.error(error);
      setFormError("An error occurred while saving settings.");
    }
  };

  const handleRecordStart = async () => {
    if (!isRecording) {
      const robot_serials_to_ignore = leaderArmSerialIds ?? null;
      console.log(
        `Starting recording. Ignoring robots: ${robot_serials_to_ignore}`,
      );
      await fetchWithBaseUrl(`/recording/start`, "POST", {
        robot_serials_to_ignore: robot_serials_to_ignore,
        leader_arm_ids: leaderArmSerialIds,
      });
      refreshStatus(); // Refresh status after operation
    }
  };

  const handleRecordStop = async () => {
    if (isPlaying) {
      // TODO: If you press STOP while playing a recording, it should stop the playback
      // For now, do nothing.
      return;
    }

    const data = await fetchWithBaseUrl(`/recording/stop`, "POST", {
      save: true,
    });
    if (data) {
      toast.success(
        `Recording stopped. Episode saved in ${data.episode_folder_path}`,
      );
    }
    refreshStatus(); // Refresh status after operation
  };

  const handleRecordDiscard = async () => {
    const data = await fetchWithBaseUrl(`/recording/stop`, "POST", {
      save: false,
    });
    if (data) {
      toast.info("Recording discarded");
    }
    refreshStatus(); // Refresh status after operation
  };

  const handleRecordPlay = async () => {
    if (isRecording || isPlaying) {
      return;
    }
    setIsPlaying(true);
    toast.info("Starting replay...");
    const data = await fetchWithBaseUrl(`/recording/play`, "POST", {
      episode_path: null,
      robot_serials_to_ignore: leaderArmSerialIds ?? null,
    }).then((data) => {
      setIsPlaying(false);
      if (!data) return false;
      return true;
    });

    if (data) {
      toast.success("Replay finished");
    }
  };

  const handleSettingsClick = () => {
    // This function will be called when "More Settings"is clicked
    navigate("/admin");
  };

  return (
    <TooltipProvider>
      <div className="flex flex-col items-center gap-2">
        <div className="flex justify-center gap-x-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={handleRecordStart}
                variant="outline"
                className="flex flex-row items-center justify-center"
                disabled={isPlaying || isRecording}
              >
                <div className="size-4 rounded-full bg-red-500" />
                <span className="ml-1">REC</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Start recording robot movements</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={handleRecordStop}
                variant="outline"
                className="flex flex-row items-center justify-center"
                disabled={!isRecording}
              >
                <Square className="size-4 mr-1" />
                <span>STOP</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Stop and save the current recording</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={handleRecordDiscard}
                variant="outline"
                className="flex flex-row items-center justify-center"
                disabled={!isRecording}
              >
                <X className="size-4 mr-1" />
                <span>DISCARD</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Discard the current recording</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={handleRecordPlay}
                variant="outline"
                className="flex flex-row items-center justify-center"
                disabled={
                  isRecording ||
                  isPlaying ||
                  serverStatus?.leader_follower_status ||
                  serverStatus?.ai_running_status === "running"
                }
              >
                <Repeat className="size-4 mr-1" />
                <span>REPLAY</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Replay the last recorded sequence</p>
              {serverStatus?.leader_follower_status && (
                <p>Stop controlling to replay the episode.</p>
              )}
            </TooltipContent>
          </Tooltip>

          <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
            <Tooltip>
              <TooltipTrigger asChild>
                <PopoverTrigger asChild>
                  <Button
                    variant="default"
                    className="flex flex-row items-center justify-center"
                  >
                    <Settings className="size-4" />
                  </Button>
                </PopoverTrigger>
              </TooltipTrigger>
              <TooltipContent>
                <p>Dataset Settings</p>
              </TooltipContent>
            </Tooltip>
            <PopoverContent className="w-80">
              <div className="space-y-4">
                <h3 className="font-medium text-lg flex gap-x-2 items-center">
                  <Database className="size-5 text-primary" />
                  Dataset Settings
                </h3>

                <div className="space-y-2">
                  <div className="flex items-center">
                    <Label htmlFor="dataset-name" className="mr-2">
                      Dataset Name
                    </Label>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle className="size-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent side="right">
                        <p>Change this to create a new dataset</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Input
                    id="dataset-name"
                    placeholder="Enter dataset name"
                    value={localSettings.dataset_name}
                    onChange={(e) =>
                      handleSettingChange("dataset_name", e.target.value)
                    }
                  />
                  {validationErrors.dataset_name && (
                    <p className="text-xs text-red-500">
                      {validationErrors.dataset_name}
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <div className="flex items-center">
                    <Label htmlFor="task-instruction" className="mr-2">
                      Task Instructions
                    </Label>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle className="size-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent side="right">
                        <p>
                          Newly recorded tasks will be labeled with this
                          instruction.
                        </p>
                        <p>
                          You can have multiple task instructions in the same
                          dataset.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Textarea
                    id="task-instruction"
                    placeholder="Enter task instructions"
                    value={localSettings.task_instruction}
                    onChange={(e) =>
                      handleSettingChange("task_instruction", e.target.value)
                    }
                    className="min-h-[80px] resize-y"
                  />
                </div>

                {formError && (
                  <Alert variant="destructive" className="py-2">
                    <AlertCircle className="size-4" />
                    <AlertTitle className="text-xs">Error</AlertTitle>
                    <AlertDescription className="text-xs">
                      {formError}
                    </AlertDescription>
                  </Alert>
                )}

                {formSubmitted && (
                  <Alert
                    variant="default"
                    className="py-2 bg-emerald-50 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800"
                  >
                    <CheckCircle2 className="size-4" />
                    <AlertTitle className="text-xs">Success</AlertTitle>
                    <AlertDescription className="text-xs">
                      Your settings have been saved successfully.
                    </AlertDescription>
                  </Alert>
                )}

                <div className="flex justify-between pt-2 gap-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleSettingsClick}
                  >
                    <Ellipsis className="size-4 mr-2" />
                    More Settings
                  </Button>
                  <Button
                    onClick={handleSaveSettings}
                    size="sm"
                    className="flex-grow"
                  >
                    <Save className="size-4 mr-2" />
                    Save
                  </Button>
                </div>
              </div>
            </PopoverContent>
          </Popover>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                className="flex flex-row items-center justify-center cursor-pointer"
                onClick={() => setShowCamera(!showCamera)}
              >
                {!showCamera && <Camera className="size-4" />}
                {showCamera && <CameraOff className="size-4" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Camera settings</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
    </TooltipProvider>
  );
}
