import { PhosphoProCallout } from "@/components/callout/phospho-pro";
import { AutoComplete, type Option } from "@/components/common/autocomplete";
import { CopyButton } from "@/components/common/copy-button";
import { LogStream } from "@/components/custom/LogsStream";
import { ModelsCard } from "@/components/custom/ModelsTable";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuth } from "@/context/AuthContext";
import { useGlobalStore } from "@/lib/hooks";
import { useLocalStorageState } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { AdminTokenSettings } from "@/types";
import { json, jsonParseLinter } from "@codemirror/lang-json";
import { linter } from "@codemirror/lint";
import { vscodeDark } from "@uiw/codemirror-theme-vscode";
import CodeMirror from "@uiw/react-codemirror";
import {
  CheckCircle2,
  Dumbbell,
  Globe,
  Lightbulb,
  Loader2,
  Lock,
  RotateCcw,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import useSWR, { mutate } from "swr";

const JsonEditor = ({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) => {
  const handleChange = useCallback(
    (val: string) => {
      onChange(val);
    },
    [onChange],
  );

  // The jsonParseLinter provides diagnostics for syntax errors
  const jsonLinter = linter(jsonParseLinter());

  return (
    <CodeMirror
      value={value}
      height="224px" // Corresponds to the original h-56
      theme={vscodeDark}
      extensions={[json(), jsonLinter]}
      onChange={handleChange}
      basicSetup={{
        lineNumbers: false,
        foldGutter: false,
        dropCursor: true,
        allowMultipleSelections: false,
        indentOnInput: true,
      }}
    />
  );
};
interface DatasetListResponse {
  pushed_datasets: string[];
  local_datasets: string[];
}

interface TrainingInfoResponse {
  status: "ok" | "error";
  message?: string;
  training_body: Record<string, unknown>;
}

export function AITrainingPage() {
  const { proUser } = useAuth();

  const selectedDataset = useGlobalStore((state) => state.selectedDataset);
  const setSelectedDataset = useGlobalStore(
    (state) => state.setSelectedDataset,
  );
  const setSelectedModelType = useGlobalStore(
    (state) => state.setSelectedModelType,
  );
  const [trainingState, setTrainingState] = useState<
    "idle" | "loading" | "success"
  >("idle");
  const selectedModelType = useGlobalStore((state) => state.selectedModelType);
  const { data: adminSettingsTokens } = useSWR<AdminTokenSettings>(
    ["/admin/settings/tokens"],
    ([url]) => fetcher(url, "POST"),
  );
  const { data: adminSettings } = useSWR(["/admin/settings"], fetcher);
  const { data: datasetsList } = useSWR<DatasetListResponse>(
    ["/dataset/list"],
    ([url]) => fetcher(url, "POST"),
  );
  const { data: datasetInfoResponse, isLoading: isDatasetInfoLoading } =
    useSWR<TrainingInfoResponse>(
      selectedModelType === "custom" || selectedDataset
        ? ["/training/info", selectedDataset, selectedModelType]
        : null,
      ([url]) =>
        fetcher(url, "POST", {
          model_id: selectedDataset,
          model_type: selectedModelType,
        }),
      {
        revalidateOnFocus: false,
        revalidateOnReconnect: false,
        refreshInterval: 0,
      },
    );

  const [currentLogFile, setCurrentLogFile] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState<boolean>(false);
  const [lightbulbOn, setLightbulbOn] = useState(false);

  // Create a unique key for localStorage based on dataset and model type
  const storageKey = `training-params-${selectedDataset}-${selectedModelType}`;

  // Track the previous storage key to detect changes
  const [prevStorageKey, setPrevStorageKey] = useState(storageKey);

  // Use localStorage state that persists across page refreshes but changes with dataset/model
  const [editableJson, setEditableJson] = useLocalStorageState(storageKey, "");

  // Clear localStorage and reset when storage key changes (dataset/model switch)
  useEffect(() => {
    if (prevStorageKey !== storageKey) {
      // Storage key changed, reset the editor to empty
      setEditableJson("");
      setPrevStorageKey(storageKey);
    }
  }, [storageKey, prevStorageKey, setEditableJson]);

  // Initialize editableJson when API data loads
  useEffect(() => {
    if (
      selectedModelType !== "custom" &&
      datasetInfoResponse?.training_body?.training_params &&
      editableJson === ""
    ) {
      // Only initialize if editableJson is empty (no stored data)
      const trainingParams = datasetInfoResponse.training_body.training_params;
      const jsonString = JSON.stringify(trainingParams, null, 2);
      setEditableJson(jsonString);
    } else if (
      selectedModelType === "custom" &&
      datasetInfoResponse?.training_body?.custom_command &&
      editableJson === ""
    ) {
      // For custom models, use the custom command as the initial JSON
      const jsonString = JSON.stringify(
        datasetInfoResponse.training_body,
        null,
        2,
      );
      setEditableJson(jsonString);
    }
  }, [
    datasetInfoResponse?.training_body?.training_params,
    datasetInfoResponse?.training_body?.custom_command,
    editableJson,
    setEditableJson,
    selectedModelType,
  ]);

  const handleTrainModel = async () => {
    if (selectedModelType !== "custom" && !selectedDataset) {
      toast.error("Please select a dataset to train the model.");
      return;
    }

    if (!adminSettingsTokens?.huggingface) {
      toast.error("Please set a valid Hugging Face token in the settings.", {
        duration: 5000,
      });
      return;
    }

    // Set loading state
    setTrainingState("loading");

    try {
      // Parse the edited training parameters JSON
      let trainingParams;
      try {
        trainingParams = JSON.parse(editableJson);
      } catch (error) {
        toast.error("Invalid JSON format. Please check your input: " + error, {
          duration: 5000,
        });
        setTrainingState("idle");
        return { success: false, error: "Invalid JSON format" };
      }

      // Add private training flag based on admin settings and PRO status
      const isPrivateTraining = proUser && adminSettings?.hf_private_mode;

      // Build the complete training body
      let response;
      if (selectedModelType !== "custom") {
        const trainingBody = {
          model_type: selectedModelType,
          dataset_name: selectedDataset,
          private_mode: isPrivateTraining,
          user_hf_token: null, // Always null - backend will handle token
          training_params: trainingParams,
        };

        // Send the edited JSON to the training endpoint
        response = await fetchWithBaseUrl(
          "/training/start",
          "POST",
          trainingBody,
        );
      } else {
        // For custom models, send the custom command directly
        const customCommand = trainingParams.custom_command || editableJson;
        response = await fetchWithBaseUrl("/training/start-custom", "POST", {
          custom_command: customCommand,
        });
      }

      if (!response) {
        setTrainingState("idle");
        return;
      }
      if (selectedModelType === "custom" && response.message) {
        setCurrentLogFile(response.message);
        setShowLogs(true);
      }

      setTrainingState("success");
      if (selectedModelType !== "custom") {
        toast.success(
          `Model training started! Check progress on Hugging Face.`,
        );
      } else {
        toast.success("Custom training job started! Check logs for details.");
      }
      // Automatically reset the Training State to idle after 500ms
      setTimeout(() => {
        setTrainingState("idle");
      }, 500);

      return { success: true };
    } catch (error) {
      console.error("Error starting training job:", error);
      setTrainingState("idle");

      const errorMessage =
        error instanceof Error
          ? error.message
          : "An error occurred while starting the training job. Please try again later.";

      toast.error(errorMessage, {
        duration: 5000,
      });

      return { success: false, error: errorMessage };
    }
  };

  return (
    <div className="container mx-auto py-2 flex flex-col gap-2">
      {!proUser && <PhosphoProCallout />}
      <Card className="w-full">
        <CardContent>
          <div className="flex flex-col md:flex-row gap-2 items-end">
            <div className="flex-1/2 flex flex-row md:flex-col gap-2 w-full">
              <div className="text-xs text-muted-foreground md:w-1/2">
                Dataset ID on Hugging Face:
              </div>
              <AutoComplete
                key="dataset-autocomplete"
                options={
                  datasetsList?.pushed_datasets.map((dataset) => ({
                    value: dataset,
                    label: dataset,
                  })) ?? []
                }
                value={{
                  value: selectedDataset,
                  label: selectedDataset,
                }}
                disabled={selectedModelType === "custom"}
                onValueChange={(option: Option) => {
                  setSelectedDataset(option.value);
                }}
                placeholder="e.g. username/dataset-name"
                className="w-full"
                emptyMessage="Make sure this is a public dataset available on Hugging Face."
              />
            </div>
            <div className="flex-1/4 flex flex-col gap-2 w-full mb-1">
              <div className="text-xs text-muted-foreground">
                Type of model to train:
              </div>
              <div className="flex items-center gap-2">
                <Select
                  defaultValue={selectedModelType}
                  onValueChange={(value) => {
                    setSelectedModelType(
                      value as
                        | "pi0.5"
                        | "gr00t"
                        | "ACT"
                        | "ACT_BBOX"
                        | "smolvla"
                        | "custom",
                    );
                    setLightbulbOn(true);
                  }}
                >
                  <SelectTrigger className="w-full border rounded-md p-2">
                    <SelectValue placeholder="Select model type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pi0.5">pi0.5 (new)</SelectItem>
                    <SelectItem value="ACT_BBOX">BB-ACT</SelectItem>
                    <SelectItem value="ACT">ACT</SelectItem>
                    <SelectItem value="smolvla">SmolVLA (new)</SelectItem>
                    <SelectItem value="gr00t">gr00t-n1.5 (updated)</SelectItem>
                    <SelectItem value="custom">Custom</SelectItem>
                  </SelectContent>
                </Select>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="p-0 w-8 h-8 flex-shrink-0"
                        onClick={() => setLightbulbOn(false)}
                      >
                        <Lightbulb
                          className={`size-5 ${
                            lightbulbOn ? "text-green-500 animate-pulse" : ""
                          }`}
                        />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent className="p-4">
                      <p className="font-bold">Training Tips</p>
                      <ul className="list-disc list-inside space-y-2 text-sm">
                        <li>
                          If your training fails with a{" "}
                          <code>Timeout error</code>, lower the number of steps
                          or epochs.
                        </li>
                        <li>
                          If your training fails with a{" "}
                          <code>Cuda out of memory error</code>, lower the batch
                          size.
                        </li>
                      </ul>
                      {selectedModelType === "ACT_BBOX" && (
                        <>
                          <p className="font-bold mt-3">BB-ACT Model Tips</p>
                          <ul className="list-disc list-inside space-y-2 text-sm">
                            <li>
                              Set <code>target_detection_instruction</code> to
                              the object you want to detect (e.g., "red lego
                              brick").
                            </li>
                            <li>
                              <code>image_key</code> should correspond to your
                              context camera's key.
                            </li>
                          </ul>
                        </>
                      )}
                      {selectedModelType === "custom" && (
                        <>
                          <p className="font-bold mt-3">
                            You have selected a custom model type.
                          </p>
                          <p className="text-sm">
                            Pressing the "Train AI model" will run the command
                            written. Use this to run any custom training script.
                          </p>
                        </>
                      )}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </div>
          </div>

          <div className="flex justify-between items-center">
            <div className="flex flex-row items-center justify-start gap-2">
              <div className="text-xs text-muted-foreground">
                Training parameters
              </div>
              {isDatasetInfoLoading && (
                <Loader2 className="size-4 animate-spin" />
              )}
            </div>
            <div className="flex flex-row items-center justify-end gap-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      onClick={() => {
                        // Clear localStorage for this dataset/model combination
                        localStorage.removeItem(storageKey);
                        // Reset the editableJson state
                        setEditableJson("");
                        // Refetch the training info data
                        mutate([
                          "/training/info",
                          selectedDataset,
                          selectedModelType,
                        ]);
                        toast.success("Training parameters reset to defaults");
                      }}
                    >
                      <RotateCcw className="h-3 w-3" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Reset</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <CopyButton
                text={editableJson}
                hint="Copy JSON"
                variant="ghost"
              />
            </div>
          </div>
          <div className="text-sm text-muted-foreground mt-2">
            {datasetInfoResponse?.status !== "error" && (
              <div className="w-full">
                <JsonEditor value={editableJson} onChange={setEditableJson} />
              </div>
            )}
            {datasetInfoResponse?.status === "error" &&
              !isDatasetInfoLoading && (
                <div className="text-red-500">
                  {datasetInfoResponse.message ||
                    "Error fetching dataset info."}
                </div>
              )}
          </div>

          <div className="flex gap-2 items-center mt-4">
            <Button
              variant="secondary"
              className="flex flex-1"
              onClick={handleTrainModel}
              disabled={
                (selectedModelType !== "custom" && !selectedDataset) ||
                trainingState === "loading" ||
                isDatasetInfoLoading ||
                datasetInfoResponse?.status === "error"
              }
            >
              {trainingState === "idle" && (
                <>
                  Train AI model
                  <Dumbbell className="size-4 mr-2" />
                </>
              )}
              {trainingState === "loading" && (
                <>
                  Starting training...
                  <Loader2 className="size-4 mr-2 animate-spin" />
                </>
              )}
              {trainingState === "success" && (
                <>
                  Training started
                  <CheckCircle2 className="size-4 mr-2 text-green-500" />
                </>
              )}
            </Button>

            {/* Privacy Status Icon */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  {proUser ? (
                    <a href="/admin">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-10 w-10 p-0 bg-transparent"
                      >
                        {adminSettings?.hf_private_mode ? (
                          <Lock className="size-4" />
                        ) : (
                          <Globe className="size-4" />
                        )}
                      </Button>
                    </a>
                  ) : (
                    <a
                      href="https://phospho.ai/pro"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-10 w-10 p-0 bg-transparent"
                      >
                        <Globe className="size-4" />
                      </Button>
                    </a>
                  )}
                </TooltipTrigger>
                <TooltipContent>
                  {proUser && adminSettings?.hf_private_mode
                    ? "Private training enabled - click to manage settings"
                    : proUser
                      ? "Public training - click to manage settings"
                      : "Public training - upgrade to PRO for private training"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          {selectedModelType === "custom" && (showLogs || currentLogFile) && (
            <LogStream
              logFile={currentLogFile}
              isLoading={trainingState === "loading"}
              onClose={() => setShowLogs(false)}
            />
          )}
        </CardContent>
      </Card>
      <ModelsCard />
    </div>
  );
}
