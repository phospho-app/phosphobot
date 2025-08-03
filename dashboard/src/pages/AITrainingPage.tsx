import { AutoComplete, type Option } from "@/components/common/autocomplete";
import { CopyButton } from "@/components/common/copy-button";
import { LogStream } from "@/components/custom/LogsStream";
import { ModelsCard } from "@/components/custom/ModelsDialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { AdminTokenSettings } from "@/types";
import {
  Ban,
  CheckCircle2,
  Dumbbell,
  Lightbulb,
  List,
  Loader2,
  Pencil,
  Save,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

// Add this after the existing imports
const JsonEditor = ({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const editorRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    try {
      // Format the JSON when it's not being edited
      if (!isEditing) {
        const parsed = JSON.parse(value);
        const formatted = JSON.stringify(parsed, null, 2);
        if (formatted !== value) {
          onChange(formatted);
        }
      }
    } catch (e) {
      console.error("Invalid JSON format:", e);
    }
  }, [value, isEditing, onChange]);

  const handleEdit = () => {
    setEditValue(value);
    setIsEditing(true);
    setTimeout(() => {
      editorRef.current?.focus();
    }, 0);
  };

  const handleSave = () => {
    try {
      // Try to parse to validate JSON
      JSON.parse(editValue);
      onChange(editValue);
      setIsEditing(false);
    } catch (e) {
      toast.error("Invalid JSON format. Please check your input: " + e, {
        duration: 5000,
      });
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
  };

  if (isEditing) {
    return (
      <div className="relative">
        <textarea
          ref={editorRef}
          className="w-full h-56 font-mono text-sm p-2 border border-gray-300 rounded"
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
        />
        <div className="absolute bottom-2 right-2 flex gap-2">
          <Button variant="outline" onClick={handleCancel}>
            <Ban className="size-4 mr-2" />
            Cancel
          </Button>
          <Button variant="default" onClick={handleSave}>
            <Save className="size-4 mr-2" />
            Save
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative group">
      <div className="cursor-pointer" onClick={handleEdit}>
        {formatJsonDisplay(value)}
      </div>
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="flex gap-x-2">
          <Button variant="outline" onClick={handleEdit}>
            <Pencil className="size-4" />
            Edit
          </Button>
          <CopyButton text={value} hint="Copy the json" variant="outline" />
        </div>
      </div>
    </div>
  );
};

// Add this helper function to format the JSON display
const formatJsonDisplay = (jsonString: string) => {
  try {
    const obj = JSON.parse(jsonString);
    return (
      <div className="text-left">
        {Object.entries(obj).map(([key, value]) => (
          <div key={key} className="mb-1">
            <span className="font-semibold text-green-500">{key}</span>
            <span className="text-muted-foreground">: </span>
            <span className="text-primary">
              {typeof value === "object"
                ? JSON.stringify(value, null, 2)
                : String(value)}
            </span>
          </div>
        ))}
      </div>
    );
  } catch (e) {
    console.log("Invalid JSON format:", e);
    return <div className="text-red-500">Invalid JSON format</div>;
  }
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
  const { data: datasetsList } = useSWR<DatasetListResponse>(
    ["/dataset/list"],
    ([url]) => fetcher(url, "POST"),
  );
  const { data: datasetInfoResponse, isLoading: isDatasetInfoLoading } =
    useSWR<TrainingInfoResponse>(
      selectedDataset || selectedModelType === "custom"
        ? ["/training/info", selectedDataset, selectedModelType]
        : null,
      ([url]) =>
        fetcher(url, "POST", {
          model_id: selectedDataset,
          model_type: selectedModelType,
        }),
    );

  const [editableJson, setEditableJson] = useState<string>("");
  const [currentLogFile, setCurrentLogFile] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState<boolean>(false);

  useEffect(() => {
    // Try to load from localStorage first
    const savedJson = localStorage.getItem("trainingBodyJson");
    if (savedJson) {
      setEditableJson(savedJson);
    }

    // Update from API response when it changes
    if (datasetInfoResponse?.training_body) {
      const jsonString = JSON.stringify(
        datasetInfoResponse.training_body,
        null,
        2,
      );
      setEditableJson(jsonString);
      // Save to localStorage
      localStorage.setItem("trainingBodyJson", jsonString);
    }
  }, [datasetInfoResponse]);

  const generateHuggingFaceModelName = async (dataset: string) => {
    // Model name followed by 10 random characters
    const randomChars = Math.random().toString(36).substring(2, 12);
    // Remove the name/... and replace with phospho-app/...
    const [, datasetName] = dataset.split("/");

    // Fetch whoami to get the username
    let middlePart = "";
    try {
      const result = await fetchWithBaseUrl(
        "/admin/huggingface/whoami",
        "POST",
      );
      // Check the status from the whoami response
      if (result.status === "success" && result.username) {
        // Include username in the model name if status is success
        middlePart = `${result.username}-${selectedModelType}-${datasetName}`;
      } else {
        // Fallback without username if status is not success
        middlePart = `${selectedModelType}-${datasetName}`;
      }
    } catch (error) {
      console.error("Error fetching whoami:", error);
      // Fallback without username in case of error
      middlePart = `${selectedModelType}-${datasetName}`;
    }
    // Ensure total model name doesn't exceed 96 characters
    // "phospho-app/" = 13 chars, "-" + randomChars = 11 chars, so middlePart max = 96 - 13 - 11 = 72
    const maxMiddleLength = 72;
    if (middlePart.length > maxMiddleLength) {
      middlePart = middlePart.substring(0, maxMiddleLength);
    }

    const modelName = `phospho-app/${middlePart}-${randomChars}`;
    return modelName;
  };

  const handleTrainModel = async () => {
    if (!selectedDataset) {
      toast.error("Please select a dataset to train the model.", {
        duration: 5000,
      });
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
      // Generate a random model name
      const modelName = await generateHuggingFaceModelName(selectedDataset);
      const modelUrl = `https://huggingface.co/${modelName}`;

      // Parse the edited JSON
      let trainingBody;
      try {
        trainingBody = JSON.parse(editableJson);
      } catch (error) {
        toast.error("Invalid JSON format. Please check your input: " + error, {
          duration: 5000,
        });
        setTrainingState("idle");
        return { success: false, error: "Invalid JSON format" };
      }

      // Send the edited JSON to the training endpoint
      const response = await fetchWithBaseUrl(
        selectedModelType !== "custom"
          ? "/training/start"
          : "/training/start-custom",
        "POST",
        trainingBody,
      );

      if (!response) {
        setTrainingState("idle");
        return;
      }

      if (selectedModelType === "custom" && response.message) {
        setCurrentLogFile(response.message);
      }

      // After successful notification, wait 1 second then show success
      await new Promise((resolve) => setTimeout(resolve, 1000));

      setTrainingState("success");
      if (selectedModelType !== "custom") {
        toast.success(
          `Model training started! Check progress at: ${modelUrl}`,
          {
            duration: 5000,
          },
        );
      } else {
        toast.success("Custom training job started! Check logs for details.", {
          duration: 5000,
        });
      }

      return { success: true, modelName };
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

  // Render button content based on training state
  const renderButtonContent = () => {
    switch (trainingState) {
      case "loading":
        return (
          <>
            <Loader2 className="size-5 mr-2 animate-spin" />
            Starting...
          </>
        );
      case "success":
        return (
          <>
            <CheckCircle2 className="size-5 mr-2 text-green-500" />
            Training job started
          </>
        );
      default:
        return (
          <>
            <Dumbbell className="size-5 mr-2" />
            Train AI model
          </>
        );
    }
  };

  return (
    <div className="container mx-auto py-8">
      <Tabs defaultValue="train">
        <div className="flex justify-between">
          <TabsList className="flex flex-col md:flex-row gap-4 border-1">
            <TabsTrigger value="train">
              <Dumbbell className="size-4 mr-2" />
              Train AI model
            </TabsTrigger>
            <TabsTrigger value="view">
              <List className="size-4 mr-2" />
              View trained models
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="train">
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
                <div className="flex-1/4 flex flex-row md:flex-col gap-2 w-full mb-1">
                  <div className="text-xs text-muted-foreground">
                    Type of model to train:
                  </div>
                  <Select
                    defaultValue={selectedModelType}
                    onValueChange={(value) =>
                      setSelectedModelType(
                        value as "gr00t" | "ACT" | "ACT_BBOX" | "smolvla" | "custom",
                      )
                    }
                  >
                    <SelectTrigger className="w-full border rounded-md p-2">
                      <SelectValue placeholder="Select model type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ACT_BBOX">
                        BB-ACT (recommended)
                      </SelectItem>
                      <SelectItem value="ACT">ACT</SelectItem>
                      <SelectItem value="gr00t">
                        gr00t-n1.5 (updated)
                      </SelectItem>
                      <SelectItem value="smolvla">SmolVLA</SelectItem>
                      <SelectItem value="custom">Custom</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              {selectedModelType === "custom" && (
                <div className="text-xs text-muted-foreground mt-4">
                  You have selected a custom model type.
                  <br />
                  Pressing the "Train AI model" will run the command written.
                  Use this to run any custom training script.
                </div>
              )}
              <div className="text-xs text-muted-foreground mt-4">
                Training parameters:
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                {isDatasetInfoLoading && (
                  <div className="flex flex-row items-center">
                    <Loader2 className="size-4 mr-2 animate-spin" />
                    Loading training parameters...
                  </div>
                )}
                {datasetInfoResponse?.status == "ok" &&
                  !isDatasetInfoLoading && (
                    <div className="bg-muted p-4 rounded-lg w-full h-64">
                      <pre className="font-mono text-sm whitespace-pre-wrap">
                        {editableJson ? (
                          <JsonEditor
                            value={editableJson}
                            onChange={(value) => {
                              setEditableJson(value);
                              localStorage.setItem("trainingBodyJson", value);
                            }}
                          />
                        ) : (
                          "No data available"
                        )}
                      </pre>
                    </div>
                  )}
                {datasetInfoResponse?.status == "error" &&
                  !isDatasetInfoLoading && (
                    <div className="text-red-500">
                      {datasetInfoResponse.message ||
                        "Error fetching dataset info."}
                    </div>
                  )}
              </div>

              {selectedModelType === "ACT_BBOX" && (
                <div className="text-xs text-muted-foreground mt-4">
                  This model works by recognizing objects in images.
                  <br />
                  Make sure to pass:
                  <br />
                  <code>target_detection_instruction</code> is the object you
                  want to detect in the images, e.g. "red lego brick", "blue
                  ball", "plushy toy", etc.
                  <br />
                  <code>image_key</code> corresponds to the key of your context
                  camera, which overviews the scene.
                </div>
              )}

              <Button
                variant="secondary"
                className="flex w-full mt-4"
                onClick={handleTrainModel}
                disabled={
                  !selectedDataset ||
                  trainingState !== "idle" ||
                  isDatasetInfoLoading ||
                  datasetInfoResponse?.status === "error"
                }
              >
                {renderButtonContent()}
              </Button>

              {selectedModelType === "custom" &&
                (showLogs || currentLogFile) && (
                  <LogStream
                    logFile={currentLogFile}
                    isLoading={trainingState === "loading"}
                    onClose={() => setShowLogs(false)}
                  />
                )}

              <div className="flex flex-row mt-4 items-center align-center">
                <Lightbulb className="size-4 mr-2 text-muted-foreground" />
                Tips
              </div>
              <div className="text-muted-foreground text-sm mt-2">
                - If your training fails with a <code>Timeout error</code>,
                lower the number of steps or epochs.
                <br />- If your training fails with a{" "}
                <code>Cuda out of memory error</code>, lower the batch size.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="view">
          <ModelsCard />
        </TabsContent>
      </Tabs>
    </div>
  );
}
