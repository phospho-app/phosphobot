import { HuggingFaceKeyInput } from "@/components/common/huggingface-key";
import { LoadingPage } from "@/components/common/loading";
import { WandBKeyInput } from "@/components/common/wandb-key";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { AdminSettings, AdminTokenSettings } from "@/types";
import { Camera, CircleCheck, Database, Key, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import useSWR from "swr";

export function AdminPage() {
  const [validationErrors, setValidationErrors] = useState<
    Record<string, string>
  >({});
  const isInitialMount = useRef(true);

  const { data: adminSettings, mutate } = useSWR<AdminSettings>(
    "/admin/settings",
    fetcher,
    { revalidateOnFocus: false, revalidateOnReconnect: false },
  );

  const { data: adminSettingsTokens } = useSWR<AdminTokenSettings>(
    ["/admin/settings/tokens"],
    ([url]) => fetcher(url, "POST"),
  );

  // Validation
  const validateDatasetName = (value: string) => {
    const validDatasetPattern = /^[a-zA-Z0-9._-]+$/;
    return validDatasetPattern.test(value)
      ? ""
      : "Dataset name can only contain letters, numbers, ., _, -";
  };
  const validateFrequency = (value: number) =>
    value > 0 ? "" : "Frequency must be greater than 0";
  const validateVideoSize = (w: number, h: number) =>
    w > 0 && h > 0 ? "" : "Video dimensions must be positive numbers";

  // Auto-save on changes
  useEffect(() => {
    if (!adminSettings) return;
    // skip initial mount
    if (isInitialMount.current) {
      isInitialMount.current = false;
    } else if (!Object.values(validationErrors).some((e) => e)) {
      fetchWithBaseUrl("/admin/form/usersettings", "POST", adminSettings);
    }
  }, [adminSettings, validationErrors]);

  const handleSettingChange = <K extends keyof AdminSettings>(
    key: K,
    value: AdminSettings[K],
  ) => {
    if (!adminSettings) return;
    let error = "";
    let [w, h] = adminSettings.video_size;

    switch (key) {
      case "dataset_name":
        error = validateDatasetName(value as string);
        break;
      case "freq":
        error = validateFrequency(value as number);
        break;
      case "video_size":
        [w, h] = value as [number, number];
        error = validateVideoSize(w, h);
        break;
    }

    setValidationErrors((prev) => ({ ...prev, [key]: error }));
    mutate({ ...adminSettings, [key]: value }, false);
  };

  if (!adminSettings) return <LoadingPage />;

  return (
    <div>
      {/* API Keys */}
      <Card className="mb-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5 text-primary" />
            API Key Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <HuggingFaceKeyInput />
          {adminSettingsTokens?.huggingface && (
            <div className="flex items-center gap-2 text-xs text-green-500">
              <CircleCheck className="h-4 w-4" /> Token set
            </div>
          )}
        </CardContent>
        <CardContent className="space-y-4">
          <WandBKeyInput />
          {adminSettingsTokens?.wandb && (
            <div className="flex items-center gap-2 text-xs text-green-500">
              <CircleCheck className="h-4 w-4" /> Token set
            </div>
          )}
        </CardContent>
      </Card>

      {/* Settings Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
        {/* Recording Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Play className="h-5 w-5 text-primary" /> Recording Settings
            </CardTitle>
            <CardDescription>Configure data recording</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="freq">Recording Frequency (Hz)</Label>
              <Input
                id="freq"
                type="number"
                value={adminSettings.freq}
                onChange={(e) => handleSettingChange("freq", +e.target.value)}
              />
              {validationErrors.freq && (
                <p className="text-red-500 text-sm">{validationErrors.freq}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label>Video Codec</Label>
              <Select
                value={adminSettings.video_codec}
                onValueChange={(v) => handleSettingChange("video_codec", v)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select codec" />
                </SelectTrigger>
                <SelectContent>
                  {["mp4v", "avc1", "hev1", "hvc1", "avc3", "av01", "vp09"].map(
                    (c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ),
                  )}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Video Size</Label>
              <div className="flex gap-2">
                <Input
                  placeholder="Width"
                  value={adminSettings.video_size[0]}
                  onChange={(e) =>
                    handleSettingChange("video_size", [
                      +e.target.value,
                      adminSettings.video_size[1],
                    ])
                  }
                />
                <Input
                  placeholder="Height"
                  value={adminSettings.video_size[1]}
                  onChange={(e) =>
                    handleSettingChange("video_size", [
                      adminSettings.video_size[0],
                      +e.target.value,
                    ])
                  }
                />
              </div>
              {validationErrors.video_size && (
                <p className="text-red-500 text-sm">
                  {validationErrors.video_size}
                </p>
              )}
            </div>
            <Button
              variant="outline"
              className="cursor-pointer"
              onClick={(e) => {
                e.preventDefault();
                window.location.href = "/viz";
              }}
            >
              <Camera className="size-4 mr-2" />
              Camera Settings
            </Button>
          </CardContent>
        </Card>

        {/* Dataset Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5 text-primary" /> Dataset Settings
            </CardTitle>
            <CardDescription>Configure dataset properties</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="dataset_name">Dataset Name</Label>
              <Input
                id="dataset_name"
                type="text"
                value={adminSettings.dataset_name}
                onChange={(e) =>
                  handleSettingChange("dataset_name", e.target.value)
                }
              />
              {validationErrors.dataset_name && (
                <p className="text-red-500 text-sm">
                  {validationErrors.dataset_name}
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="task_instruction">Task Instruction</Label>
              <Textarea
                id="task_instruction"
                value={adminSettings.task_instruction}
                onChange={(e) =>
                  handleSettingChange("task_instruction", e.target.value)
                }
                className="min-h-[80px] resize-y"
              />
            </div>
            <div className="space-y-2">
              <Label>Episode Format</Label>
              <Select
                value={adminSettings.episode_format}
                onValueChange={(v) =>
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  handleSettingChange("episode_format", v as any)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select format" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="lerobot_v2.1">lerobot_v2.1</SelectItem>
                  <SelectItem value="lerobot_v2">lerobot_v2</SelectItem>
                  <SelectItem value="json">json</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
