import { AddZMQCameraModal } from "@/components/common/add-zmq-camera-modal";
import { CameraStreamCard } from "@/components/common/camera-stream-card";
import { Button } from "@/components/ui/button";
import { useCameraControls } from "@/lib/hooks";
import { cn, fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { AdminSettings, ServerStatus } from "@/types";
import { RotateCw, Video } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";

export function ViewVideoPage({ labelText }: { labelText?: string }) {
  if (!labelText) labelText = "Camera Stream";
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isZMQModalOpen, setIsZMQModalOpen] = useState(false);

  const { data: serverStatus, mutate: mutateStatus } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 5000,
    },
  );

  const { data: adminSettings, mutate: mutateSettings } = useSWR<AdminSettings>(
    "/admin/settings",
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const { updateCameraRecording, isCameraEnabled } = useCameraControls(
    adminSettings,
    mutateSettings,
  );

  return (
    <>
      <div className="mb-2 flex justify-end gap-x-2">
        <Button
          variant="outline"
          className="ml-2"
          onClick={() => setIsZMQModalOpen(true)}
        >
          Add ZMQ Camera
        </Button>
        <Button
          variant="outline"
          onClick={() => {
            setIsRefreshing(true);
            fetchWithBaseUrl("/cameras/refresh", "POST").then(() => {
              mutateStatus();
              mutateSettings();
              setIsRefreshing(false);
            });
          }}
          disabled={isRefreshing}
        >
          <div className="flex items-center gap-2">
            <RotateCw
              className={cn("h-4 w-4", isRefreshing && "animate-spin")}
            />
            Rescan cameras...
          </div>
        </Button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {isRefreshing && (
          <div className="col-span-1 sm:col-span-2 md:col-span-3 text-center text-muted-foreground">
            <p>
              Disconnecting the camera streams and restarting camera
              discovery...
            </p>
          </div>
        )}
        {!isRefreshing &&
          serverStatus?.cameras.video_cameras_ids.map((cameraId) => {
            return (
              <CameraStreamCard
                key={cameraId}
                id={cameraId}
                title={`Camera ${cameraId}`}
                streamPath={`/video/${cameraId}`}
                alt={`Video Stream ${cameraId}`}
                icon={<Video className="h-4 w-4" />}
                isRecording={isCameraEnabled(cameraId)}
                onRecordingToggle={updateCameraRecording}
                showRecordingControls={true}
                labelText={labelText}
              />
            );
          })}
      </div>
      <AddZMQCameraModal
        open={isZMQModalOpen}
        onOpenChange={setIsZMQModalOpen}
      />
    </>
  );
}
