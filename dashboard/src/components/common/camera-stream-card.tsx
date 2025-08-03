import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { CameraOff, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type React from "react";

// Default parameters for the streams (these can be passed as props or come from config)
const defaultQuality = 8;
const highQuality = 80;

// Helper to compute stream URL based on current hostname, port, and query parameters
const getStreamUrl = (streamPath: string, quality: number) =>
  `http://${window.location.hostname}:${window.location.port}${streamPath}?quality=${quality}`;

export interface CameraStreamProps {
  id: number;
  title: string;
  streamPath: string;
  alt?: string;
  icon?: React.ReactNode;
  isRecording?: boolean;
  onRecordingToggle?: (id: number, isRecording: boolean) => void;
  showRecordingControls?: boolean;
  labelText?: string;
}

export const CardContentPiece = ({
  id,
  streamPath,
  alt,
  isRecording,
  showRecordingControls,
}: {
  id: number;
  streamPath: string;
  alt?: string;
  isRecording: boolean;
  showRecordingControls: boolean;
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleImageLoad = () => {
    setIsLoading(false);
    setHasError(false);
  };

  const handleImageError = () => {
    setIsLoading(false);
    setHasError(true);
  };

  useEffect(() => {
    const img = imgRef.current;
    return () => {
      if (img) {
        img.src = ""; // Disconnect when component unmounts
      }
    };
  }, []);

  return (
    <div className="relative bg-muted">
      {hasError && (
        <div className="flex items-center gap-1">
          <CameraOff className="size-6" />
          Stream Unavailable
        </div>
      )}
      {!isRecording && showRecordingControls && (
        <div>
          <div className="flex items-center gap-1">
            <X className="size-6" />
            Feed Disabled
          </div>
          <p className="text-sm">Enable this camera to view the feed</p>
        </div>
      )}
      <img
        id={`view-video-${id}`}
        ref={imgRef}
        src={getStreamUrl(streamPath, highQuality) || "/placeholder.svg"}
        alt={alt}
        className={`w-full h-full object-cover transition-opacity duration-300 ${
          isLoading || hasError || (!isRecording && showRecordingControls)
            ? "opacity-0"
            : "opacity-100"
        }`}
        onLoad={handleImageLoad}
        onError={handleImageError}
      />
    </div>
  );
};

export const CameraStreamCard = ({
  id,
  title,
  streamPath,
  alt = "Camera Stream",
  icon,
  isRecording = false,
  onRecordingToggle,
  showRecordingControls = false,
  labelText = "Record",
}: CameraStreamProps) => {
  const [quality, setQuality] = useState(defaultQuality);
  const toggleQuality = () => {
    setQuality(quality === defaultQuality ? highQuality : defaultQuality);
  };

  const handleRecordingChange = (checked: boolean) => {
    if (onRecordingToggle) {
      onRecordingToggle(id, checked);
    }
  };

  return (
    <Card className="overflow-hidden">
      {title !== "" && (
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl flex items-center gap-2">
            {icon}
            {title}
          </CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <CardContentPiece
          id={id}
          streamPath={streamPath}
          alt={alt}
          isRecording={isRecording}
          showRecordingControls={showRecordingControls}
        />
      </CardContent>
      <CardFooter className="justify-between">
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className="hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            onClick={toggleQuality}
          >
            Preview:{" "}
            {quality === defaultQuality ? "Low quality" : "High quality"}
          </Badge>
        </div>
        {showRecordingControls && (
          <div className="flex items-center gap-2">
            <Checkbox
              id={`record-${id}`}
              checked={isRecording}
              onCheckedChange={handleRecordingChange}
            />
            <label
              htmlFor={`record-${id}`}
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
            >
              {labelText}
            </label>
          </div>
        )}
      </CardFooter>
    </Card>
  );
};
