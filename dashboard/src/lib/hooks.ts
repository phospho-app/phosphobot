import { fetchWithBaseUrl } from "@/lib/utils";
import type { AdminSettings } from "@/types";
import { useEffect, useState } from "react";
import { useCallback } from "react";
import { toast } from "sonner";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export function useFetchCode(url: string) {
  const [code, setCode] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchCode = async () => {
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const content = await response.text();
        setCode(content);
      } catch (err) {
        setError(err as Error);
        setCode(null);
      } finally {
        setLoading(false);
      }
    };

    fetchCode();
  }, [url]);

  return { code, loading, error };
}

interface GlobalStore {
  leaderArmSerialIds: string[];
  setLeaderArmSerialIds: (ids: string[]) => void;
  addLeaderArmSerialId: (armId: string) => void;
  removeLeaderArmSerialId: (armId: string) => void;
  showCamera: boolean;
  setShowCamera: (showCamera: boolean) => void;
  cameraKeysMapping: Record<string, number> | null;
  setCameraKeysMapping: (mapping: Record<string, number> | null) => void;
  modelId: string;
  setModelId: (modelId: string) => void;
  selectedModelType: "pi0.5" | "ACT" | "ACT_BBOX" | "gr00t" | "smolvla" | "custom";
  setSelectedModelType: (
    modelType: "pi0.5" | "ACT" | "ACT_BBOX" | "gr00t" | "smolvla" | "custom",
  ) => void;
  selectedAngleFormat: "degrees" | "radians" | "other";
  setSelectedAngleFormat: (
    angleFormat: "degrees" | "radians" | "other",
  ) => void;
  minAngle: number;
  setMinAngle: (minAngle: number) => void;
  maxAngle: number;
  setMaxAngle: (maxAngle: number) => void;
  selectedDataset: string;
  setSelectedDataset: (dataset: string) => void;
  selectedCameraId: number;
  setSelectedCameraId: (cameraId: number) => void;
  urdfPath: string;
  setUrdfPath: (path: string) => void;
  urdfPathHistory: string[];
  addUrdfPathToHistory: (path: string) => void;
  endEffectorLinkIndex: number;
  setEndEffectorLinkIndex: (index: number) => void;
  gripperJointIndex: number;
  setGripperJointIndex: (index: number) => void;
  // New fields for ZMQ server configuration
  zmqServerUrl: string;
  setZmqServerUrl: (url: string) => void;
  zmqTopic: string;
  setZmqTopic: (topic: string) => void;
  urdfUseZmq: boolean;
  setUrdfUseZmq: (useZmq: boolean) => void;
}

const useGlobalStore = create(
  persist<GlobalStore>(
    (set) => ({
      leaderArmSerialIds: [],
      setLeaderArmSerialIds: (ids) => set(() => ({ leaderArmSerialIds: ids })),
      addLeaderArmSerialId: (armId) =>
        set((state) => ({
          leaderArmSerialIds: state.leaderArmSerialIds.includes(armId)
            ? state.leaderArmSerialIds
            : [...state.leaderArmSerialIds, armId],
        })),

      removeLeaderArmSerialId: (armId) =>
        set((state) => ({
          leaderArmSerialIds: state.leaderArmSerialIds.filter(
            (id) => id !== armId,
          ),
        })),
      showCamera: false,
      setShowCamera: (newShowCamera: boolean) =>
        set(() => ({
          showCamera: newShowCamera,
        })),
      cameraKeysMapping: null,
      setCameraKeysMapping: (mapping: Record<string, number> | null) =>
        set(() => ({
          cameraKeysMapping: mapping,
        })),
      modelId: "",
      setModelId: (modelName: string) =>
        set(() => ({
          modelId: modelName,
        })),
      selectedModelType: "ACT_BBOX",
      setSelectedModelType: (
        modelType: "pi0.5" | "ACT" | "ACT_BBOX" | "gr00t" | "smolvla" | "custom",
      ) =>
        set(() => ({
          selectedModelType: modelType,
        })),
      selectedAngleFormat: "radians",
      setSelectedAngleFormat: (angleFormat: "degrees" | "radians" | "other") =>
        set(() => ({
          selectedAngleFormat: angleFormat,
        })),
      minAngle: -3.14,
      setMinAngle: (minAngle: number) =>
        set(() => ({
          minAngle: minAngle,
        })),
      maxAngle: 3.14,
      setMaxAngle: (maxAngle: number) =>
        set(() => ({
          maxAngle: maxAngle,
        })),
      selectedDataset: "",
      setSelectedDataset: (dataset: string) =>
        set(() => ({
          selectedDataset: dataset,
        })),
      selectedCameraId: 0,
      setSelectedCameraId: (cameraId: number) =>
        set(() => ({
          selectedCameraId: cameraId,
        })),
      urdfPath: "",
      setUrdfPath: (path: string) =>
        set(() => ({
          urdfPath: path,
        })),
      urdfPathHistory: [],
      addUrdfPathToHistory: (path: string) =>
        set((state) => {
          const newHistory = [
            path,
            ...state.urdfPathHistory.filter((p) => p !== path),
          ].slice(0, 5);
          return {
            urdfPathHistory: newHistory,
          };
        }),
      endEffectorLinkIndex: 0,
      setEndEffectorLinkIndex: (index: number) =>
        set(() => ({
          endEffectorLinkIndex: index,
        })),
      gripperJointIndex: 0,
      setGripperJointIndex: (index: number) =>
        set(() => ({
          gripperJointIndex: index,
        })),
      zmqServerUrl: "tcp://localhost:5555",
      setZmqServerUrl: (url: string) => set({ zmqServerUrl: url }),
      zmqTopic: "observations",
      setZmqTopic: (topic: string) => set({ zmqTopic: topic }),
      urdfUseZmq: false,
      setUrdfUseZmq: (useZmq: boolean) => set({ urdfUseZmq: useZmq }),
    }),

    {
      name: "phosphobot-global-store",
      storage: createJSONStorage(() => sessionStorage),
    },
  ),
);

export function useCameraControls(
  adminSettings: AdminSettings | undefined,
  mutateSettings: (
    data?: AdminSettings,
    shouldRevalidate?: boolean,
  ) => Promise<AdminSettings | undefined>,
) {
  const updateCameraRecording = useCallback(
    async (cameraId: number, isRecording: boolean) => {
      if (!adminSettings) return;

      // Create a new array of cameras to record
      let newCamerasToRecord = [...(adminSettings.cameras_to_record || [])];

      if (isRecording) {
        // Add camera if not already in the list
        if (!newCamerasToRecord.includes(cameraId)) {
          newCamerasToRecord.push(cameraId);
        }
      } else {
        // Remove camera from the list
        newCamerasToRecord = newCamerasToRecord.filter((id) => id !== cameraId);
      }

      // Create updated settings
      const updatedSettings = {
        ...adminSettings,
        cameras_to_record: newCamerasToRecord,
      };

      try {
        // Optimistically update the UI
        await mutateSettings(updatedSettings, false);

        // Send the update to the server
        const result = await fetchWithBaseUrl(
          "/admin/form/usersettings",
          "POST",
          updatedSettings,
        );

        if (result?.status === "success") {
          toast.success(
            `Camera ${cameraId} ${isRecording ? "enabled" : "disabled"}`,
          );
          // Revalidate the data
          mutateSettings();
        }
      } catch (error) {
        // Revert on error
        mutateSettings();
        console.error("Failed to update camera status:", error);
        toast.error("Failed to update camera settings");
      }
    },
    [adminSettings, mutateSettings],
  );

  const isCameraEnabled = useCallback(
    (cameraId: number) => {
      // If cameras_to_record is null or undefined, all cameras should be enabled
      if (!adminSettings?.cameras_to_record) return true;
      return adminSettings.cameras_to_record.includes(cameraId);
    },
    [adminSettings],
  );

  return {
    updateCameraRecording,
    isCameraEnabled,
  };
}

export function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    // Initial check
    checkIsMobile();

    // Add event listener
    window.addEventListener("resize", checkIsMobile);

    // Clean up
    return () => {
      window.removeEventListener("resize", checkIsMobile);
    };
  }, []);

  return isMobile;
}

/**
 * A custom hook to manage state that is persisted in localStorage.
 * It syncs the state with localStorage on every change and loads the
 * initial state from localStorage on mount.
 *
 * @param key The key to use in localStorage.
 * @param defaultValue The default value to use if nothing is in localStorage.
 * @returns A state and a setter function, like React.useState.
 */
function useLocalStorageState<T>(
  key: string,
  defaultValue: T,
): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    // Check if running on the client side
    if (typeof window === "undefined") {
      return defaultValue;
    }
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error("Error reading from localStorage", error);
      return defaultValue;
    }
  });

  useEffect(() => {
    // This effect runs only on the client side
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error("Error writing to localStorage", error);
    }
  }, [key, value]);

  return [value, setValue];
}

export { useGlobalStore, useLocalStorageState };
