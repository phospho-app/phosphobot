"use client";

import placeholderSvg from "@/assets/placeholder.svg";
import { AutoComplete } from "@/components/common/autocomplete";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { Loader2, TrafficCone } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import useSWR, { mutate } from "swr";

// Data model for robot types
const ROBOT_TYPES = [
  {
    id: "phosphobot",
    name: "Remote phosphobot server",
    category: "manipulator",
    image: placeholderSvg,
    fields: [
      { name: "ip", label: "IP Address", type: "ip" },
      { name: "port", label: "Port", type: "number", default: 80 },
      { name: "robot_id", label: "Robot ID", type: "number", default: 0 },
    ],
  },
  {
    id: "unitree-go2",
    name: "Unitree Go2",
    category: "mobile",
    image: placeholderSvg,
    fields: [{ name: "ip", label: "IP Address", type: "ip" }],
  },
  {
    id: "so-100",
    name: "SO-100 / SO-101",
    category: "manipulator",
    image: placeholderSvg,
    fields: [{ name: "device_name", label: "USB Port", type: "device_name" }],
  },
  {
    id: "koch-v1.1",
    name: "Koch 1.1",
    category: "manipulator",
    image: placeholderSvg,
    fields: [{ name: "device_name", label: "USB Port", type: "device_name" }],
  },
  {
    id: "lekiwi",
    name: "LeKiwi",
    category: "mobile",
    image: placeholderSvg,
    fields: [
      { name: "ip", label: "IP Address", type: "ip" },
      { name: "port", label: "Port", type: "number", default: 5555 },
    ],
  },
  {
    id: "urdf_loader",
    name: "URDF loader",
    category: "manipulator",
    image: placeholderSvg,
    fields: [
      { name: "urdf_path", label: "URDF Path", type: "urdf_path" },
      {
        name: "end_effector_link_index",
        label: "End Effector Link Index",
        type: "number",
      },
      {
        name: "gripper_joint_index",
        label: "Gripper Joint Index",
        type: "number",
      },
    ],
  },
];

interface RobotConfigModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}
interface NetworkDevice {
  ip: string;
  mac: string;
}

interface NetworkReponse {
  devices: NetworkDevice[];
}

interface LocalDevice {
  name: string;
  device: string;
  serial_number?: string;
  pid?: number;
  interface?: string;
}

interface LocalResponse {
  devices: LocalDevice[];
}

export function RobotConfigModal({
  open,
  onOpenChange,
}: RobotConfigModalProps) {
  const [selectedRobotType, setSelectedRobotType] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [formValues, setFormValues] = useState<Record<string, any>>({});

  const {
    urdfPath,
    setUrdfPath,
    urdfPathHistory,
    addUrdfPathToHistory,
    endEffectorLinkIndex,
    setEndEffectorLinkIndex,
    gripperJointIndex,
    setGripperJointIndex,
  } = useGlobalStore();

  const selectedRobot = ROBOT_TYPES.find(
    (robot) => robot.id === selectedRobotType,
  );

  // Fetch IP addresses for autocomplete
  const { data: networkDevices, isLoading: isLoadingDevices } =
    useSWR<NetworkReponse>(["/network/scan-devices"], ([endpoint]) =>
      fetcher(endpoint, "POST"),
    );

  // Fetch USB ports for autocomplete
  const { data: usbPorts, isLoading: isLoadingUsb } = useSWR<LocalResponse>(
    ["/local/scan-devices"],
    ([endpoint]) => fetcher(endpoint, "POST"),
  );

  const handleRobotTypeChange = (value: string) => {
    setSelectedRobotType(value);
    // Initialize form values with defaults when robot type changes
    const robot = ROBOT_TYPES.find((r) => r.id === value);
    if (robot) {
      const defaultValues = robot.fields.reduce(
        (acc, field) => {
          if (field.default !== undefined) {
            acc[field.name] = field.default;
          } else if (field.name === "urdf_path" && urdfPath) {
            // Pre-populate URDF path from store
            acc[field.name] = urdfPath;
          } else if (field.name === "end_effector_link_index") {
            // Pre-populate end effector link index from store
            acc[field.name] = endEffectorLinkIndex;
          } else if (field.name === "gripper_joint_index") {
            // Pre-populate gripper joint index from store
            acc[field.name] = gripperJointIndex;
          }
          return acc;
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        {} as Record<string, any>,
      );
      setFormValues(defaultValues);
    } else {
      setFormValues({});
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleFieldChange = (fieldName: string, value: any) => {
    setFormValues((prev) => ({
      ...prev,
      [fieldName]: value,
    }));

    // Also persist to store for specific fields
    if (fieldName === "end_effector_link_index") {
      setEndEffectorLinkIndex(parseInt(value) || 1);
    } else if (fieldName === "gripper_joint_index") {
      setGripperJointIndex(parseInt(value) || 1);
    }
  };

  const handleSubmit = async () => {
    if (!selectedRobot) return;

    // Check if all required fields are filled
    const missingFields = selectedRobot.fields.filter(
      (field) =>
        formValues[field.name] === undefined && field.default === undefined,
    );

    if (missingFields.length > 0) {
      toast.error(
        `Please fill in all required fields: ${missingFields.map((f) => f.label).join(", ")}`,
      );
      return;
    }
    setIsSubmitting(true);

    // Create the proper form:
    // {ip: formValues.ip, port: formValues.port, ...}
    const connectionDetails = selectedRobot.fields.reduce(
      (acc, field) => {
        // Use form value if provided, otherwise use default if available
        const fieldValue =
          formValues[field.name] !== undefined
            ? formValues[field.name]
            : field.default;

        if (fieldValue !== undefined) {
          // if fieldValue is also an object with a value property, get that
          acc[field.name] =
            typeof fieldValue === "object" && fieldValue.value
              ? fieldValue.value
              : fieldValue;
        }
        return acc;
      },
      {} as Record<string, string | number>,
    );
    console.log("Connection details:", connectionDetails);

    try {
      // Prepare payload
      const payload = {
        robot_name: selectedRobotType,
        connection_details: connectionDetails,
      };

      // Call API to add robot
      const response = await fetchWithBaseUrl(
        "/robot/add-connection",
        "POST",
        payload,
      );

      if (response) {
        toast.success(
          `${selectedRobot.name} robot has been added successfully.`,
        );

        // Add URDF path to history if it's a URDF loader
        if (selectedRobotType === "urdf_loader" && urdfPath) {
          addUrdfPathToHistory(urdfPath);
        }

        // Close modal on success
        onOpenChange(false);

        // Reset form
        setSelectedRobotType("");
        setFormValues({});

        // mutate /status endpoint
        mutate("/status");
      }
    } catch (error) {
      console.error("Error adding robot:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Connect to another robot</DialogTitle>
          <DialogDescription className="flex flex-col gap-y-2">
            <div>
              Manually connect to a robot by selecting its type and entering the
              connection details.
            </div>
            <div className="border border-destructive bg-destructive/10 text-destructive rounded-md p-2">
              <TrafficCone className="inline mr-2 size-6" />
              This feature is experimental and may not work as expected. Please
              report any issue you encounter{" "}
              <a
                href="https://discord.gg/cbkggY6NSK"
                target="_blank"
                className="underline"
              >
                on Discord!
              </a>
            </div>
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 py-2">
          <div className="grid grid-cols-[2fr_1fr] gap-4 items-start">
            <div className="space-y-2">
              <Label htmlFor="robot-type">Robot Type</Label>
              <Select
                value={selectedRobotType}
                onValueChange={handleRobotTypeChange}
              >
                <SelectTrigger id="robot-type">
                  <SelectValue placeholder="Select robot type" />
                </SelectTrigger>
                <SelectContent>
                  {ROBOT_TYPES.map((robot) => (
                    <SelectItem key={robot.id} value={robot.id}>
                      {robot.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedRobot && (
              <div className="flex flex-col items-center justify-center">
                <div className="relative h-[120px] w-[120px] rounded-md border overflow-hidden">
                  <img
                    src={selectedRobot.image || "/placeholder.svg"}
                    alt={selectedRobot.name}
                    className="object-cover w-[120px] h-[120px]"
                  />
                </div>
                <span className="text-xs text-muted-foreground mt-1">
                  {selectedRobot.category === "mobile"
                    ? "Mobile Unit"
                    : "Manipulator"}
                </span>
              </div>
            )}
          </div>

          {selectedRobot && (
            <div className="space-y-4">
              {selectedRobot.fields.map((field) => (
                <div key={field.name} className="space-y-2">
                  <Label htmlFor={field.name}>{field.label}</Label>

                  {field.type === "ip" && (
                    <AutoComplete
                      options={
                        networkDevices?.devices.map((device) => ({
                          value: device.ip,
                          label: `${device.ip} (${device.mac})`,
                        })) || []
                      }
                      value={formValues[field.name]}
                      onValueChange={(value) =>
                        handleFieldChange(field.name, value)
                      }
                      isLoading={isLoadingDevices}
                      placeholder="Select or enter IP address"
                      emptyMessage="No IP addresses found"
                      allowCustomValue={true}
                    />
                  )}

                  {field.type === "device_name" && (
                    <AutoComplete
                      options={
                        usbPorts?.devices.map((device) => {
                          let label = `${device.device}`;
                          if (device.serial_number) {
                            label += ` (${device.serial_number}`;
                          }
                          if (device.pid) {
                            label += ` | ${device.pid}`;
                          }
                          // add closing parenthesis if it was opened
                          if (label.includes("(")) {
                            label += ")";
                          }
                          return {
                            value: device.device,
                            label: label,
                          };
                        }) || []
                      }
                      value={formValues[field.name]}
                      onValueChange={(value) =>
                        handleFieldChange(field.name, value)
                      }
                      isLoading={isLoadingUsb}
                      placeholder="Select USB port"
                      emptyMessage="No USB ports detected"
                      allowCustomValue={true}
                    />
                  )}

                  {field.type === "urdf_path" && (
                    <AutoComplete
                      options={urdfPathHistory.map((path) => ({
                        value: path,
                        label: path,
                      }))}
                      value={
                        urdfPath
                          ? { value: urdfPath, label: urdfPath }
                          : undefined
                      }
                      onValueChange={(option) => {
                        const path = option.value;
                        setUrdfPath(path);
                        handleFieldChange(field.name, path);
                      }}
                      placeholder="Enter or select URDF path"
                      emptyMessage="No recent URDF paths"
                      allowCustomValue={true}
                    />
                  )}

                  {field.type === "number" && (
                    <Input
                      id={field.name}
                      type="number"
                      placeholder={
                        field.default !== undefined
                          ? `Default: ${field.default}`
                          : "Enter number"
                      }
                      value={
                        formValues[field.name] !== undefined
                          ? formValues[field.name]
                          : ""
                      }
                      onChange={(e) =>
                        handleFieldChange(field.name, e.target.value)
                      }
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!selectedRobot || isSubmitting}
            className="min-w-[120px]"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Connecting...
              </>
            ) : (
              "Add Robot"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
