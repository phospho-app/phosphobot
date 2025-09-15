import position1 from "@/assets/CalibrationPosition1.jpg";
import position2 from "@/assets/CalibrationPosition2.jpg";
import { LoadingPage } from "@/components/common/loading";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { ServerStatus } from "@/types";
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  Home,
  Keyboard,
  Loader2,
  RotateCcw,
} from "lucide-react";
import { ReactNode, useState } from "react";
import { Link } from "react-router-dom";
import useSWR from "swr";

export function CalibrationSequence() {
  const { data: serverStatus, error: serverError } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 5000,
    },
  );

  const [step, setStep] = useState(1);
  const [calibrationStatus, setCalibrationStatus] = useState<
    "idle" | "loading" | "success" | "error" | "in_progress"
  >("idle");
  const [message, setMessage] = useState<ReactNode>("");
  const [selectedRobotName, setSelectedRobotName] = useState<string | null>(
    null,
  );

  const totalSteps = 3;

  const handleNextStep = async () => {
    if (step < totalSteps) {
      setStep((prevStep) => prevStep + 1);
    }

    setCalibrationStatus("loading");
    // robot_id is the index of the robot in the robot_status array
    const robot_id = serverStatus?.robot_status.findIndex(
      (robot) => robot.device_name === selectedRobotName,
    );
    if (robot_id === -1 || robot_id === undefined) {
      throw new Error(`Robot not found: ${robot_id}`);
    }
    const queryParam = new URLSearchParams({ robot_id: robot_id.toString() });
    const data = await fetchWithBaseUrl(
      `/calibrate?${queryParam.toString()}`,
      "POST",
    );
    setCalibrationStatus(data.calibration_status);
    // setMessage(data.message || "Calibration completed successfully!");
    if (data.calibration_status === "error") {
      setMessage(
        <>
          <p>An error occurred during calibration:</p>
          <code>{data.message || "Unknown error"}</code>
          <ul className="mt-2 list-disc pl-5 space-y-1">
            <li>
              Make the robot is connected to USB-C <u>and</u> to the power
              cable.
            </li>
            <li>
              If you have a <b>Torque Read error,</b> check that all the wires
              are fully plugged in the servomotors. Lose wires can cause this!
            </li>
            <li>
              Make sure your robot servomotors have the <b>same voltage</b> as
              the power cable (eg: 12V servos and 12V power cable, 7.2V servos
              and 5.2V power cable).
            </li>
            <li>
              Ensure your terminal is allowed to access the USB port. This is a
              common issue on Windows.
            </li>
            <li>
              If you have a <b>"Access is denied" error on Windows,</b> a common
              workaround is to{" "}
              <a
                href="https://docs.phospho.ai/installation#windows-wsl"
                target="_blank"
                className="underline"
              >
                install phosphobot with WSL (Windows Subsystem for Linux).
              </a>
            </li>
          </ul>
        </>,
      );
    }
  };

  const handleRestart = () => {
    setStep(1);
    setCalibrationStatus("idle");
    setMessage("");
  };

  const renderStepContent = (currentStep: number) => {
    if (calibrationStatus === "error") {
      return (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-5 w-5" />
          <AlertTitle>Calibration Failed</AlertTitle>
          <AlertDescription>{message}</AlertDescription>
        </Alert>
      );
    }

    switch (currentStep) {
      case 1:
        if (
          selectedRobotName === null &&
          serverStatus &&
          serverStatus.robot_status.length > 0 &&
          serverStatus.robot_status[0].device_name
        ) {
          setSelectedRobotName(serverStatus.robot_status[0].device_name);
        }

        return (
          <div className="flex flex-col gap-y-4">
            <h2 className="text-xl font-semibold">Prepare Your Robot</h2>
            <Alert
              variant="destructive"
              className="mb-6 flex items-start gap-3"
            >
              <div className="flex flex-col">
                <AlertTitle className="mb-1">
                  <div className="flex items-center gap-1">
                    <AlertTriangle className="size-5" />
                    Safety Warning
                  </div>
                </AlertTitle>
                <AlertDescription>
                  Make sure you can safely catch your robot. Calibration
                  disables torque.
                </AlertDescription>
              </div>
            </Alert>
            {serverStatus && serverStatus?.robot_status.length > 0 && (
              <div className="space-y-2">
                <Label htmlFor="follower-robot">Robot to calibrate</Label>
                <Select
                  value={selectedRobotName || ""}
                  onValueChange={(value) => setSelectedRobotName(value)}
                >
                  <SelectTrigger id="follower-robot">
                    <SelectValue placeholder="Select robot to calibrate" />
                  </SelectTrigger>
                  <SelectContent>
                    {serverStatus?.robot_status.map((robot) => (
                      <SelectItem
                        key={robot.device_name}
                        value={robot.device_name || "Undefined port"}
                      >
                        {robot.name} ({robot.device_name})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        );
      case 2:
        return (
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-4">
              Move Robot to Position 1
            </h2>
            <p className="text-muted-foreground mb-6">
              Move the arm forward and fully close the gripper. The gripper
              moving claw should be to the left of the arm.
            </p>
            <div>
              <img
                src={position1 || "/placeholder.svg"}
                alt="Calibration Position 1"
                className="rounded-md w-full max-w-md mx-auto"
              />
              <p className="mt-3 text-sm text-muted-foreground">
                Red: X-axis, Green: Y-axis, Blue: Z-axis
              </p>
            </div>
          </div>
        );
      case 3:
        return (
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-4">
              Move Robot to Position 2
            </h2>
            <p className="text-muted-foreground mb-6">
              Twist the arm to the left and fully open the gripper.
            </p>
            <div>
              <img
                src={position2 || "/placeholder.svg"}
                alt="Calibration Position 2"
                className="rounded-md w-full max-w-md mx-auto"
              />
              <p className="mt-3 text-sm text-muted-foreground">
                Red: X-axis, Green: Y-axis, Blue: Z-axis
              </p>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // Loading state
  if (!serverStatus && !serverError) {
    <LoadingPage />;
  }

  return (
    <>
      <Alert variant={"default"} className="mb-6">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>
          Calibration is only required if you're not a phospho.ai customer.
        </AlertTitle>
        <AlertDescription>
          <p>
            Robots you got from{" "}
            <a href="https://robots.phospho.ai" className="underline">
              robots.phospho.ai
            </a>{" "}
            are already calibrated.
          </p>
        </AlertDescription>
      </Alert>

      <div className="mb-8">{renderStepContent(step)}</div>

      {calibrationStatus === "loading" && (
        <Alert className="mb-6 bg-primary/10 border-primary/20">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <AlertTitle>Calibrating your robot...</AlertTitle>
          <AlertDescription>
            This may take a few moments. Please don't move the robot.
          </AlertDescription>
        </Alert>
      )}

      {(calibrationStatus === "success" ||
        calibrationStatus === "in_progress") && (
        <Alert
          variant={"default"}
          className={`mb-6 ${calibrationStatus === "success" ? " border-green-500 border-2" : ""}`}
        >
          {calibrationStatus === "success" ? (
            <CheckCircle className="h-5 w-5 text-green-500" />
          ) : (
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
          )}
          <AlertTitle>
            {calibrationStatus === "success" && "Calibration Complete"}
            {calibrationStatus === "in_progress" && "Calibration In Progress"}
          </AlertTitle>
          <AlertDescription>{message}</AlertDescription>
        </Alert>
      )}

      <div className="flex flex-col gap-4">
        {calibrationStatus === "idle" && (
          <Button onClick={handleNextStep}>Start Calibration</Button>
        )}
        {calibrationStatus === "error" && (
          <Button onClick={handleRestart} variant="secondary">
            <RotateCcw className="mr-2 h-5 w-5" />
            Restart Calibration
          </Button>
        )}
        {calibrationStatus === "loading" && (
          <Button disabled={true}>
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Calibrating...
          </Button>
        )}
        {calibrationStatus === "in_progress" && (
          <Button onClick={handleNextStep}>
            {step < totalSteps ? (
              <>
                Next Step
                <ArrowRight className="ml-2 h-5 w-5" />
              </>
            ) : (
              "Complete Calibration"
            )}
          </Button>
        )}
        {calibrationStatus === "success" && (
          <div className="flex flex-col gap-2 w-full">
            <Link to="/">
              <Button className="w-full">
                <Home className="mr-2 h-5 w-5" />
                Back to Home
              </Button>
            </Link>
            <Button onClick={handleRestart} variant="secondary">
              <RotateCcw className="mr-2 h-5 w-5" />
              Restart Calibration
            </Button>
            <Link to="/control">
              <Button className="w-full" variant="secondary">
                <Keyboard className="mr-2 h-5 w-5" />
                Keyboard Control
              </Button>
            </Link>
          </div>
        )}

        <div className="mb-8">
          <div className="flex items-end justify-between mb-2">
            <span className="text-xs text-muted-foreground">
              Step {step} of {totalSteps}
            </span>
          </div>
          <Progress value={(step / totalSteps) * 100} className="h-2" />
        </div>
      </div>

      <a
        className="text-xs text-muted-foreground underline cursor-pointer"
        href="https://www.youtube.com/watch?v=65DW8yLcRmM"
      >
        Need help? Watch the video tutorial.
      </a>
    </>
  );
}
