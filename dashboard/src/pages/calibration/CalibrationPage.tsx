import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CalibrationSequence } from "@/pages/calibration/calibration-sequence";
import { JointControl } from "@/pages/calibration/joint-control";

export function CalibrationPage() {
  return (
    <Tabs defaultValue="calibrate">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="calibrate">Calibration</TabsTrigger>
        <TabsTrigger value="joint-control">Joints control</TabsTrigger>
      </TabsList>
      <TabsContent value="calibrate">
        <CalibrationSequence />
      </TabsContent>
      <TabsContent value="joint-control">
        <JointControl />
      </TabsContent>
    </Tabs>
  );
}
