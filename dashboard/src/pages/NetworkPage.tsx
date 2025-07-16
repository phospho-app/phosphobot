import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle, CheckCircle, Loader2, Wifi, WifiOff } from "lucide-react";
import { useState } from "react";

export function NetworkPage() {
  const [message, setMessage] = useState({ text: "", type: "" });
  const [ssid, setSsid] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const showMessage = (text: string, type: string) => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: "", type: "" }), 5000);
  };

  const connectToNetwork = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);
    showMessage(
      "Attempting to connect to network... Please wait and check the LED indicator.",
      "info",
    );
    try {
      const response = await fetch("/network/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ssid, password }),
      });
      if (response.ok) {
        showMessage(
          "Successfully connected! Solid green LED indicates network connection.",
          "success",
        );
        setSsid("");
        setPassword("");
      } else if (response.status === 400) {
        showMessage(
          "This command is intended for use on your control module.",
          "error",
        );
      } else {
        showMessage(
          "Connection failed. Check network details or contact support if LED shows slow blinks.",
          "error",
        );
      }
    } catch (error) {
      showMessage(
        `Network connection error. Restart Raspberry Pi or contact support. ${error}`,
        "error",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const activateHotspot = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);
    showMessage(
      "Activating hotspot... Please wait and check the LED indicator.",
      "info",
    );
    try {
      const response = await fetch("/network/hotspot", { method: "POST" });
      if (response.ok) {
        showMessage(
          'Hotspot activated. Find "phosphobot" in available networks.',
          "success",
        );
      } else if (response.status === 400) {
        showMessage(
          "This command is intended for use on your control module.",
          "error",
        );
      } else {
        showMessage(
          "Hotspot activation failed. Restart Raspberry Pi or contact support.",
          "error",
        );
      }
    } catch (error) {
      showMessage(
        `Hotspot activation error. Restart Raspberry Pi or contact support. ${error}`,
        "error",
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {message.text && (
        <Alert
          variant={
            message.type === "success"
              ? "default"
              : message.type === "error"
                ? "destructive"
                : "default"
          }
          className="mb-6"
        >
          {message.type === "success" && <CheckCircle className="h-4 w-4" />}
          {message.type === "error" && <AlertCircle className="h-4 w-4" />}
          {message.type === "info" && (
            <Loader2 className="h-4 w-4 animate-spin" />
          )}
          <AlertTitle>
            {message.type === "success" && "Success"}
            {message.type === "error" && "Error"}
            {message.type === "info" && "Info"}
          </AlertTitle>
          <AlertDescription>{message.text}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="connect">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="connect">
            <Wifi className="mr-2 h-4 w-4" />
            Connect
          </TabsTrigger>
          <TabsTrigger value="hotspot">
            <WifiOff className="mr-2 h-4 w-4" />
            Hotspot
          </TabsTrigger>
        </TabsList>
        <TabsContent value="connect">
          <Card>
            <CardHeader>
              <CardTitle>Connect to Network</CardTitle>
              <CardDescription>
                Enter the details of the Wi-Fi network you want to connect to.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="space-y-1">
                <Label htmlFor="ssid">Network Name (SSID)</Label>
                <Input
                  id="ssid"
                  placeholder="Enter network name"
                  value={ssid}
                  onChange={(e) => setSsid(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter network password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </CardContent>
            <CardFooter>
              <Button
                onClick={connectToNetwork}
                disabled={isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Wifi className="mr-2 h-4 w-4" />
                    Connect to Network
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        <TabsContent value="hotspot">
          <Card>
            <CardHeader>
              <CardTitle>Activate Hotspot</CardTitle>
              <CardDescription>
                Create a Wi-Fi hotspot for other devices to connect to.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <p>
                  <strong>SSID:</strong> phosphobot
                </p>
                <p>
                  <strong>Password:</strong> phosphobot123
                </p>
              </div>
            </CardContent>
            <CardFooter>
              <Button
                onClick={activateHotspot}
                disabled={isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Activating...
                  </>
                ) : (
                  <>
                    <WifiOff className="mr-2 h-4 w-4" />
                    Activate Hotspot
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>LED Status Guide</CardTitle>
          <CardDescription>
            The LED on your control module indicates the network connection
            status.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="list-disc list-inside space-y-1">
            <li>
              <strong>Solid Green:</strong> Connected to a network
            </li>
            <li>
              <strong>4 Quick Blinks, Pause:</strong> Hotspot active
            </li>
            <li>
              <strong>Slow Blink:</strong> Issue, restart the pi
            </li>
          </ul>
        </CardContent>
      </Card>
    </>
  );
}
