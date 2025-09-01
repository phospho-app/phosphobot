"use client";

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
import { useLocalStorageState } from "@/lib/hooks";
import { fetchWithBaseUrl } from "@/lib/utils";
import { Loader2, Video } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { mutate } from "swr";

interface AddZMQCameraModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AddZMQCameraModal({
  open,
  onOpenChange,
}: AddZMQCameraModalProps) {
  // --- STATE ---
  // The form state is now persisted in localStorage.
  const [tcpAddress, setTcpAddress] = useLocalStorageState(
    "add_zmq_tcp_address",
    "tcp://localhost:5555",
  );
  const [topic, setTopic] = useLocalStorageState("add_zmq_topic", "cabin_view");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (isSubmitting) return;

    // --- VALIDATION ---
    // TCP address is required, but topic can be an empty string.
    if (!tcpAddress.trim()) {
      toast.error("Please enter a TCP address");
      return;
    }
    if (!tcpAddress.startsWith("tcp://")) {
      toast.error("TCP address must start with 'tcp://'");
      return;
    }

    setIsSubmitting(true);

    const payload = {
      // We still trim the values before sending to the backend
      tcp_address: tcpAddress.trim(),
      topic: topic.trim(),
    };

    const response = await fetchWithBaseUrl(
      "/cameras/add-zmq",
      "POST",
      payload,
    );

    if (response) {
      toast.success("ZMQ camera has been added successfully.");
      onOpenChange(false); // Close modal on success
      mutate("/status");
      mutate("/admin/settings");
    }

    setIsSubmitting(false);
  };

  const isFormInvalid = !tcpAddress.trim();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Video className="h-5 w-5" />
            Add ZMQ Camera
          </DialogTitle>
          <DialogDescription>
            Enter the TCP address and topic for the ZMQ publisher. Your entries
            will be saved for next time.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit}>
          <div className="grid gap-6 py-4">
            {/* TCP Address Input */}
            <div className="space-y-2">
              <Label htmlFor="tcp-address">TCP Address</Label>
              <Input
                id="tcp-address"
                type="text"
                placeholder="tcp://localhost:5555"
                value={tcpAddress}
                onChange={(e) => setTcpAddress(e.target.value)}
                disabled={isSubmitting}
              />
              <p className="text-sm text-muted-foreground">
                Format: tcp://&lt;host&gt;:&lt;port&gt;
              </p>
            </div>

            {/* Topic Input Field */}
            <div className="space-y-2">
              <Label htmlFor="topic">Topic</Label>
              <Input
                id="topic"
                type="text"
                placeholder="cabin_view"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                disabled={isSubmitting}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isSubmitting && !isFormInvalid) {
                    handleSubmit(e);
                  }
                }}
              />
              <p className="text-sm text-muted-foreground">
                The topic to subscribe to. Leave empty to subscribe to all
                topics (for legacy publishers).
              </p>
            </div>
          </div>
        </form>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || isFormInvalid}
            className="min-w-[120px]"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Adding...
              </>
            ) : (
              "Add Camera"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
