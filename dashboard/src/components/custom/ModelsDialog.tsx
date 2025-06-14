"use client";

import { CopyButton } from "@/components/common/copy-button";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { SupabaseTrainingModel, TrainingConfig } from "@/types";
import { Ban, Check, ExternalLink, Loader2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type React from "react";
import { toast } from "sonner";
import useSWR from "swr";

type ModelStatus = "succeeded" | "failed" | "running" | "canceled" | null;

interface ModelStatusFilterProps {
  onStatusChange: (status: ModelStatus) => void;
  className?: string;
}

export function ModelStatusFilter({
  onStatusChange,
  className,
}: ModelStatusFilterProps) {
  const [selectedStatus, setSelectedStatus] = useState<ModelStatus>(null);

  const handleStatusChange = (value: string) => {
    // If clicking the already selected status, clear the selection
    const newStatus = value === selectedStatus ? null : (value as ModelStatus);
    setSelectedStatus(newStatus);
    onStatusChange(newStatus);
  };

  return (
    <div className={cn("gap-y-2", className)}>
      <ToggleGroup
        type="single"
        value={selectedStatus || ""}
        onValueChange={handleStatusChange}
      >
        <ToggleGroupItem
          value="succeeded"
          aria-label="Filter by succeeded status"
          className={cn(
            "text-xs flex items-center gap-1.5",
            selectedStatus === "succeeded" &&
              "text-green-600 dark:text-green-500",
          )}
        >
          <Check className="h-4 w-4 text-green-600 dark:text-green-500" />
          <span>Succeeded</span>
        </ToggleGroupItem>

        <ToggleGroupItem
          value="failed"
          aria-label="Filter by failed status"
          className={cn(
            "text-xs flex items-center gap-1.5",
            selectedStatus === "failed" && "text-red-600 dark:text-red-500",
          )}
        >
          <X className="h-4 w-4 text-red-600 dark:text-red-500" />
          <span>Failed</span>
        </ToggleGroupItem>

        <ToggleGroupItem
          value="running"
          aria-label="Filter by running status"
          className={cn(
            "text-xs flex items-center gap-1.5",
            selectedStatus === "running" && "text-blue-600 dark:text-blue-500",
          )}
        >
          <Loader2 className="h-4 w-4 text-blue-600 dark:text-blue-500" />
          <span>Running</span>
        </ToggleGroupItem>

        <ToggleGroupItem
          value="canceled"
          aria-label="Filter by canceled status"
          className={cn(
            "flex items-center gap-1.5",
            selectedStatus === "canceled" && "text-gray-600 dark:text-gray-400",
          )}
        >
          <Ban className="h-4 w-4 text-gray-600 dark:text-gray-400" />
          <span>Canceled</span>
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  );
}

interface ModelsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const ValueWithTooltip = ({ value }: { value: string }) => {
  // Don't use tooltip if value is short
  if (!value || value.length < 30) {
    return <span>{value}</span>;
  }

  // Show a preview of the first 20 characters after the last "/"
  const preview = value.split("/").pop()?.slice(0, 25) + "..." || value;

  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <span className="border-b border-dotted border-muted-foreground">
            {preview}
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-md break-words">
          {value}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

// ModelRow component
const ModelRow: React.FC<{ model: SupabaseTrainingModel }> = ({ model }) => {
  const [isCanceling, setIsCanceling] = useState(false);

  // Set uppercase status for display
  const status = model.status.charAt(0).toUpperCase() + model.status.slice(1);

  // Define model url
  const url = "https://huggingface.co/" + model.model_name;

  const handleCancel = async () => {
    if (
      !confirm(
        "Are you sure you want to cancel this training? This action cannot be undone.",
      )
    ) {
      return;
    }

    setIsCanceling(true);
    const status_response = await fetchWithBaseUrl("/training/cancel", "POST", {
      training_id: model.id,
    });

    if (status_response?.status === "ok") {
      toast.success(status_response?.message);
    }
  };

  return (
    <>
      <TableRow>
        <TableCell>
          {status === "Succeeded" && (
            <Check className="h-4 w-4 inline mr-1 text-green-500" />
          )}
          {status === "Running" && (
            <Loader2 className="h-4 w-4 animate-spin inline mr-1" />
          )}
          {status === "Failed" && (
            <X className="h-4 w-4 inline mr-1 text-red-500" />
          )}
          {status === "Canceled" && (
            <Ban className="h-4 w-4 inline mr-1 text-gray-500" />
          )}
          {status}
        </TableCell>
        <TableCell>
          <div className="flex items-center flex-row justify-between">
            {ValueWithTooltip({ value: model.model_name })}
            <div className="flex items-center">
              <CopyButton text={model.model_name} hint={"Copy model name"} />
              {/* Button to open model page */}
              <Button
                onClick={() => window.open(url, "_blank")}
                title="Go to model"
                aria-label="Go to model"
                className="text-blue-500 hover:bg-blue-50 cursor-pointer"
                variant="ghost"
                size="icon"
              >
                <ExternalLink className="h-4 w-4" />
              </Button>
              {/* Cancel button - only show for running models */}
              {model.status === "running" && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        onClick={handleCancel}
                        disabled={isCanceling}
                        variant="ghost"
                        size="icon"
                        className="text-orange-600 hover:bg-orange-50 hover:text-orange-700 dark:text-orange-400 dark:hover:bg-orange-950"
                      >
                        {isCanceling ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Ban className="h-4 w-4" />
                        )}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      {isCanceling ? "Canceling..." : "Cancel training"}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          </div>
        </TableCell>
        <TableCell>{model.model_type}</TableCell>
        <TableCell>
          <div className="flex flex-col">
            {model.training_params &&
              Object.entries(model.training_params).map(
                ([key, value]) =>
                  value !== null && (
                    <div key={key}>
                      <strong>{key}:</strong> {String(value)}
                    </div>
                  ),
              )}
          </div>
        </TableCell>
        <TableCell>
          {ValueWithTooltip({ value: model.dataset_name })}
          <CopyButton text={model.dataset_name} hint={"Copy dataset name"} />
        </TableCell>
        <TableCell>
          {model.used_wandb ? (
            <Check className="h-4 w-4 inline mr-1 text-green-500" />
          ) : (
            <X className="h-4 w-4 inline mr-1 text-red-500" />
          )}
          {model.used_wandb ? "Yes" : "No"}
        </TableCell>
        <TableCell>{new Date(model.requested_at).toLocaleString()}</TableCell>
      </TableRow>
    </>
  );
};

export const ModelsDialog: React.FC<ModelsDialogProps> = ({
  open,
  onOpenChange,
}) => {
  const {
    data: modelsData,
    isLoading,
    error,
  } = useSWR<TrainingConfig>(
    ["/training/models/read"],
    ([endpoint]) => fetcher(endpoint, "POST"),
    {
      refreshInterval: 5000,
    },
  );

  const models = modelsData?.models || [];
  const [statusFilter, setStatusFilter] = useState<
    "succeeded" | "failed" | "running" | "canceled" | null
  >(null);

  // Filter models based on status
  const filteredModels = useMemo(() => {
    if (!statusFilter) return models;
    return models.filter((model) => model.status === statusFilter);
  }, [models, statusFilter]);

  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  const totalPages = Math.ceil(filteredModels.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentModels = filteredModels.slice(startIndex, endIndex);

  // Reset to first page when models or filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filteredModels.length]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[1200px] w-full">
        <DialogHeader>
          <DialogTitle>Trained Models</DialogTitle>
          <DialogDescription>
            You can only run one training job at a time.
            <br />
            If you get a "Failed" status, please check the error log on the
            Hugging Face model page.
            <br />
            For more advanced options, such as changing the number of epochs,
            steps, etc..., please use the <code>/training/start</code> api
            endpoint.
          </DialogDescription>
        </DialogHeader>
        <ModelStatusFilter
          onStatusChange={setStatusFilter}
          className="border-b pb-4"
        />

        {isLoading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center py-4">
            <p className="text-red-500">{error}</p>
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="text-center py-4">
            {statusFilter
              ? `No ${statusFilter} models found.`
              : "No models found."}
          </div>
        ) : (
          <div className="max-h-[50vh] overflow-auto border rounded-md">
            <Table>
              <TableHeader className="sticky top-0 bg-white z-10">
                <TableRow>
                  <TableHead>Status</TableHead>
                  <TableHead>Model Name</TableHead>
                  <TableHead>Model Type</TableHead>
                  <TableHead>Training Parameters</TableHead>
                  <TableHead>Dataset Name</TableHead>
                  <TableHead>Wandb</TableHead>
                  <TableHead>Created at</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {currentModels.map((model, index) => (
                  <ModelRow key={index} model={model} />
                ))}
              </TableBody>
            </Table>
          </div>
        )}
        {filteredModels.length > itemsPerPage && (
          <div className="flex justify-center mt-4">
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    className={
                      currentPage === 1
                        ? "pointer-events-none opacity-50"
                        : "cursor-pointer"
                    }
                  />
                </PaginationItem>
                {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                  (page) => (
                    <PaginationItem key={page}>
                      <PaginationLink
                        onClick={() => setCurrentPage(page)}
                        isActive={currentPage === page}
                        className="cursor-pointer"
                      >
                        {page}
                      </PaginationLink>
                    </PaginationItem>
                  ),
                )}
                <PaginationItem>
                  <PaginationNext
                    onClick={() =>
                      setCurrentPage(Math.min(totalPages, currentPage + 1))
                    }
                    className={
                      currentPage === totalPages
                        ? "pointer-events-none opacity-50"
                        : "cursor-pointer"
                    }
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export const ModelsCard: React.FC = () => {
  const {
    data: modelsData,
    isLoading,
    error,
  } = useSWR<TrainingConfig>(
    ["/training/models/read"],
    ([endpoint]) => fetcher(endpoint, "POST"),
    {
      refreshInterval: 5000,
    },
  );

  const models = modelsData?.models || [];
  const [statusFilter, setStatusFilter] = useState<
    "succeeded" | "failed" | "running" | "canceled" | null
  >(null);

  // Filter models based on status
  const filteredModels = useMemo(() => {
    if (!statusFilter) return models;
    return models.filter((model) => model.status === statusFilter);
  }, [models, statusFilter]);

  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;
  const totalPages = Math.ceil(filteredModels.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentModels = filteredModels.slice(startIndex, endIndex);

  // Reset to first page when models or filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filteredModels.length]);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Trained Models</CardTitle>
        <CardDescription>
          You can only run one training job at a time.
          <br />
          If you get a "Failed" status, please check the error log on the
          Hugging Face model page.
          <br />
          For more advanced options, such as changing the number of epochs,
          steps, etc..., please use the <code>/training/start</code> api
          endpoint.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-y-4">
        <div className="flex justify-between items-center w-full">
          {filteredModels.length > itemsPerPage && (
            <Pagination className="flex justify-start gap-x-2">
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    className={
                      currentPage === 1
                        ? "text-xs pointer-events-none opacity-50"
                        : "text-xs cursor-pointer"
                    }
                  />
                </PaginationItem>
                {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                  (page) => (
                    <PaginationItem key={page}>
                      <PaginationLink
                        onClick={() => setCurrentPage(page)}
                        isActive={currentPage === page}
                        className="text-xs cursor-pointer"
                      >
                        {page}
                      </PaginationLink>
                    </PaginationItem>
                  ),
                )}
                <PaginationItem>
                  <PaginationNext
                    onClick={() =>
                      setCurrentPage(Math.min(totalPages, currentPage + 1))
                    }
                    className={
                      currentPage === totalPages
                        ? "text-xs pointer-events-none opacity-50"
                        : "text-xs cursor-pointer"
                    }
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          )}
          <div></div>
          <ModelStatusFilter onStatusChange={setStatusFilter} />
        </div>
        {isLoading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center py-4">
            <p className="text-red-500">{error.toString()}</p>
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="text-center py-4">
            {statusFilter
              ? `No ${statusFilter} models found.`
              : "No models found."}
          </div>
        ) : (
          <div className="max-h-[50vh] overflow-auto border rounded-md">
            <Table>
              <TableHeader className="sticky top-0 bg-white z-10">
                <TableRow>
                  <TableHead>Status</TableHead>
                  <TableHead>Model Name</TableHead>
                  <TableHead>Model Type</TableHead>
                  <TableHead>Training Parameters</TableHead>
                  <TableHead>Dataset Name</TableHead>
                  <TableHead>Wandb</TableHead>
                  <TableHead>Created at</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {currentModels.map((model, index) => (
                  <ModelRow key={index} model={model} />
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
