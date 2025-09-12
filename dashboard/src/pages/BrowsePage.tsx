import { LoadingPage } from "@/components/common/loading";
import { Modal } from "@/components/common/modal";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { DatasetInfoResponse, ServerStatus } from "@/types";
import {
  AlertCircle,
  ArrowUpFromLine,
  ChevronRight,
  //Columns3,
  Download,
  ExternalLink,
  Eye,
  File,
  Folder,
  LoaderCircle,
  MoreVertical,
  Plus,
  Repeat,
  Shuffle,
  Split,
  Trash2,
  Wrench,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { redirect, useParams, useSearchParams } from "react-router-dom";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

interface FileItem {
  name: string;
  path: string;
  absolute_path: string;
  is_dir: boolean;
  is_dataset_dir: boolean;
  browseUrl: string;
  huggingfaceUrl?: string;
  downloadUrl?: string;
  previewUrl?: string | null;
  canDeleteDataset?: boolean;
  deleteDatasetAction?: string;
}
interface BrowseData {
  directoryTitle: string;
  tokenError?: string;
  items: FileItem[];
  episode_ids?: number[];
  episode_paths?: string[];
}

interface DatasetInfos {
  [key: string]: DatasetInfoResponse | null;
}
interface MergeDialogProps {
  selectedItems: string[];
  datasetInfos: DatasetInfos;
  setMergeModalOpen: (open: boolean) => void;
  mergeMultipleDatasets: (
    mergedName: string,
    imageKeyMappings: Record<string, string>,
  ) => void;
  loading: boolean;
}

const MergeDialog: React.FC<MergeDialogProps> = ({
  selectedItems,
  datasetInfos,
  setMergeModalOpen,
  mergeMultipleDatasets,
  loading,
}) => {
  const [mergedDatasetName, setMergedDatasetName] = useState("");

  // Create state to track image key mappings
  const [imageKeyMappings, setImageKeyMappings] = useState<
    Record<string, string>
  >({});

  // Get source dataset image keys
  const sourceDatasetImageKeys =
    datasetInfos[selectedItems[0]]?.image_keys || [];
  // Get target dataset image keys
  const targetDatasetImageKeys =
    datasetInfos[selectedItems[1]]?.image_keys || [];

  // Handle image key selection change
  const handleImageKeyChange = (sourceKey: string, targetKey: string) => {
    setImageKeyMappings((prev) => ({
      ...prev,
      [sourceKey]: targetKey,
    }));
  };

  const handleMerge = () => {
    // Pass both the dataset name and the image key mappings
    mergeMultipleDatasets(mergedDatasetName, imageKeyMappings);
    setImageKeyMappings({});
  };

  const allKeysMapped =
    sourceDatasetImageKeys.length > 0 &&
    sourceDatasetImageKeys.every((key) => !!imageKeyMappings[key]);

  const previewImageComponent = (
    imageKey: string,
    datasetInfos: DatasetInfos,
    sourceOrTarget: "Source" | "Target",
  ) => {
    return (
      <div className="border rounded-md p-2 w-32">
        <p className="text-xs mb-1">{sourceOrTarget}</p>
        <div className="h-20 flex items-center justify-center">
          <img
            src={`data:image/jpeg;base64,${datasetInfos[selectedItems[sourceOrTarget == "Source" ? 0 : 1]]?.image_frames?.[imageKey] ?? ""}`}
            alt={`Preview of ${imageKey}`}
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <p className="text-xs mt-1 truncate">{imageKey}</p>
      </div>
    );
  };

  return (
    <DialogContent className="sm:max-w-2xl max-h-[110vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>Merge Datasets</DialogTitle>
      </DialogHeader>

      <div className="grid gap-4 py-4">
        <div className="grid gap-2">
          <Label htmlFor="mergedDatasetName">New Dataset Name</Label>
          <Input
            id="mergedDatasetName"
            value={mergedDatasetName}
            onChange={(e) => {
              const value = e.target.value;
              // Allow only characters that are not whitespace or "/"
              if (/^[^\s/]*$/.test(value)) {
                setMergedDatasetName(value);
              }
            }}
            placeholder="Enter a name for the merged dataset"
            className="w-full"
          />
        </div>

        {sourceDatasetImageKeys.length > 0 && (
          <p className="text-sm text-muted-foreground">
            Map all image keys from dataset <code>{selectedItems[0]}</code> to
            their corresponding keys in dataset <code>{selectedItems[1]}</code>.
            {!allKeysMapped && sourceDatasetImageKeys.length > 0 && (
              <span className="text-accent ml-1">
                All image keys must be mapped before merging.
              </span>
            )}
          </p>
        )}

        <div className="space-y-4">
          {sourceDatasetImageKeys.length > 0 &&
            targetDatasetImageKeys.length > 0 &&
            sourceDatasetImageKeys.map((sourceKey) => (
              <div key={sourceKey} className="grid gap-2">
                <Label className="text-sm">
                  Match <code>{sourceKey}</code> from dataset{" "}
                  <code>{selectedItems[0]}</code>
                </Label>

                <div className="flex gap-3 items-center">
                  {/* Source image preview */}
                  {previewImageComponent(sourceKey, datasetInfos, "Source")}

                  <div className="text-center text-sm">→</div>

                  {/* Target image preview (only shows when selected) */}
                  {imageKeyMappings[sourceKey] ? (
                    <>
                      {previewImageComponent(
                        imageKeyMappings[sourceKey],
                        datasetInfos,
                        "Target",
                      )}
                    </>
                  ) : (
                    <div className="border border-dashed rounded-md p-2 h-32 w-32 flex items-center justify-center ">
                      <p className="text-xs text-muted ">Select target image</p>
                    </div>
                  )}

                  <div className="flex-grow">
                    <Select
                      onValueChange={(value) =>
                        handleImageKeyChange(sourceKey, value)
                      }
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select matching image key" />
                      </SelectTrigger>
                      <SelectContent className="max-h-60 overflow-y-auto">
                        {targetDatasetImageKeys.map((targetKey) => (
                          <SelectItem
                            key={targetKey}
                            value={targetKey}
                            className="py-2"
                          >
                            <div className="flex items-center gap-2">
                              <div className="w-12 h-9 flex items-center justify-center overflow-hidden rounded">
                                <img
                                  src={`data:image/jpeg;base64,${datasetInfos[selectedItems[1]]?.image_frames?.[targetKey] ?? ""}`}
                                  alt={`Preview of ${targetKey}`}
                                  className="max-h-full max-w-full object-contain"
                                />
                              </div>
                              <span className="truncate text-sm">
                                {targetKey}
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={() => setMergeModalOpen(false)}>
          Cancel
        </Button>
        <Button
          onClick={handleMerge}
          disabled={!mergedDatasetName.trim() || !allKeysMapped}
        >
          {loading ? (
            <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
          ) : null}
          {loading ? "Merging..." : "Merge"}
        </Button>
      </DialogFooter>
    </DialogContent>
  );
};

export function BrowsePage() {
  const params = useParams();
  const [searchParams] = useSearchParams();
  const path = params.path || searchParams.get("path") || "";

  const { data, error, mutate } = useSWR<BrowseData>(
    ["/files", path],
    ([url]) => fetcher(url, "POST", { path }),
  );

  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [confirmEpisodeDeleteOpen, setConfirmEpisodeDeleteOpen] =
    useState(false);
  const [selectedEpisode, setSelectedEpisode] = useState("");
  const leaderArmSerialIds = useGlobalStore(
    (state) => state.leaderArmSerialIds,
  );
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [datasetInfos, setDatasetInfos] = useState<
    Record<string, DatasetInfoResponse | null>
  >({});
  const [mergeModalOpen, setMergeModalOpen] = useState(false);
  const [openDownloadModal, setOpenDownloadModal] = useState(false);
  const [hfDatasetName, setHFDatasetName] = useState("");
  const [confirmRepairOpen, setConfirmRepairOpen] = useState(false);

  // Shuffle modal
  const [openShuffleModal, setOpenShuffleModal] = useState(false);

  // Split modal
  const [openSplitModel, setOpenSplitModel] = useState(false);
  const [selectedSplitRatio, setSelectedSplitRatio] = useState(0.8);
  const [firstSplitName, setFirstSplitName] = useState("");
  const [secondSplitName, setSecondSplitName] = useState("");

  // Loading state for episode deletion
  const [loading, setLoading] = useState(false);

  const { data: serverStatus } = useSWR<ServerStatus>(
    ["/status"],
    ([url]) => fetcher(url),
    {
      refreshInterval: 5000,
    },
  );

  const fetchDatasetInfo = async (
    path: string,
  ): Promise<DatasetInfoResponse | null> => {
    const response = await fetchWithBaseUrl(
      `/dataset/info?path=${encodeURIComponent(path)}`,
      "POST",
    );
    if (response.status !== "ok") {
      return null;
    }
    return response as DatasetInfoResponse;
  };

  useEffect(() => {
    const fetchInfos = async () => {
      if (!data || !data.items) return;

      const infos: Record<string, DatasetInfoResponse | null> = {};
      await Promise.all(
        data.items
          .filter((item) => item.is_dataset_dir)
          .map(async (item) => {
            const info = await fetchDatasetInfo(item.path);
            infos[item.path] = info;
          }),
      );
      setDatasetInfos(infos);
    };

    fetchInfos();
  }, [data]);

  const isRobotConnected = useMemo(() => {
    return serverStatus?.robots && serverStatus.robots.length > 0;
  }, [serverStatus]);

  if (error)
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>Failed to load data.</AlertDescription>
      </Alert>
    );

  if (!data) return <LoadingPage />;

  const handleSelectItem = (path: string) => {
    setSelectedItems((prev) => {
      if (prev.includes(path)) {
        return prev.filter((item) => item !== path);
      } else {
        return [...prev, path];
      }
    });
  };

  const handleDeleteMultipleDatasets = async () => {
    if (selectedItems.length === 0) {
      toast.error("No datasets selected for deletion");
      return;
    }
    setLoading(true);
    const deletePromises = selectedItems.map(async (item) => {
      const resp = await fetchWithBaseUrl(
        `/dataset/delete?path=${encodeURIComponent(item)}`,
        "POST",
      );
      if (resp.status !== "ok") {
        toast.error(`Failed to delete dataset: ${item}`);
        return;
      }
      toast.success(`Dataset deleted successfully: ${item}`);
    });
    await Promise.all(deletePromises);
    setLoading(false);
    setConfirmDeleteOpen(false);
    setSelectedItems([]);
    mutate();
  };

  const handleMergeCheck = async () => {
    if (selectedItems.length !== 2) {
      toast.error("Please select exactly 2 datasets to merge");
      return;
    }

    if (
      datasetInfos[selectedItems[0]]?.robot_type !==
      datasetInfos[selectedItems[1]]?.robot_type
    ) {
      toast.error("Datasets have different robot types. Cannot merge.");
      return;
    }

    if (
      datasetInfos[selectedItems[0]]?.robot_dof !==
      datasetInfos[selectedItems[1]]?.robot_dof
    ) {
      toast.error("Datasets have different DOF. Cannot merge.");
      return;
    }

    if (
      datasetInfos[selectedItems[0]]?.image_keys?.length !==
      datasetInfos[selectedItems[1]]?.image_keys?.length
    ) {
      toast.error(
        "Datasets have different number of image keys. Cannot merge.",
      );
      return;
    }

    setMergeModalOpen(true);
  };

  const handleSplitCheck = async () => {
    if (selectedItems.length !== 1) {
      toast.error("Please select exactly 1 dataset to split");
      return;
    }

    setOpenSplitModel(true);
  };

  // const handleColumnCheck = async () => {
  //   if (selectedItems.length !== 1) {
  //     toast.error("Please select exactly 1 dataset to check columns");
  //     return;
  //   }
  //   setOpenDeleteModal(true);
  // };

  const mergeMultipleDatasets = async (
    newDatasetName: string,
    imageKeyMappings?: Record<string, string>,
  ) => {
    setLoading(true);
    console.log("Merging datasets:", selectedItems);

    const data = await fetchWithBaseUrl(`/dataset/merge`, "POST", {
      first_dataset: selectedItems[0],
      second_dataset: selectedItems[1],
      new_dataset_name: newDatasetName,
      image_key_mappings: imageKeyMappings,
    });

    if (data) {
      toast.success("Episode merged successfully");
      mutate();
      redirect(path);
      console.log("Merged datasets:", selectedItems);
    }

    setLoading(false);
    setMergeModalOpen(false);
  };

  const handleRepairDataset = async () => {
    if (selectedItems.length === 0) {
      toast.error("No datasets selected for repair");
      return;
    }
    setLoading(true);
    const repairPromises = selectedItems.map(async (item) => {
      console.log("Repairing dataset:", item);
      const resp = await fetchWithBaseUrl(`/dataset/repair`, "POST", {
        dataset_path: item,
      });
      if (resp.status !== "ok") {
        toast.error(`Failed to repair dataset: ${item}`);
        return;
      }
      toast.success(`Dataset repaired successfully: ${item}`);
    });
    await Promise.all(repairPromises);
    setLoading(false);
    setConfirmRepairOpen(false);
    setSelectedItems([]);
    mutate();
  };

  const handleDeleteEpisode = async () => {
    // Ensure an episode is selected.
    if (!selectedEpisode) return;
    fetchWithBaseUrl(`/episode/delete`, "POST", {
      path: path,
      episode_id: selectedEpisode,
    }).then((response) => {
      if (response.status !== "ok") {
        toast.error("Failed to delete episode");
      } else {
        toast.success("Episode deleted successfully");
        mutate();
        redirect(path);
      }
    });
  };

  const handleReplayEpisode = async () => {
    if (!selectedEpisode || !data.episode_paths) return;
    const episode_path = data.episode_paths[parseInt(selectedEpisode)];
    if (!episode_path) return;
    const robot_serials_to_ignore = leaderArmSerialIds ?? null;
    fetchWithBaseUrl(`/recording/play`, "POST", {
      episode_path,
      robot_serials_to_ignore,
    });
  };

  const handlePushToHub = async (item: FileItem) => {
    fetchWithBaseUrl(`/dataset/sync?path=${item.path}`, "POST");
  };

  const pathParts = path.split("/").filter(Boolean);

  return (
    <div className="container mx-auto py-6 px-4 md:px-6">
      <div className="flex items-center justify-between mb-4">
        <Breadcrumb className="mb-4">
          <BreadcrumbList>
            <BreadcrumbItem>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <BreadcrumbLink asChild>
                      <Link to="/browse">phosphobot</Link>
                    </BreadcrumbLink>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>
                      Explore your datasets in{" "}
                      <code>~/phosphobot/recordings</code>
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </BreadcrumbItem>
            <BreadcrumbSeparator>
              <ChevronRight className="h-4 w-4" />
            </BreadcrumbSeparator>
            {pathParts.map((part, index) => (
              <BreadcrumbItem key={index}>
                <BreadcrumbLink asChild>
                  <Link
                    to={`/browse?path=${pathParts.slice(0, index + 1).join("/")}`}
                  >
                    {part}
                  </Link>
                </BreadcrumbLink>
                {index < pathParts.length - 1 && (
                  <BreadcrumbSeparator>
                    <ChevronRight className="h-4 w-4" />
                  </BreadcrumbSeparator>
                )}
              </BreadcrumbItem>
            ))}
          </BreadcrumbList>
        </Breadcrumb>

        <Button variant="outline" onClick={() => setOpenDownloadModal(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Add dataset from Hub 🤗
        </Button>
      </div>
      {data.tokenError && (
        <Alert variant="destructive" className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{data.tokenError}</AlertDescription>
        </Alert>
      )}
      <Table className="bg-background rounded-lg">
        <TableHeader>
          <TableRow>
            <TableCell className="w-[50px]" />
            <TableCell>Name</TableCell>
            {path.endsWith("lerobot_v2") || path.endsWith("lerobot_v2.1") ? (
              <>
                <TableCell className="text-muted-foreground">
                  Robot Type
                </TableCell>
                <TableCell className="text-muted-foreground">DOF</TableCell>
                <TableCell className="text-muted-foreground">
                  Episodes
                </TableCell>
                <TableCell className="text-muted-foreground">
                  Image Keys
                </TableCell>
              </>
            ) : null}
            <TableCell></TableCell>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.items.map((item) => (
            <TableRow key={item.path}>
              {/* Show checkbox only if path ends with 'lerobot_v2.1' or lerobot_v2 */}
              <TableCell className="w-[50px]">
                {item.is_dataset_dir ? (
                  <Checkbox
                    checked={selectedItems.includes(item.path)}
                    onCheckedChange={() => handleSelectItem(item.path)}
                    aria-label={`Select ${item.name}`}
                    key={item.path}
                  />
                ) : null}
              </TableCell>
              <TableCell>
                {item.is_dir ? (
                  <Link
                    to={item.browseUrl}
                    className="flex items-center text-blue-500 hover:underline"
                  >
                    <Folder className="mr-2 h-4 w-4" />
                    {item.name}
                  </Link>
                ) : (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="flex items-center">
                          <File className="mr-2 h-4 w-4" />
                          {item.name}
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        There is no preview available for this file. Open it in
                        your file explorer to view it.
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {datasetInfos[item.path]?.robot_type &&
                  datasetInfos[item.path]?.robot_type}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {datasetInfos[item.path]?.robot_dof &&
                  datasetInfos[item.path]?.robot_dof}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {datasetInfos[item.path]?.number_of_episodes &&
                  datasetInfos[item.path]?.number_of_episodes}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {datasetInfos[item.path]?.image_keys &&
                  datasetInfos[item.path]?.image_keys?.length}
              </TableCell>
              <TableCell>
                <div className="flex space-x-2 justify-end">
                  {(item.downloadUrl || item.canDeleteDataset) && (
                    <DropdownMenu>
                      {item.previewUrl && (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <a
                                href={item.previewUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex"
                              >
                                <Button variant={"outline"} size="sm">
                                  <Eye className="mr-2 h-4 w-4" />
                                  Preview
                                </Button>
                              </a>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>
                                Please use HVC1 or AVC1 codec for videos to be
                                visible in LeRobot dataset viewer.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}

                      <DropdownMenuTrigger asChild className="cursor-pointer">
                        <Button variant="outline" size="sm">
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {item.huggingfaceUrl && (
                          <DropdownMenuItem asChild>
                            <a
                              href={item.huggingfaceUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center cursor-pointer"
                            >
                              <ExternalLink className="mr-2 h-4 w-4" />
                              See on Hugging Face Hub
                            </a>
                          </DropdownMenuItem>
                        )}
                        {item.downloadUrl && item.is_dataset_dir && (
                          <DropdownMenuItem
                            onClick={() => handlePushToHub(item)}
                            className="cursor-pointer"
                          >
                            <ArrowUpFromLine className="mr-2 h-4 w-4" />
                            Push to Hugging Face Hub
                          </DropdownMenuItem>
                        )}
                        {item.downloadUrl && (
                          <DropdownMenuItem asChild>
                            <a
                              href={item.downloadUrl}
                              className="flex items-center cursor-pointer"
                            >
                              <Download className="mr-2 h-4 w-4" />
                              Download
                            </a>
                          </DropdownMenuItem>
                        )}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  )}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {data.items.length > 0 && data.items[0].previewUrl && (
        <Alert variant="default" className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Preview Feature</AlertTitle>
          <AlertDescription>
            To use the preview feature, please upload your dataset to Hugging
            Face first.
          </AlertDescription>
        </Alert>
      )}
      {/* Show episode selection and replay/delete buttons */}
      {data.episode_ids && data.episode_ids.length > 0 && (
        <div className="mt-6">
          <div className="flex items-end gap-2">
            <Select
              value={selectedEpisode}
              name="episode_id"
              onValueChange={setSelectedEpisode}
            >
              <SelectTrigger id="episode-select">
                <SelectValue placeholder="Select Episode ID" />
              </SelectTrigger>
              <SelectContent>
                {data.episode_ids.map((episode) => (
                  <SelectItem key={episode} value={episode.toString()}>
                    Episode {episode}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              type="button"
              variant="destructive"
              onClick={() => {
                if (!selectedEpisode) return;
                setConfirmEpisodeDeleteOpen(true);
              }}
            >
              <Trash2 className="mr-2 h-4 w-3" />
              Delete Episode
            </Button>
            {isRobotConnected && (
              <Button
                type="button"
                variant="default"
                onClick={async () => {
                  await handleReplayEpisode();
                }}
              >
                <Repeat className="mr-2 h-4 w-3" />
                Replay Episode
              </Button>
            )}
          </div>
        </div>
      )}
      {selectedItems.length > 0 &&
        (path.endsWith("lerobot_v2") || path.endsWith("lerobot_v2.1")) && (
          <div className="flex flex-col md:flex-row md:space-x-2 mt-6">
            {path.endsWith("lerobot_v2.1") && (
              <>
                <Button
                  className="mb-4"
                  variant="outline"
                  onClick={() => handleMergeCheck()}
                >
                  <Repeat className="mr-2 h-4 w-3" />
                  Merge Selected Datasets
                </Button>
                <Button
                  className="mb-4"
                  variant="outline"
                  onClick={() => handleSplitCheck()}
                >
                  <Split className="mr-2 h-4 w-3" />
                  Split Selected Datasets
                </Button>
                {/* <Button
                  className="mb-4"
                  variant="outline"
                  onClick={() => handleColumnCheck()}
                >
                  <Columns3 className="mr-2 h-4 w-3" />
                  Check Selected Columns
                </Button> */}
                <Button
                  className="mb-4"
                  variant="outline"
                  onClick={() => setOpenShuffleModal(true)}
                >
                  <Shuffle className="mr-2 h-4 w-3" />
                  Shuffle Selected Datasets
                </Button>
              </>
            )}
            <Button
              className="mb-4"
              variant="outline"
              onClick={() => setConfirmRepairOpen(true)}
            >
              <Wrench className="mr-2 h-4 w-3" />
              Repair Selected Datasets
            </Button>
            <Button
              className="mb-4"
              variant="destructive"
              onClick={() => setConfirmDeleteOpen(true)}
            >
              <Trash2 className="mr-2 h-4 w-3" />
              Delete Selected Datasets
            </Button>
          </div>
        )}
      {/* Dataset deletion dialog */}
      <Modal
        open={confirmDeleteOpen}
        onOpenChange={setConfirmDeleteOpen}
        title="Confirm Delete Dataset"
        description={
          <>
            Are you sure you want to delete the selected datasets:
            <br />
            {selectedItems.length > 0
              ? selectedItems.map((item) => (
                  <span key={item}>
                    <strong>{item}</strong>
                    <br />
                  </span>
                ))
              : null}
            This action cannot be undone.
          </>
        }
        confirmLabel={loading ? "Deleting..." : "Delete"}
        confirmVariant="destructive"
        isLoading={loading}
        onConfirm={handleDeleteMultipleDatasets}
      />
      {/* Episode deletion dialog */}
      <Modal
        open={confirmEpisodeDeleteOpen}
        onOpenChange={setConfirmEpisodeDeleteOpen}
        title="Confirm Delete Episode"
        description={
          <>
            Are you sure you want to delete episode{" "}
            <strong>{selectedEpisode}</strong>? This action cannot be undone.
          </>
        }
        confirmLabel={loading ? "Deleting..." : "Delete"}
        confirmVariant="destructive"
        isLoading={loading}
        onConfirm={async () => {
          setLoading(true);
          await handleDeleteEpisode();
          setConfirmEpisodeDeleteOpen(false);
          setLoading(false);
          redirect(path);
        }}
      />
      {/* Split modal */}
      <Modal
        open={openSplitModel}
        onOpenChange={setOpenSplitModel}
        title="Split Dataset"
        description="This will split the selected dataset into separate datasets."
        confirmLabel={loading ? "Splitting..." : "Split"}
        onConfirm={async () => {
          if (selectedItems.length !== 1) {
            toast.error("Please select exactly 1 dataset to split");
            return;
          }
          setLoading(true);
          const resp = await fetchWithBaseUrl(`/dataset/split`, "POST", {
            dataset_path: selectedItems[0],
            split_ratio: selectedSplitRatio,
            first_split_name: firstSplitName,
            second_split_name: secondSplitName,
          });
          if (resp.status !== "ok") {
            toast.error("Failed to split dataset: " + resp.message);
          }
          setLoading(false);
          setOpenSplitModel(false);
          mutate();
          redirect(path);
        }}
        isLoading={loading}
      >
        <div className="grid gap-4 py-4">
          <Label htmlFor="split-ratio">Split Ratio</Label>
          <p className="text-sm text-muted-foreground">
            The ratio of the first split to the total dataset. For example, a
            value of 0.8 means 80% of the data will go to the first split and
            20% to the second.
          </p>
          <Input
            id="split-ratio"
            type="number"
            value={selectedSplitRatio}
            onChange={(e) => {
              const value = parseFloat(e.target.value);
              if (!isNaN(value) && value >= 0 && value <= 1) {
                setSelectedSplitRatio(value);
              }
            }}
            step="0.01"
            min="0"
            max="1"
            placeholder="Enter the split ratio (0-1)"
            className="w-full"
          />
          <Label htmlFor="first-split-name">First Split Name</Label>
          <Input
            id="first-split-name"
            value={firstSplitName}
            onChange={(e) => {
              const value = e.target.value;
              // Allow only characters that are not whitespace
              if (/^[^\s]*$/.test(value)) {
                setFirstSplitName(value);
              }
            }}
            placeholder="Enter the name for the first split"
            className="w-full"
          />
          <Label htmlFor="second-split-name">Second Split Name</Label>
          <Input
            id="second-split-name"
            value={secondSplitName}
            onChange={(e) => {
              const value = e.target.value;
              // Allow only characters that are not whitespace
              if (/^[^\s]*$/.test(value)) {
                setSecondSplitName(value);
              }
            }}
            placeholder="Enter the name for the second split"
            className="w-full"
          />
        </div>
      </Modal>
      {/* Download modal */}
      <Modal
        open={openDownloadModal}
        onOpenChange={setOpenDownloadModal}
        title="Download dataset from the Hugging Face Hub 🤗"
        description="The dataset should be available as public on the Hugging Face Hub."
        confirmLabel={loading ? "Downloading..." : "Download"}
        isLoading={loading}
        onConfirm={async () => {
          if (hfDatasetName.trim() === "") {
            toast.error("No dataset selected for download");
            return;
          }
          setLoading(true);
          const resp = await fetchWithBaseUrl(`/dataset/hf_download`, "POST", {
            dataset_name: hfDatasetName,
          });
          if (resp.status !== "ok") {
            toast.error("Failed to download dataset: " + resp.message);
          } else {
            toast.success("Dataset downloaded successfully");
          }
          setLoading(false);
          setOpenDownloadModal(false);
          mutate();
          redirect(path);
        }}
      >
        <div className="grid gap-4 py-4">
          <Label htmlFor="dataset-select">Select Dataset</Label>
          <Input
            id="dataset-select"
            value={hfDatasetName}
            onChange={(e) => {
              const value = e.target.value;
              // Allow only characters that are not whitespace
              if (/^[^\s]*$/.test(value)) {
                setHFDatasetName(value);
              }
            }}
            placeholder="hf_username/dataset_name"
            className="w-full"
          />
        </div>
      </Modal>
      {/* Dataset repair dialog */}
      <Modal
        open={confirmRepairOpen}
        onOpenChange={setConfirmRepairOpen}
        title="Repair Dataset"
        description={
          <>
            This will attempt to repair the selected datasets:
            <br />
            {selectedItems.length > 0
              ? selectedItems.map((item) => (
                  <span key={item}>
                    <strong>{item}</strong>
                    <br />
                  </span>
                ))
              : null}
            For now, this will only recalculate the parquets files, not the meta
            data.
          </>
        }
        confirmLabel={loading ? "Repairing..." : "Repair"}
        isLoading={loading}
        onConfirm={handleRepairDataset}
      />
      {/* Dataset shuffle modal */}
      <Modal
        open={openShuffleModal}
        onOpenChange={setOpenShuffleModal}
        title="Shuffle Datasets"
        description="This will shuffle the selected datasets in place (no copy)."
        confirmLabel={loading ? "Shuffling..." : "Shuffle"}
        isLoading={loading}
        onConfirm={async () => {
          if (selectedItems.length === 0) {
            toast.error("No datasets selected for shuffling");
            return;
          }
          setLoading(true);
          // iterate over selected items and shuffle them
          Promise.all(
            selectedItems.map(async (item) => {
              const resp = await fetchWithBaseUrl(`/dataset/shuffle`, "POST", {
                dataset_path: item,
              });
              if (resp.status !== "ok") {
                toast.error(`Failed to shuffle dataset: ${item}`);
              } else {
                toast.success(`Dataset shuffled successfully: ${item}`);
              }
            }),
          );
          setLoading(false);
          setOpenShuffleModal(false);
          mutate();
          redirect(path);
        }}
      >
        <div className="grid gap-4 py-4">
          <p className="text-sm text-muted-foreground">
            This will shuffle the selected datasets randomly.
            <br />
            {selectedItems.length > 0
              ? selectedItems.map((item) => (
                  <span key={item}>
                    <strong>{item}</strong>
                    <br />
                  </span>
                ))
              : null}
          </p>
        </div>
      </Modal>
      {/* Merge modal */}
      <Dialog open={mergeModalOpen} onOpenChange={setMergeModalOpen}>
        <MergeDialog
          selectedItems={selectedItems}
          datasetInfos={datasetInfos}
          setMergeModalOpen={setMergeModalOpen}
          mergeMultipleDatasets={mergeMultipleDatasets}
          loading={loading}
        />
      </Dialog>
    </div>
  );
}
