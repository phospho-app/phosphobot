import { MobileMenu } from "@/components/common/mobile-menu";
import { RobotStatusDropdown } from "@/components/common/robot-status-button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuth } from "@/context/AuthContext";
import { fetcher } from "@/lib/utils";
import { ServerStatus } from "@/types";
import {
  BookText,
  BrainCircuit,
  LogOut,
  Mail,
  TestTubeDiagonal,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import useSWR from "swr";

const routeMap = [
  { path: "/", title: "Dashboard" },
  { path: "/control", title: "Robot Control" },
  { path: "/calibration", title: "Calibration" },
  { path: "/inference", title: "AI Control" },
  { path: "/admin", title: "Admin Configuration" },
  { path: "/docs", title: "API Documentation" },
  { path: "/viz", title: "Camera Overview" },
  { path: "/network", title: "Network Management" },
  {
    path: "/browse",
    title: "Browse Datasets",
    isPrefix: true,
  },
];

function ServerIP() {
  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], ([url]) =>
    fetcher(url),
  );

  if (!serverStatus) {
    return null;
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="text-xs text-muted-foreground cursor-pointer">
            {serverStatus.server_ip}:{serverStatus.server_port}
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <p className="max-w-xs">{serverStatus.name}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function RecordingStatus() {
  // Pulsating recording red circle when recording is active
  // Nothing when recording is inactive
  // Tooltip with recording status when hovered
  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], ([url]) =>
    fetcher(url),
  );

  if (!serverStatus) {
    return null;
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="text-xs text-muted-foreground cursor-pointer">
            {serverStatus.is_recording ? (
              <span className="relative inline-block">
                <span className="animate-ping absolute inline-flex size-3 rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full size-3 bg-red-500"></span>
              </span>
            ) : (
              <></>
            )}
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <p className="max-w-xs">
            {serverStatus.is_recording ? "Recording" : "Not Recording"}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function AIControlStatus() {
  // AI Control status badge
  // Display a green pulsating BrainCircuit icon when AI is running. When you click on it, it takes you to the /inference page
  // Tooltip with AI status when hovered
  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], ([url]) =>
    fetcher(url),
  );

  if (!serverStatus) {
    return null;
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="text-xs text-muted-foreground cursor-pointer">
            {serverStatus.ai_running_status === "running" && (
              // Link to AI Control page
              <a href="/inference">
                <span className="relative inline-block">
                  <span className="animate-ping absolute inline-flex size-5 rounded-full bg-green-400 opacity-75"></span>
                  <BrainCircuit className="size-5 text-green" />
                </span>
              </a>
            )}
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <p className="max-w-xs">
            AI Control status: {serverStatus.ai_running_status}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function AccountTopBar() {
  const { session, proUser, logout } = useAuth();

  // Get first letter of email for avatar
  const getInitial = (email: string) => {
    return email ? email.charAt(0).toUpperCase() : "U";
  };

  if (!session) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <div className="relative cursor-pointer pr-1">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-primary text-primary-foreground">
              {getInitial(session.user_email)}
            </AvatarFallback>
          </Avatar>
          {/* Badge overlay */}
          <div
            className={`absolute -bottom-2 -right-3 px-1 py-0.5 font-extrabold rounded-sm outline-[0.5px] outline-white font- ${
              proUser
                ? "bg-black text-green-500 text-[10px]"
                : "bg-gray-200 text-black text-[10px]"
            }`}
          >
            {proUser ? "PRO" : "FREE"}
          </div>
        </div>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem className="text-sm text-muted-foreground">
          <div className="flex flex-col gap-y-1">
            <div>{session.user_email}</div>
            <div>{proUser ? "Pro User" : "Free User"}</div>
          </div>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        {!proUser && (
          <DropdownMenuItem asChild>
            <a
              href="https://phospho.ai/pro?utm_source=phosphobot_app"
              className="flex items-center"
              target="_blank"
            >
              <TestTubeDiagonal className="mr-1 size-4 text-green-500" />
              Upgrade to Pro
            </a>
          </DropdownMenuItem>
        )}
        {proUser && (
          <DropdownMenuItem asChild>
            <a
              href="https://billing.phospho.ai/p/login/bIYdRr5xq8BhdagbII"
              className="flex items-center"
              target="_blank"
            >
              <TestTubeDiagonal className="mr-1 size-4 text-green-500" />
              Manage your subscription
            </a>
          </DropdownMenuItem>
        )}
        <DropdownMenuItem asChild>
          <a
            href="mailto:contact@phospho.ai"
            className="flex items-center"
            target="_blank"
          >
            <Mail className="mr-1 h-4 w-4" />
            Contact us
          </a>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={logout}>
          <LogOut className="mr-1 h-4 w-4" />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function TopBar() {
  const currentPath = window.location.pathname;
  const { session } = useAuth();
  const navigate = useNavigate();

  const matchedRoute = routeMap.find(({ path, isPrefix }) =>
    isPrefix ? currentPath.startsWith(path) : currentPath === path,
  );

  return (
    <div className="fixed top-0 left-0 right-0 z-50 flex flex-col md:flex-row justify-between items-center gap-4 p-4 bg-background border-b">
      {currentPath === "/" && (
        <div className="flex-1">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-green-500">
            phosphobot
          </h1>
        </div>
      )}
      {currentPath !== "/" && (
        <div className="flex-1">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-green-500">
            {matchedRoute?.title ?? "phosphobot"}
          </h1>
        </div>
      )}

      <div className="flex items-center gap-2 md:w-auto">
        {/* Back button on mobile */}
        <MobileMenu />
        <ServerIP />
        <AIControlStatus />
        <RecordingStatus />
        <RobotStatusDropdown />
        <Button variant="outline" asChild>
          <a
            href="https://docs.phospho.ai/welcome"
            className="flex items-center gap-1 text-sm"
            target="_blank"
            rel="noopener noreferrer"
          >
            <BookText className="size-5" />
            Documentation
          </a>
        </Button>
        <ThemeToggle />
        <div className="flex items-center gap-2">
          {session ? (
            <AccountTopBar />
          ) : (
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                onClick={() => {
                  navigate("/sign-in");
                }}
              >
                Sign in
              </Button>
              <Button
                variant="default"
                onClick={() => {
                  navigate("/sign-up");
                }}
              >
                Sign up
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
