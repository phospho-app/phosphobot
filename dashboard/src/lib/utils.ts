import { type ClassValue, clsx } from "clsx";
import { toast } from "sonner";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const fetcherSWR = async (
  url: string,
  method?: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  body?: any | null,
) => {
  // Use same base URL logic as fetchWithBaseUrl
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
  const fullUrl = API_BASE_URL ? `${API_BASE_URL}${url}` : url;
  
  const response = await fetch(fullUrl, {
    method: method || "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });

  if (response.ok) return await response.json();

  console.error("Error fetching data:", response);

  let errorMessage = response.statusText;
  try {
    const errorData = await response.json();
    if (errorData.detail) errorMessage = errorData.detail;
    if (typeof errorMessage === "object") {
      errorMessage = JSON.stringify(errorMessage);
    }
  } catch {
    // Use default statusText if JSON parsing fails
    errorMessage = response.statusText;
  }

  toast.error(`Error: ${errorMessage}`);
  return undefined;
};

const fetchWithBaseUrl = async (
  endpoint: string,
  method?: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  body?: any | null,
) => {
  // Determine the base URL:
  // - If VITE_API_BASE_URL is set (Tauri with sidecar), use it
  // - Otherwise use relative URLs (old Python method serving everything)
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
  const url = API_BASE_URL ? `${API_BASE_URL}${endpoint}` : endpoint;
  
  const response = await fetch(url, {
    method: method || "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });

  if (response.ok) return await response.json();

  console.error("Error fetching data:", response);

  let errorMessage = response.statusText;
  try {
    const errorData = await response.json();
    if (errorData.detail) errorMessage = errorData.detail;
    if (typeof errorMessage === "object") {
      errorMessage = JSON.stringify(errorMessage);
    }
  } catch {
    // Fallback to statusText if JSON parsing fails
    errorMessage = response.statusText;
  }

  toast.error(`Error: ${errorMessage}`);
  return undefined;
};

export { cn, fetcherSWR as fetcher, fetchWithBaseUrl };
