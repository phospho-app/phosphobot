import { statSync, existsSync, readdirSync } from "fs";
import { join } from "path";

// Paths to check (now relative to src-tauri directory)
const sourceDir = "../phosphobot/phosphobot";
const configFile = "../phosphobot/pyproject.toml";
const outputDir = "../phosphobot/target/release";

const tauriOutputDir = "binaries";

// Helper: Get the latest modification time of all files in a directory recursively
function getLastModifiedTime(dir) {
  let latestTime = 0;

  // Read all files and subdirectories
  for (const file of readdirSync(dir)) {
    const filePath = join(dir, file);
    const stats = statSync(filePath);

    if (stats.isDirectory()) {
      // Recurse into subdirectories
      latestTime = Math.max(latestTime, getLastModifiedTime(filePath));
    } else {
      // Compare file modification times
      latestTime = Math.max(latestTime, stats.mtimeMs);
    }
  }

  return latestTime;
}

// Check if a build is needed
function isBuildNeeded() {
  // If the output directory does not exist, we need to build
  if (!existsSync(outputDir)) {
    console.log("BUILD_NEEDED"); // Output a specific string to stdout
    process.exit(0)
  }

  // If the tauri output directory does not exist, we need to build
  if (!existsSync(tauriOutputDir)) {
    console.log("BUILD_NEEDED"); // Output a specific string to stdout
    process.exit(0)
  }

  // If the tauri output directory exists but is empty, we need to build
  if (readdirSync(tauriOutputDir).length === 0) {
    console.log("BUILD_NEEDED"); // Output a specific string to stdout
    process.exit(0)
  }

  // Get the last modification times
  const sourceTime = getLastModifiedTime(sourceDir);
  // console.log("sourceTime", sourceTime);
  const configTime = existsSync(configFile) ? statSync(configFile).mtimeMs : 0;
  // console.log("configTime", configTime);
  const outputTime = statSync(outputDir).mtimeMs;
  // console.log("outputTime", outputTime);

  // Build if any source or config file is newer than the output
  return Math.max(sourceTime, configTime) > outputTime;
}

// Run the check
if (isBuildNeeded()) {
  console.log("BUILD_NEEDED"); // Output a specific string to stdout
  process.exit(0)
} else {
  process.exit(0)
} 