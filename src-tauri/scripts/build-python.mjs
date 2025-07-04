#!/usr/bin/env node

import { execSync } from 'child_process';
import { existsSync, mkdirSync, copyFileSync, chmodSync, readdirSync, unlinkSync } from 'fs';
import { join, dirname, basename, extname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const platform = process.platform;
const isWindows = platform === 'win32';
const projectRoot = join(__dirname, '..', '..');
const phosphobotDir = join(projectRoot, 'phosphobot');
const binariesDir = join(__dirname, '..', 'binaries');

console.log(`Building Python package for ${platform}...`);

try {
  // Clean build directories
  const buildDirs = [join(phosphobotDir, 'build'), join(phosphobotDir, 'dist')];
  buildDirs.forEach(dir => {
    if (existsSync(dir)) {
      console.log(`Cleaning ${dir}...`);
      execSync(isWindows ? `rmdir /s /q "${dir}"` : `rm -rf "${dir}"`, { stdio: 'inherit' });
    }
  });

  // Run box package with target architecture
  console.log('Running box package...');
  const boxEnv = { ...process.env };
  
  // Set target architecture for PyApp if we have a specific target
  if (envTarget) {
    console.log(`Setting PYAPP_PLATFORM for target: ${envTarget}`);
    boxEnv.PYAPP_PLATFORM = envTarget;
  }
  
  execSync('uvx --from box-packager box package', {
    cwd: phosphobotDir,
    stdio: 'inherit',
    env: boxEnv
  });

  // Create binaries directory
  if (!existsSync(binariesDir)) {
    mkdirSync(binariesDir, { recursive: true });
  }

  // Get target architecture from environment or host
  const envTarget = process.env.TAURI_ENV_TARGET_TRIPLE;
  let targetTriple;
  
  if (envTarget) {
    targetTriple = envTarget;
    console.log(`Using target from environment: ${targetTriple}`);
  } else {
    const rustInfo = execSync('rustc -Vv', { encoding: 'utf8' });
    const hostMatch = rustInfo.match(/host: (.+)/);
    targetTriple = hostMatch ? hostMatch[1] : 'unknown';
    console.log(`Using host target: ${targetTriple}`);
  }

  console.log(`Target triple: ${targetTriple}`);

  // Copy and rename binaries
  const releaseDir = join(phosphobotDir, 'target', 'release');
  if (existsSync(releaseDir)) {
    const files = readdirSync(releaseDir);
    
    files.forEach(file => {
      const srcPath = join(releaseDir, file);
      const ext = extname(file);
      const baseName = basename(file, ext);
      
      // Skip directories
      if (!existsSync(srcPath) || readdirSync(releaseDir, { withFileTypes: true }).find(dirent => dirent.name === file && dirent.isDirectory())) {
        return;
      }

      const finalExt = ext || (isWindows ? '.exe' : '');
      const destName = `${baseName}-${targetTriple}${finalExt}`;
      const destPath = join(binariesDir, destName);
      
      console.log(`Copying ${file} -> ${destName}`);
      copyFileSync(srcPath, destPath);
      
      // Make executable on Unix systems
      if (!isWindows) {
        chmodSync(destPath, 0o755);
      }
    });

    // Create the main binary that Tauri expects
    const phosphobotBinary = files.find(file => {
      const baseName = basename(file, extname(file));
      return baseName === 'phosphobot';
    });
    
    if (phosphobotBinary) {
      const srcName = `phosphobot-${targetTriple}${extname(phosphobotBinary) || (isWindows ? '.exe' : '')}`;
      const srcPath = join(binariesDir, srcName);
      
      // Create the binary name that Tauri expects
      const mainBinaryName = 'phosphobot'; // Always without extension for Tauri config
      const mainBinaryPath = join(binariesDir, mainBinaryName);
      
      if (existsSync(srcPath)) {
        if (existsSync(mainBinaryPath)) {
          unlinkSync(mainBinaryPath);
        }
        
        if (isWindows) {
          // On Windows, also create the .exe version that might be expected
          const exePath = join(binariesDir, 'phosphobot.exe');
          copyFileSync(srcPath, exePath);
          copyFileSync(srcPath, mainBinaryPath);
          console.log(`Created Windows binaries: phosphobot and phosphobot.exe`);
        } else {
          // On Unix, create a symlink
          execSync(`ln -sf "${srcName}" "${mainBinaryName}"`, {
            cwd: binariesDir,
            stdio: 'inherit'
          });
          console.log(`Created Unix binary symlink: ${mainBinaryName} -> ${srcName}`);
        }
      }
    }
  }

  console.log('Python package build completed successfully!');
  
} catch (error) {
  console.error('Python package build failed:', error.message);
  process.exit(1);
} 