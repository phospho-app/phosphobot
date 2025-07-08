#!/usr/bin/env node

import { execSync } from 'child_process';
import { existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const binariesDir = join(__dirname, '..', 'binaries');

console.log('Resetting Python package binaries...');

try {
  if (!existsSync(binariesDir)) {
    console.log('No binaries directory found, nothing to reset.');
    process.exit(0);
  }

  const files = readdirSync(binariesDir).filter(f => f.startsWith('phosphobot-'));
  
  if (files.length === 0) {
    console.log('No phosphobot binaries found to reset.');
    process.exit(0);
  }

  files.forEach(file => {
    const binaryPath = join(binariesDir, file);
    try {
      execSync(`"${binaryPath}" self restore`, { 
        stdio: 'pipe',
        timeout: 30000 // 30 second timeout
      });
      console.log(`Reset successful: ${file}`);
    } catch (error) {
      console.log(`No reset needed for: ${file}`);
    }
  });

  console.log('Python package reset completed.');
} catch (error) {
  console.error('Reset failed:', error.message);
  process.exit(1);
} 