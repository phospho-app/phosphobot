#!/usr/bin/env node

import { existsSync, rmSync } from 'fs';
import { join, resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const dirsToClean = [
  'target',
  'binaries',
  '../phosphobot/target'
];

console.log('Cleaning build directories...');

try {
  dirsToClean.forEach(dir => {
    const fullPath = resolve(__dirname, '..', dir);
    
    if (existsSync(fullPath)) {
      rmSync(fullPath, { recursive: true, force: true });
      console.log(`✓ Cleaned: ${fullPath}`);
    } else {
      console.log(`⚬ Skipped (not found): ${fullPath}`);
    }
  });

  console.log('Clean completed successfully.');
} catch (error) {
  console.error('Clean failed:', error.message);
  process.exit(1);
} 