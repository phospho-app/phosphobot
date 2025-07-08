#!/usr/bin/env node

import { execSync } from 'child_process';

console.log('Checking if Python package build is needed...');

try {
  // Check if build is needed
  const checkOutput = execSync('npm run python:package:check', { encoding: 'utf8' });
  const needsBuild = checkOutput.includes('BUILD_NEEDED');
  
  if (needsBuild) {
    console.log('Build needed, starting Python package build...');
    execSync('npm run python:package:build', { stdio: 'inherit' });
    
    console.log('Running package reset...');
    execSync('npm run python:package:reset', { stdio: 'inherit' });
    
    console.log('Python package build and reset completed.');
  } else {
    console.log('No build needed, skipping Python package build.');
  }
} catch (error) {
  console.error('Build check failed:', error.message);
  process.exit(1);
} 