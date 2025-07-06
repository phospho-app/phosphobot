#!/usr/bin/env node

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const TAURI_DIR = join(__dirname, '..', '..');
const KEYS_DIR = join(TAURI_DIR, 'updater-keys');
const PRIVATE_KEY_FILE = join(KEYS_DIR, 'phosphobot.key');
const PUBLIC_KEY_FILE = join(KEYS_DIR, 'phosphobot.pub');
const TAURI_CONF_FILE = join(TAURI_DIR, 'src-tauri', 'tauri.conf.json');

console.log('🔐 Generating Tauri Updater Keys...');

// Create keys directory if it doesn't exist
if (!existsSync(KEYS_DIR)) {
    mkdirSync(KEYS_DIR, { recursive: true });
}

// Check if keys already exist
if (existsSync(PRIVATE_KEY_FILE) && existsSync(PUBLIC_KEY_FILE)) {
    console.log('⚠️  Keys already exist. Skipping generation.');
    console.log(`Private key: ${PRIVATE_KEY_FILE}`);
    console.log(`Public key: ${PUBLIC_KEY_FILE}`);
    process.exit(0);
}

try {
    // Generate the keypair using Tauri CLI
    console.log('Generating keypair...');
    execSync(`npx tauri signer generate -w "${PRIVATE_KEY_FILE}"`, { 
        stdio: 'inherit',
        cwd: TAURI_DIR
    });

    // Read the generated public key
    const publicKey = readFileSync(PUBLIC_KEY_FILE, 'utf-8').trim();
    
    // Update tauri.conf.json with the public key
    const tauriConf = JSON.parse(readFileSync(TAURI_CONF_FILE, 'utf-8'));
    
    if (tauriConf.plugins && tauriConf.plugins.updater) {
        tauriConf.plugins.updater.pubkey = publicKey;
        writeFileSync(TAURI_CONF_FILE, JSON.stringify(tauriConf, null, 2));
        console.log('✅ Updated tauri.conf.json with public key');
    }

    console.log('\n🎉 Updater keys generated successfully!');
    console.log('\n📋 Next steps:');
    console.log('1. For local development, copy .env.example to .env and update the paths:');
    console.log('   cp .env.example .env');
    console.log('   # Then edit .env to set the correct paths');
    console.log('\n2. For GitHub Actions, add these as secrets:');
    console.log('   - TAURI_PRIVATE_KEY: (content of the private key file)');
    console.log('   - TAURI_KEY_PASSWORD: (password if you set one)');
    console.log('\n3. Example .env file content:');
    console.log(`   TAURI_PRIVATE_KEY=${PRIVATE_KEY_FILE}`);
    console.log('   TAURI_KEY_PASSWORD=  # if you set a password');
    console.log('\n⚠️  IMPORTANT: Keep your private key safe and never commit it to version control!');
    console.log(`   Private key location: ${PRIVATE_KEY_FILE}`);
    console.log(`   Public key location: ${PUBLIC_KEY_FILE}`);

} catch (error) {
    console.error('❌ Failed to generate keys:', error.message);
    process.exit(1);
} 