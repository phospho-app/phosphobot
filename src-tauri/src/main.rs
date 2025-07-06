// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod state;

use log::info;
use state::AppState;
use tauri::{Manager, State, RunEvent};
use tauri_plugin_shell::{ShellExt, process::CommandEvent};

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn is_phosphobot_server_running() -> bool {
    let client = tauri_plugin_http::reqwest::Client::new();
    match client.get("http://127.0.0.1:8432/status").send().await {
        Ok(resp) => {
            log::info!("Status check response: {:?}", resp.status());
            resp.status().is_success()
        },
        Err(e) => {
            log::warn!("Status check failed: {}", e);
            false
        }
    }
}

#[tauri::command] 
async fn start_phosphobot_server<'a>(
    app: tauri::AppHandle,
    state: State<'a, AppState>,
) -> Result<String, String> {
    info!("Attempting to launch sidecar: phosphobot");
    
    let sidecar_command = app.shell().sidecar("phosphobot").map_err(|e| {
        log::error!("Failed to get sidecar: {}", e);
        e.to_string()
    })?;
    
    log::info!("Sidecar command successfully created.");
    
    let (mut rx, child) = sidecar_command
        .args([
            "run",
            "--host=127.0.0.1", 
            "--port=8432",
            "--simulation=headless",
            "--simulate-cameras",
            "--only-simulation",
            "--no-telemetry",
            "--api-only"
        ])
        .spawn()
        .map_err(|e| {
            log::error!("Failed to spawn sidecar: {}", e);
            e.to_string()
        })?;
    
    log::info!("Sidecar process spawned with PID: {:?}", child.pid());
    
    // Save child handle in global app state so it can be killed later
    {
        let mut child_lock = state.sidecar_child.lock().unwrap();
        *child_lock = Some(child);
    }
    
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    if let Ok(output) = String::from_utf8(line) {
                        log::info!("[phosphobot stdout] {}", output.trim());
                    }
                }
                CommandEvent::Stderr(line) => {
                    if let Ok(output) = String::from_utf8(line) {
                        log::error!("[phosphobot stderr] {}", output.trim());
                    }
                }
                other => {
                    log::debug!("[phosphobot] Other event: {:?}", other);
                }
            }
        }
    });
    
    Ok("Phosphobot server started and logging.".to_string())
}

#[tauri::command]
async fn stop_phosphobot_server<'a>(
    state: State<'a, AppState>,
) -> Result<String, String> {
    info!("Stopping phosphobot server...");
    
    // Kill the sidecar process directly
    let mut child_lock = state.sidecar_child.lock().unwrap();
    if let Some(child) = child_lock.take() {
        match child.kill() {
            Ok(_) => {
                log::info!("Sidecar process killed successfully");
                Ok("Server stopped.".to_string())
            },
            Err(e) => {
                log::error!("Failed to kill sidecar process: {}", e);
                Err(format!("Failed to stop server: {}", e))
            }
        }
    } else {
        log::info!("No sidecar process to kill");
        Ok("No server running.".to_string())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();
    
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // Focus the main window if app is already running
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_focus();
            }
        }))
        .manage(AppState::new())
        .setup(|app| {
            #[cfg(desktop)]
            {
                // Initialize updater plugin - it will handle everything automatically with built-in dialogs
                let _ = app.handle().plugin(tauri_plugin_updater::Builder::new().build());
            }
            
            info!("Phosphobot desktop app is starting...");
            
            // Get the app handle before the async block
            let app_handle = app.handle().clone();
            
            // Auto-start the Python phosphobot server
            tauri::async_runtime::spawn(async move {
                // Check if server is already running
                for i in 0..3 {
                    if is_phosphobot_server_running().await {
                        info!("Phosphobot server is already running.");
                        return;
                    }
                    info!("Phosphobot server not yet running... retry {i}/3");
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
                
                info!("Phosphobot server not running. Launching...");
                let state = app_handle.state::<AppState>();
                let handle_clone = app_handle.clone();
                if let Err(err) = start_phosphobot_server(handle_clone, state).await {
                    log::error!("Error launching phosphobot server: {}", err);
                }
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            is_phosphobot_server_running,
            start_phosphobot_server,
            stop_phosphobot_server
        ])
        .build(tauri::generate_context!())
        .expect("error while running tauri application");
    
    // Handle app exit events to ensure phosphobot server is always stopped
    app.run(move |app_handle, event| {
        match event {
            RunEvent::ExitRequested { api, code, .. } => {
                info!("ExitRequested event received");
                
                // Kill the phosphobot server when app exit is requested
                let app_handle_clone = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    let state = app_handle_clone.state::<AppState>();
                    if let Err(err) = stop_phosphobot_server(state).await {
                        log::error!("Error stopping phosphobot server during exit: {}", err);
                    } else {
                        log::info!("Phosphobot server stopped successfully during exit");
                    }
                });
                
                // If the exit was requested programmatically (code is Some), allow it to proceed
                // If it was requested by user interaction (code is None), prevent it temporarily
                // to allow cleanup to complete, then exit programmatically
                if code.is_none() {
                    api.prevent_exit();
                    
                    // Give a small delay for cleanup to complete, then exit
                    let app_handle_for_exit = app_handle.clone();
                    tauri::async_runtime::spawn(async move {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        app_handle_for_exit.exit(0);
                    });
                }
            }
            _ => {}
        }
    });
}

fn main() {
    run();
} 