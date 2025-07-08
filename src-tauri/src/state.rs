use std::sync::Mutex;
use tauri_plugin_shell::process::CommandChild;

pub struct AppState {
    pub sidecar_child: Mutex<Option<CommandChild>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            sidecar_child: Mutex::new(None),
        }
    }
} 