// Prevent an unnecessary main function on mobile builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    edge_gui_lib::run();
}
