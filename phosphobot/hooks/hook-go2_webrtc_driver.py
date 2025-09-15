import os

from PyInstaller.utils.hooks import collect_data_files, get_package_paths

# We need to ensure this is the correct Python package name that PyInstaller can find.
_package_name = "go2_webrtc_driver"
_sub_dir_containing_wasm = "lidar"  # The subdirectory within go2_webrtc_driver

datas = []


try:
    # Verify the package exists to avoid PyInstaller errors if it's mistyped
    # get_package_paths returns the base directory(ies) of the package
    pkg_base_dirs = get_package_paths(_package_name)
    if not pkg_base_dirs:
        raise ImportError(f"Package '{_package_name}' not found by PyInstaller.")

    # This will collect files from the 'lidar' subdirectory of the 'go2_webrtc_driver' package.
    # The destination paths in the bundle will be like 'go2_webrtc_driver/lidar/filename'.
    collected_files = collect_data_files(
        package=_package_name,
        subdir=_sub_dir_containing_wasm,
        include_py_files=False,  # We only want data files like .wasm
    )

    if collected_files:
        datas.extend(collected_files)
        print(
            f"HOOK go2_webrtc_driver: Bundled data from '{_package_name}/{_sub_dir_containing_wasm}': {collected_files}"
        )
    else:
        # Construct the expected path to check if it exists for better logging
        expected_source_path = "unknown"
        if pkg_base_dirs:
            expected_source_path = os.path.join(
                pkg_base_dirs[0],
                _package_name.replace(".", os.sep),
                _sub_dir_containing_wasm,
            )
        print(
            f"HOOK WARNING: No data files found by collect_data_files for '{_package_name}' in subdir '{_sub_dir_containing_wasm}'. Searched near: {expected_source_path}"
        )

except ImportError as e:
    print(f"HOOK ERROR (go2_webrtc_driver): {e}. Cannot collect data files.")
except Exception as e:
    # Catch other potential errors during hook execution
    print(f"HOOK UNEXPECTED ERROR (go2_webrtc_driver): {e}")
