#!/bin/bash
# Build the native launcher without modifying the backend's release workflow.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

source_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$source_dir/../.." && pwd)"
mkdir -p "$1"
out="$(cd "$1" && pwd)"

# Keep user files intact when rebuilding into a previously used directory.
if [[ -e "$out/phosphobot.app" || -L "$out/phosphobot.app" ]]; then
    bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$out/phosphobot.app/Contents/Info.plist" 2>/dev/null || true)"
    if [[ -L "$out/phosphobot.app" || "$bundle_id" != "ai.phospho.phosphobot.launcher" ]]; then
        echo "An unrelated phosphobot.app already exists in $out." >&2
        exit 1
    fi
fi

scratch="$(mktemp -d "$out/.phosphobot-build.XXXXXXXX")"
trap 'rm -rf "$scratch"' EXIT
image_root="$scratch/disk"
app="$image_root/phosphobot.app"
resources="$app/Contents/Resources"
iconset="$scratch/phosphobot.iconset"
mkdir -p "$app/Contents/MacOS" "$resources" "$iconset"
cp "$source_dir/Info.plist" "$app/Contents/"
cp "$repo_root/LICENSE" "$resources/LICENSE.txt"
plutil -lint "$app/Contents/Info.plist" >/dev/null
for arch in arm64 x86_64; do
    xcrun swiftc -swift-version 5 -parse-as-library -O -target "$arch-apple-macos13.0" \
        -framework AppKit "$source_dir"/*.swift -o "$scratch/launcher-$arch"
done
executable="$app/Contents/MacOS/phosphobot-launcher"
xcrun lipo -create "$scratch/launcher-arm64" "$scratch/launcher-x86_64" -output "$executable"
xcrun lipo -verify_arch arm64 "$executable"
xcrun lipo -verify_arch x86_64 "$executable"

# Reuse the dashboard's square phospho logo. iconutil requires the complete
# Apple iconset naming convention, including the Retina versions.
icon="$repo_root/phosphobot/resources/swagger-ui/favicon.png"
for size in 16 32 128 256 512; do
    sips --resampleHeightWidth "$size" "$size" "$icon" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    sips --resampleHeightWidth "$((size * 2))" "$((size * 2))" "$icon" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$resources/phosphobot.icns"

# Development builds use an ad hoc identity. A release publisher can supply a
# Developer ID Application identity from its own keychain.
identity="${CODESIGN_IDENTITY:--}"
if [[ "$identity" == "-" ]]; then
    codesign --force --sign - "$app"
else
    codesign --force --options runtime --timestamp --sign "$identity" "$app"
fi
codesign --verify --all-architectures --strict "$app"

ln -s /Applications "$image_root/Applications"
dmg="phosphobot-macos-universal.dmg"
hdiutil create -volname phosphobot -format UDZO -fs HFS+ -srcfolder "$image_root" "$scratch/$dmg" >/dev/null
if [[ "$identity" != "-" ]]; then
    codesign --timestamp --sign "$identity" "$scratch/$dmg"
fi
hdiutil verify "$scratch/$dmg" >/dev/null

if [[ -d "$out/phosphobot.app" ]]; then
    mv "$out/phosphobot.app" "$scratch/previous.app"
fi
mv "$app" "$out/phosphobot.app"
mv -f "$scratch/$dmg" "$out/$dmg"
printf 'Built %s\nBuilt %s\n' "$out/phosphobot.app" "$out/$dmg"
