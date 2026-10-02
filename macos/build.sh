#!/usr/bin/env bash
# Builds Trainform.app and dist/Trainform-<version>-arm64.dmg for Apple silicon Macs (macOS 13+).
# Needs the Xcode command line tools (swiftc) and Node. Python and PostgreSQL are downloaded (pinned, checksummed)
# into macos/.cache and bundled, so the app runs on a Mac with neither installed.
set -euo pipefail
cd "$(dirname "$0")/.."

PY_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20261001/cpython-3.14.8+20261001-aarch64-apple-darwin-install_only_stripped.tar.gz
PY_SHA=69e48cd7f54261df5b6abbd374f69fe6e496fa781d271592aeef49cd5ffbea6c
PG_URL=https://repo1.maven.org/maven2/io/zonky/test/postgres/embedded-postgres-binaries-darwin-arm64v8/17.11.0/embedded-postgres-binaries-darwin-arm64v8-17.11.0.jar
PG_SHA=a1c2786acb0c398f9b2d76806fc52f5dc8b222cbc8e9383a9b9702084daaf3a5

VERSION=$(plutil -extract CFBundleShortVersionString raw macos/Info.plist)
CACHE=macos/.cache
STAGE=dist/dmg-root  # becomes the DMG's contents: Trainform.app next to an Applications link
APP=$STAGE/Trainform.app
RES=$APP/Contents/Resources
DMG=dist/Trainform-$VERSION-arm64.dmg

fetch() {  # url sha256 -> path of the verified download
  local file="$CACHE/${1##*/}"
  if [ ! -f "$file" ]; then
    curl -fsSL --create-dirs -o "$file.part" "$1"
    mv "$file.part" "$file"
  fi
  echo "$2  $file" | shasum -a 256 -c - >/dev/null || { rm -f "$file"; echo "checksum mismatch: $1" >&2; exit 1; }
  echo "$file"
}

echo "==> Frontend"
(cd frontend && npm ci --no-audit --no-fund && npm run build)

echo "==> App bundle"
rm -rf "$STAGE" "$DMG"
mkdir -p "$APP/Contents/MacOS" "$RES/backend" "$RES/postgres"
cp macos/Info.plist "$APP/Contents/"
swiftc -O -target arm64-apple-macos13 -o "$APP/Contents/MacOS/Trainform" macos/Trainform.swift
cp -R frontend/dist "$RES/frontend"
rsync -a --exclude __pycache__ backend/app "$RES/backend/"

echo "==> Python"
tar -xzf "$(fetch "$PY_URL" "$PY_SHA")" -C "$RES"  # unpacks to $RES/python
"$RES/python/bin/python3" -m pip install --quiet --disable-pip-version-check --no-compile \
  -r <(grep -v '^pytest' backend/requirements.txt)
rm -rf "$RES"/python/lib/python3.*/test  # the stdlib's own test suite: large, unused
# The app runs with PYTHONDONTWRITEBYTECODE (writing into a signed bundle breaks it), so compile now.
"$RES/python/bin/python3" -m compileall -q -j 0 "$RES/python/lib" "$RES/backend" >/dev/null

echo "==> PostgreSQL"
unzip -p "$(fetch "$PG_URL" "$PG_SHA")" postgres-darwin-arm_64.txz | tar -xJf - -C "$RES/postgres"
# The jar ships each dylib version alias as a full universal copy: link the aliases and keep only arm64.
(cd "$RES/postgres/lib" && shasum ./*.dylib | sort | awk '$1 == h { print "ln -sf " f " " $2; next } { h = $1; f = $2 }' | sh)
find "$RES/postgres" -type f \( -path '*/bin/*' -o -name '*.dylib' \) | while read -r f; do
  if lipo -archs "$f" | grep -q x86_64; then lipo "$f" -thin arm64 -output "$f"; fi
done

echo "==> Icon"
ICONS=$(mktemp -d)
qlmanage -t -s 1024 -o "$ICONS" frontend/public/favicon.svg >/dev/null
mkdir "$ICONS/Trainform.iconset"
for s in 16 32 128 256 512; do
  sips -z $s $s "$ICONS/favicon.svg.png" --out "$ICONS/Trainform.iconset/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$ICONS/favicon.svg.png" --out "$ICONS/Trainform.iconset/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns -o "$RES/Trainform.icns" "$ICONS/Trainform.iconset"
rm -rf "$ICONS"

echo "==> Sign (ad hoc) and package"
# Ad hoc: runs on this Mac and on others after right-click > Open. Distribution needs a Developer ID + notarization.
codesign --force --sign - "$APP"
codesign --verify --strict "$APP"
ln -s /Applications "$STAGE/Applications"
hdiutil create -quiet -volname Trainform -srcfolder "$STAGE" -fs HFS+ -format ULFO -ov "$DMG"

echo "Built $APP ($(du -sh "$APP" | cut -f1)) and $DMG ($(du -sh "$DMG" | cut -f1))"
