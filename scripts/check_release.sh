#!/usr/bin/env bash
# Check that the workspace is ready to release VERSION, and package every
# crate that scripts/publish_crates.sh will upload. Nothing is uploaded.
#
#   scripts/check_release.sh [VERSION]   # default: the version in Cargo.toml
#
# Run it from a clean checkout of the commit to be tagged.
set -euo pipefail
cd "$(dirname "$0")/.."

fail() {
  echo "check_release: $*" >&2
  exit 1
}

mapfile -t crates < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' scripts/release-crates.txt)
root_version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)
version=${1:-$root_version}
version=${version#v}

[ "$root_version" = "$version" ] || fail "Cargo.toml is at $root_version, not $version"
grep -q "^## $version " CHANGELOG.md || fail "CHANGELOG.md has no '## $version - DATE' section"
! grep -qi '^## Unreleased' CHANGELOG.md || fail "CHANGELOG.md still has an Unreleased section"

metadata=$(cargo metadata --no-deps --format-version 1 --locked) ||
  fail "Cargo.lock is not up to date (cargo metadata --locked failed)"

# The list holds every publishable workspace member and nothing else.
listed=$(printf '%s\n' "${crates[@]}" | sort)
publishable=$(jq -r '.packages[] | select(.publish != []) | .name' <<<"$metadata" | sort)
[ "$listed" = "$publishable" ] ||
  fail "scripts/release-crates.txt does not list the publishable workspace members:
$(diff <(echo "$listed") <(echo "$publishable") || true)"

declare -A position
for index in "${!crates[@]}"; do
  position[${crates[$index]}]=$index
done

for crate in "${crates[@]}"; do
  package=$(jq -c --arg name "$crate" '.packages[] | select(.name == $name)' <<<"$metadata")
  crate_version=$(jq -r .version <<<"$package")
  [ "$crate_version" = "$version" ] || fail "$crate is at $crate_version, not $version"
  [ "$(jq -r .rust_version <<<"$package")" = "1.97" ] || fail "$crate does not declare rust-version 1.97"
  grep -A1 "^name = \"$crate\"$" Cargo.lock | grep -qx "version = \"$version\"" ||
    fail "Cargo.lock has no $crate $version"
  while IFS=$'\t' read -r dep kind req source; do
    if [ -n "${position[$dep]+set}" ]; then
      # A path-only dev-dependency is dropped from the published manifest.
      [ "$kind" = dev ] && [ "$req" = '*' ] && continue
      [ "$req" = "^$version" ] || fail "$crate requires $dep $req, not ^$version"
      [ "${position[$dep]}" -lt "${position[$crate]}" ] ||
        fail "$crate is listed before its dependency $dep in scripts/release-crates.txt"
    elif [[ "$source" == git+* ]]; then
      [ "$req" != '*' ] || fail "$crate takes $dep from git without a version for crates.io"
    fi
  done < <(jq -r '.dependencies[] | [.name, (.kind // "normal"), .req, (.source // "-")] | @tsv' <<<"$package")
done

# External dependencies taken from git must already be on crates.io at the
# version the manifest asks for, or the upload is refused.
index_url() {
  local name=$1
  case ${#name} in
    1) echo "https://index.crates.io/1/$name" ;;
    2) echo "https://index.crates.io/2/$name" ;;
    3) echo "https://index.crates.io/3/${name:0:1}/$name" ;;
    *) echo "https://index.crates.io/${name:0:2}/${name:2:2}/$name" ;;
  esac
}
while IFS=$'\t' read -r dep req; do
  wanted=${req#^}
  curl --fail --silent --show-error "$(index_url "$dep")" |
    jq --exit-status --arg version "$wanted" 'select(.vers == $version)' >/dev/null ||
    fail "$dep $wanted is not on crates.io"
done < <(jq -r '.packages[] | select(.publish != []) | .dependencies[]
  | select((.source // "") | startswith("git+")) | [.name, .req] | @tsv' <<<"$metadata" | sort -u)

if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  echo "check_release: warning: the working tree has uncommitted changes; cargo publish will refuse it" >&2
fi

# Packaging the crates together lets Cargo verify each one against the others
# as they will be on crates.io, before any of them is uploaded.
package_args=()
for crate in "${crates[@]}"; do
  package_args+=(-p "$crate")
done
cargo package --locked --allow-dirty "${package_args[@]}"

target_dir=$(cargo metadata --format-version 1 --no-deps | jq -r .target_directory)
for crate in "${crates[@]}"; do
  file="$target_dir/package/$crate-$version.crate"
  size=$(stat -c %s "$file")
  [ "$size" -lt 10000000 ] || fail "$file is $size bytes; crates.io accepts at most 10 MB"
  printf '%-18s %8d KiB\n' "$crate" $((size / 1024))
done

echo "release metadata and crate packages are consistent at $version"
echo "publish order: ${crates[*]}"
