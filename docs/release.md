# Releasing GlySys

crates.io versions are permanent: a published version can be yanked but never
replaced, so a mistake needs a new patch release. Never move or reuse a
version tag.

The crates are published in the order of
[`scripts/release-crates.txt`](../scripts/release-crates.txt):

```text
glysys  glysys-opt  glysys-energy  glysys-dynamics  glysys-gpu  glysys-runtime  glysys-md
```

Every crate in the list carries the workspace version, and the crates depend
on each other at that exact version. GlySys takes crabWURCS from its git tag
during development; the published manifests point at the same version on
crates.io, which must be released first.

## Prepare

1. Set the new version in `Cargo.toml` and in every crate of
   `crates/*/Cargo.toml` (the `version` field and the internal dependency
   versions), then run `cargo update -w` so `Cargo.lock` follows.
2. Turn the changelog's `Unreleased` section into `## VERSION - DATE`.
3. From a clean checkout, run the CI checks and the release check:

   ```console
   cargo fmt --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace --all-targets
   cargo test --workspace --doc
   scripts/check_release.sh VERSION
   ```

   `check_release.sh` checks the versions, the lock file, the changelog, that
   the crate list names every publishable member in dependency order, and that
   git dependencies (crabWURCS) are on crates.io at the required version. It
   then packages all the crates together, so Cargo verifies each one against
   the others as they will be on crates.io, and reports their sizes.
4. Merge the release pull request.

## Publish from a maintainer machine

This is the current route (the repository has no crates.io token secret).
The machine needs `cargo login` once with a token that may publish the
`glysys*` crates.

```console
git fetch origin && git checkout --detach origin/main
scripts/check_release.sh VERSION
git tag -a vVERSION -m "GlySys VERSION" && git push origin vVERSION
scripts/publish_crates.sh VERSION
```

`publish_crates.sh` runs `cargo publish -p <crate> --locked` for each crate
in order and waits until the crate is in the crates.io index before the next.
A crate that is already in the index at VERSION is skipped, so the script can
be run again after an interruption. Without the script, the same sequence is

```console
cargo publish -p glysys --locked
cargo publish -p glysys-opt --locked
cargo publish -p glysys-energy --locked
cargo publish -p glysys-dynamics --locked
cargo publish -p glysys-gpu --locked
cargo publish -p glysys-runtime --locked
cargo publish -p glysys-md --locked
```

waiting after each until `cargo search <crate>` (or
`https://index.crates.io/gl/ys/<crate>`) shows VERSION.

Pushing the tag also starts the release workflow. It validates the tag,
skips publication with a notice when there is no token, and creates the
GitHub release. Once the crates are on crates.io, run the workflow again by
hand (Actions, Release, Run workflow, VERSION) to attach the published
`.crate` files and their `SHA256SUMS` to the release.

## Publish from GitHub Actions

With a `CARGO_REGISTRY_TOKEN` repository secret, pushing the tag is the whole
release: `.github/workflows/release.yml` repeats the CI checks and
`check_release.sh`, publishes with `publish_crates.sh`, then attaches the
`.crate` files downloaded from crates.io (checked against the index) and their
`SHA256SUMS` to the GitHub release.

## After a release

ReGlyco builds against these crates. Its release waits until the new GlySys
version is in the crates.io index (its `docs/release.md`).
