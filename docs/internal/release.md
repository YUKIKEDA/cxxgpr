# Release procedure

cxxgpr is distributed through GitHub Releases. C++ has no standard package manager, so each release carries prebuilt packages for the supported platforms, the documentation sites and checksums. Source archives are attached by GitHub automatically. Users can also consume the repository through `FetchContent` at the release tag.

Publishing a release is the owner's decision. Automation prepares a **draft**; the owner triggers the `release` workflow from the GitHub Actions UI (with `create_draft_release: true` on `main`, which validates, builds, tests, packages, creates and pushes the annotated tag `vX.Y.Z` after everything passes, and creates the draft release) or pushes a tag directly, then publishes the draft.

## What a release contains

| Asset | Content |
|---|---|
| `cxxgpr-<version>-windows-x64-msvc.zip` | Installed tree built with MSVC in Release: headers, `cxxgpr.lib`, CMake package, `LICENSE`, `THIRD_PARTY_NOTICES.md` |
| `cxxgpr-<version>-linux-x64-gcc14.tar.gz` | The same, built with GCC 14 |
| `cxxgpr-<version>-docs.zip` | The English and Japanese documentation sites (`en/`, `ja/`) |
| `SHA256SUMS.txt` | SHA-256 of every asset above |
| Source code (zip, tar.gz) | Added by GitHub |

The packages are static-library installs. They locate Eigen, nlohmann/json and OpenMP with `find_dependency`, so users must have those available when they configure (for example through vcpkg).

## Steps for the owner

1. Make sure `main` is green and contains everything for the release.
2. Set the version in the top-level `CMakeLists.txt` (`project(cxxgpr VERSION X.Y.Z ...)`), update `CHANGELOG.md` (move `Unreleased` entries under the new version and date), and merge that change through a pull request.
3. Run the publication review (`publish-review` Skill; `just doc-check`, `just arch-check`, `just trace-check`) on the merge commit.
4. Trigger the `release` workflow from GitHub (**Actions** → **release** → **Run workflow** on branch `main`, enabling `create_draft_release`). After all platform builds, tests, example runs, and documentation checks pass, the workflow automatically creates and pushes annotated tag `vX.Y.Z` on that `main` commit and creates the draft release. (Alternatively, an owner may still push the tag manually: `git tag -a vX.Y.Z -m "cxxgpr X.Y.Z"` and `git push origin vX.Y.Z`.)
5. Open the new **draft** release on GitHub, check the assets and the generated notes, edit the notes, and publish.

A dry run is available without creating a tag or release: run the `release` workflow manually (`workflow_dispatch`) with `create_draft_release: false` (the default). It builds, tests, packages and uploads the artifacts to the workflow run, and never creates a tag or release. Pull requests that touch the workflow or `tools/release/` run the same dry run.

## What the workflow checks

- For tag pushes, the tag equals `v` plus the version in `CMakeLists.txt`; for `workflow_dispatch` with `create_draft_release: true`, it verifies `ref == refs/heads/main` and that `vX.Y.Z` does not already exist on `origin`, failing before building anything if either check fails.
- On each platform: Release build, the full Release test suite (which includes the install-and-consume package test), installation into a staging directory, and the documentation examples built against that installed package and run, exactly as a user project would.
- The documentation sites build with warnings as errors, and every public API symbol is covered in both languages (`tools/qa/doc_api_coverage.py`).
- Only then does the `publish` job create the draft release (tags only).

## Files

| File | Role |
|---|---|
| `.github/workflows/release.yml` | The workflow |
| `tools/release/package.py` | `version`, `check-version`, `archive` and `checksums` helpers used by the workflow (also runnable locally) |
