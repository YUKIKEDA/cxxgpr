---
name: publish-review
description: >-
  Review cxxgpr's bilingual READMEs, public C++23 headers, Doxygen comments,
  and third-party license notices before release tagging or packaging.
---

# Publish review

Run this review before creating a release tag or distributing a package (`REL-2`), and whenever `include/cxxgpr/**/*.hpp`, `README.md`, `README.ja.md`, or `THIRD_PARTY_NOTICES.md` changes for public release.

Require all repository quality gates to pass before approving:

- `just gate`
- `just format-check`
- `just lint`
- `just lint-python`
- `just skills-check`
- `just trace-check`
- `just arch-check`
- `just doc-check`
- `just results-check`

## Read first

- `.cursor/rules/publish.mdc`
- `.cursor/rules/dev-docs.mdc`
- `.cursor/rules/api-contract.mdc`
- `docs/internal/quality/quality-plan.md` (`QA-09`)
- `.agents/skills/natural-japanese/SKILL.md` (`score` mode, for inspecting Japanese prose in `README.ja.md`)

Do not restate those rules here; audit every public surface directly against them.

## Surfaces to audit

1. `README.md` (English entry point) and `README.ja.md` (Japanese entry point)
2. `LICENSE` (Apache-2.0) and `THIRD_PARTY_NOTICES.md` against `vcpkg.json` and `CMakeLists.txt`
3. Every public header under `include/cxxgpr/**/*.hpp` and its Doxygen comments

Internal planning and specification documents (`docs/internal/design.md`, `docs/internal/persist-format.md`, `docs/internal/roadmap.md`, `docs/internal/adr/`, `docs/internal/quality/`) and `detail` implementation symbols are excluded from the external-reader restrictions below.

## Fail criteria

Read as a C++23 library user who only sees the repository landing page, build files, and `include/cxxgpr/` headers without opening the issue tracker. Reject the release review if any of the following holds:

1. **Third-party or internal header leakage**: Any file under `include/cxxgpr/**/*.hpp` includes `<Eigen/...>`, `<nlohmann/...>`, `<omp.h>`, any `detail/` header, or any path under `src/`.
2. **Internal tracking IDs or local paths exposed**: Public Doxygen comments in `include/cxxgpr/**/*.hpp`, `README.md`, or `README.ja.md` contain internal spec/roadmap/issue identifiers (`REQ-*`, `DES-*`, `FMT-*`, `QA-*`, `M*-*`, `#N`) or hardcoded local filesystem paths (`C:\Users\...`, `/home/...`).
3. **Missing, non-English, or commandless Doxygen comments**: Any C++ file (`include/cxxgpr/**/*.hpp`, `src/**/*.{hpp,cpp}`, `tests/**/*.{hpp,cpp}`, `benchmarks/**/*.cpp`, `qa/**/*.cpp`) lacks `/// @file` / `/// @brief`, any type, enum, enumerator, member, function, or test lacks `///` Doxygen documentation with `@brief` (plus `@tparam`, `@param[in]`/`@param[out]`/`@param[in,out]` with explicit direction tags, and `@return` where applicable), or any C++ comment contains non-English prose.
4. **API contract inaccuracy**: Documentation contradicts actual API signatures or contracts, including:
   - Column-major (`column-major`) storage layout for matrices passed via `std::span<const double>` (`x`, `n_rows`, `n_cols`);
   - `Gpr::fit(...)` and `Gpr::factor(...)` being `const` member functions returning `[[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError>` without throwing exceptions across the API boundary and leaving `*this` and output arguments unchanged on failure;
   - `GaussianLikelihood` noise variance handling and `0.x` pre-1.0 API stability expectations.
5. **Out-of-scope claims**: `README.md`, `README.ja.md`, or public headers claim support for out-of-scope models (`Sgpr`, `Svgp`, `OnlineGpr`) or unimplemented features as if shipped.
6. **Bilingual README drift or unnatural Japanese**: `README.md` and `README.ja.md` diverge in section structure, supported scope, build steps, or code examples; relative links are broken; unverified benchmark/accuracy claims appear without source attribution; or `README.ja.md` fails the `natural-japanese` skill criteria.
7. **License and notice mismatch**: `LICENSE` is not Apache-2.0, `THIRD_PARTY_NOTICES.md` disagrees with the pinned versions or licenses in `vcpkg.json`, or `EIGEN_MPL2_ONLY` enforcement is omitted.

## Report

For each failure, state the file path, the offending text, the violated rule, and why an external C++ user is misled or blocked. Fix all findings before tagging or packaging a release.
