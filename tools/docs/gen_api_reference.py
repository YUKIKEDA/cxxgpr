# SPDX-License-Identifier: Apache-2.0
"""Generate the Breathe-backed API reference pages (one page per module, per language).

Every public header under include/cxxgpr/ gets exactly one ``doxygenfile`` directive, so the
reference is complete by construction. ``tools/qa/doc_api_coverage.py`` verifies that.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "include" / "cxxgpr"

# Module -> (English lead, Japanese lead).
MODULES: dict[str, tuple[str, str]] = {
    "core": (
        "Errors, views, intervals, likelihood, checked arithmetic, random numbers, build info.",
        "エラー型、行列ビュー、区間型、尤度、検査付き算術、乱数、ビルド情報。",
    ),
    "precision": (
        "Precision policies (`DoublePrecision`, `SinglePrecision`, `MixedPrecision`), refinement.",
        "精度ポリシー（`DoublePrecision`・`SinglePrecision`・`MixedPrecision`）と反復改良エンジン。",
    ),
    "linalg": (
        "Cholesky factorisation, triangular solves, jitter, deterministic parallel helpers.",
        "Cholesky 分解、三角求解、Jitter ポリシー、決定的な並列補助。",
    ),
    "transform": (
        "Input and target transforms, pipelines and column-wise composition.",
        "入力・目的変数の変換、パイプライン、列ごとの合成。",
    ),
    "kernel": (
        "Built-in kernels, composition, custom kernels, supplied distances, exponent backends.",
        "組込みカーネル、合成、独自カーネル、供給二乗距離、指数評価バックエンド。",
    ),
    "workspace": (
        "Pre-allocated buffers, leaf caches, marginal-likelihood and inference kernels.",
        "事前確保バッファ、葉キャッシュ、周辺尤度と推論の計算核。",
    ),
    "optimizer": (
        "Objective interfaces, bounds and the five optimizers.",
        "目的関数インターフェース、境界、5 種の最適化器。",
    ),
    "gpr": (
        "`Gpr` (trainer) and `FittedGpr` (trained model).",
        "`Gpr`（学習器）と `FittedGpr`（学習済みモデル）。",
    ),
    "persist": (
        "Saving and loading models (`config.json` + `model.safetensors`).",
        "モデルの保存と読込（`config.json` + `model.safetensors`）。",
    ),
}

# Umbrella headers (``kernel.hpp`` etc.) carry no declarations of their own.
UMBRELLA = {"kernel.hpp", "optimizer.hpp", "transform.hpp", "workspace.hpp"}

INTRO = {
    "en": (
        "# API reference\n\n"
        "Generated from the Doxygen comments of the public headers in `include/cxxgpr/`. "
        "Every public declaration appears here. For task-oriented usage, see the "
        "[how-to guides](../../how-to/index.md).\n"
    ),
    "ja": (
        "# API リファレンス\n\n"
        "`include/cxxgpr/` の公開ヘッダにある Doxygen コメントから生成しています。"
        "公開宣言はすべてここに載ります。目的別の使い方は"
        "[ハウツーガイド](../../how-to/index.md)を参照してください。"
        "なお、宣言の説明文は英語のままです。\n"
    ),
}
UMBRELLA_LABEL = {"en": "Umbrella header: ", "ja": "まとめて取り込むヘッダ: "}


def render(module: str, lang: str) -> str:
    """Render one module page."""
    lead = MODULES[module][0 if lang == "en" else 1]
    names = sorted(p.name for p in (INCLUDE / module).glob("*.hpp"))
    headers = [n for n in names if n not in UMBRELLA]
    umbrella = [n for n in names if n in UMBRELLA]
    out = [f"# `cxxgpr/{module}`", "", lead, ""]
    if umbrella:
        listed = ", ".join(f"`cxxgpr/{module}/{n}`" for n in umbrella)
        out += [f"{UMBRELLA_LABEL[lang]}{listed}.", ""]
    for name in headers:
        out += [
            f"## `cxxgpr/{module}/{name}`",
            "",
            f"```{{doxygenfile}} {name}",
            ":project: cxxgpr",
            "```",
            "",
        ]
    return "\n".join(out)


def main() -> None:
    """Write the module pages and the index for both languages."""
    for lang in ("en", "ja"):
        target = ROOT / "docs" / lang / "reference" / "api"
        target.mkdir(parents=True, exist_ok=True)
        for module in MODULES:
            (target / f"{module}.md").write_text(
                render(module, lang), encoding="utf-8", newline="\n"
            )
        toc = "\n".join(MODULES)
        index = INTRO[lang] + "\n```{toctree}\n:maxdepth: 1\n\n" + toc + "\n```\n"
        (target / "index.md").write_text(index, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
