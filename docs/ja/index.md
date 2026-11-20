# cxxgpr

**C++23 のための厳密なガウス過程回帰ライブラリ。** 値を返し例外を投げない API、10 種の組込みカーネル、
自己完結の 5 種の最適化器、倍精度・単精度・混合精度のソルバ、可搬なモデル保存形式を備えた静的ライブラリです。

```{code-block} cpp
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
// 学習し、予測平均と不確かさを得る。エラーは例外ではなく値で返る。
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} チュートリアル
:link: tutorials/index
:link-type: doc

手を動かして学びます。最初のモデルを学習し、多次元の実用的な流れへ広げます。
:::

:::{grid-item-card} ハウツーガイド
:link: how-to/index
:link-type: doc

目的別の手順です。カーネルの構築、データ変換、最適化器の選択、モデル保存、性能調整。
:::

:::{grid-item-card} リファレンス
:link: reference/index
:link-type: doc

調べるための資料です。全公開クラス・関数・列挙子、エラーコード、ビルドオプション、ファイル形式。
:::

:::{grid-item-card} 解説
:link: explanation/index
:link-type: doc

理由を理解します。数理、数値上の選択、設計判断を説明します。
:::
::::

## どこから読むか

- ガウス過程が初めてなら、{doc}`explanation/gaussian-processes` を読み、{doc}`tutorials/first-regression` に進んでください。
- 理論を知っているなら、{doc}`how-to/install-and-link` と {doc}`how-to/index` へ進んでください。

```{toctree}
:hidden:
:maxdepth: 2

tutorials/index
how-to/index
reference/index
explanation/index
```
