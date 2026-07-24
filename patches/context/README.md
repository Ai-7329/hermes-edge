# コンテキスト管理パッチ集（デフォルト Hermes 向け）

hermes-edge フォークのうち **コンテキスト管理に関する変更だけ** を、素の
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
に外部から適用できる形で切り出したパッチ集。

fork 全体の他の機構（single-flight ゲート、タイムアウト境界、タイトル生成の
off スイッチ、deny_timeout など）は含まない。ベースは upstream コミット
`b9b463f`（fork 分岐点）で、各パッチが `git apply` で衝突なく当たることと、
適用後にテストが通ること（23 passed）を検証済み。

## 構成

| # | パッチ | 中身 | 依存 |
|---|---|---|---|
| 1 | `0001-mechanical-context-engine.patch` | 機械式ダイジェストエンジン（LLM 不要のコンパクション）。`plugins/context_engine/mechanical/` + テスト | **なし** — upstream のプラグイン機構（`agent/context_engine.py`）だけで動く純粋な追加ファイル |
| 2 | `0002-compression-budget-tokens.patch` | `compression.budget_tokens` ワーキングセット上限。`agent/context_compressor.py` のみ変更 | なし（1 と独立。ただし 1 と併用時はエンジンが継承で自動的に budget を尊重） |
| 3 | `0003-compression-warmup-optional.patch` | コンパクション直後のアイドル時間に圧縮後プロンプトをローカルサーバへ再プリフィルする warmup。`run_agent.py` + `agent/local_runtime.py`（新規モジュール）+ テスト | **オプション**。fork の `local_runtime` モジュールを同梱する（EndpointGate / prefill_tps を warmup が使うため）。設定しなければ完全に不活性 |

適用順は 1 → 2 → 3。1 だけ、1+2 だけでも成立する。
3 は「ローカル llama.cpp 系バックエンドで、コンパクション後の数分単位の
再プリフィルをユーザーの次ターンから隠したい」場合のみ。

補足:
- パッチ 3 の `agent/local_runtime.py` は fork ではリクエスト直列化ゲート全体を
  担うモジュールで、warmup が使わない関数も含まれるが、すべて設定なしでは
  no-op。upstream の他のコードからは参照されないため同梱しても挙動は変わらない。
- テストファイル `tests/agent/test_compression_budget_warmup.py`
  （budget の閾値計算テストを含む）は `agent/local_runtime` を import するため
  パッチ 3 に同梱している。1+2 のみの適用でも本体コードは完全に機能する。

## 適用方法

```bash
cd hermes-agent   # upstream のチェックアウト
git apply path/to/0001-mechanical-context-engine.patch
git apply path/to/0002-compression-budget-tokens.patch
git apply path/to/0003-compression-warmup-optional.patch   # 任意
```

pip / installer 導入版に当てる場合はインストール先の site-packages ではなく
ソースチェックアウトに当てて実行すること（`plugins/` と `agent/` の相対配置が
前提）。パッチ 1 はファイル追加だけなので、`git apply` の代わりに
`plugins/context_engine/mechanical/` ディレクトリをそのままコピーしてもよい。

## 設定（config.yaml）

```yaml
# パッチ1: LLM を呼ばない決定的コンパクション
context:
  engine: mechanical

# パッチ2: コンパクション政策上のワーキングセット上限（0/未設定 = 無効）
# トリガー = min(budget, context_length - max_tokens) × threshold_percent
compression:
  budget_tokens: 32768

# パッチ3: コンパクション後のアイドル時再プリフィル（既定 off）
local_runtime:
  compression_warmup: true
  prefill_tps: 50        # 実測プリフィル速度 — warmup タイムアウト下限に使用
```

いずれも opt-in。設定しなければ upstream と同一挙動（クラウド Provider には
warmup は自動判定で常に不活性）。

## 各パッチが解く問題

1. **mechanical engine** — 低速なシングルスロットのローカルバックエンドでは、
   LLM 要約によるコンパクションが「コンテキストが最も膨らんだ瞬間に
   全ウィンドウのプリフィルを要求する」ため構造的にタイムアウトし、
   compress → timeout → cooldown → compress のライブロックに入る
   （実測: 66 回試行 / 0 成功 / 9h54m）。機械式エンジンはメモリ上の
   ウィンドウの純関数としてダイジェストを決定的に生成する — ネットワーク
   なし、タイムアウトなし。予算比例のセクション配分で小予算でも
   ツール索引・現在位置が欠落しない（無言の切り捨てをしない契約）。

2. **budget_tokens** — 131K ウィンドウを既定トリガー（~92K）まで使うと、
   コンパクション後の再プリフィルが ~50 tok/s では数十分かかる。明示予算で
   トリガーを早め、MINIMUM_CONTEXT_LENGTH 床と small-context 引き上げを
   バイパスする（既定値保護の仕組みであり、明示予算は operator の意思のため）。
   `context_length` 自体は他の用途に対して正直なまま。

3. **warmup** — コンパクションは履歴を書き換えるので、次の実呼び出しが
   書き換え後プロンプト全体を再プリフィルする。warmup はターン間の
   アイドル時間に 1 トークン要求で同一プロンプトを送り、サーバの
   プレフィックスキャッシュを先に再構築する。実リクエストとの競合は
   skip-when-busy で常に実リクエストが勝つ。

## 再生成方法

このリポジトリで:

```bash
git diff b9b463f HEAD -- plugins/context_engine/mechanical tests/plugins/context_engine \
  > 0001-mechanical-context-engine.patch
git diff b9b463f HEAD -- agent/context_compressor.py \
  > 0002-compression-budget-tokens.patch
# 0003 は run_agent.py の warmup ハンク（compress_context ラッパ以降）のみを
# 抽出し、agent/local_runtime.py と warmup テストの diff を連結したもの。
# run_agent.py のもう一方のハンク（stale タイムアウト境界）は local_runtime
# 機能でありコンテキスト管理ではないため含めていない。
```
