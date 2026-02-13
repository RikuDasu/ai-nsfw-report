# AI生成NSFWコンテンツ：技術パイプライン・エコシステム調査レポート

## 1. 技術基盤：画像生成モデルの系譜

### 1.1 Stable Diffusion エコシステム

| モデル | リリース | 解像度 | NSFW対応 | 特徴 |
|--------|----------|--------|----------|------|
| SD 1.5 | 2022/10 | 512x512 | 最も成熟 | LoRA/チェックポイント最多、NSFW生態系の中核 |
| SDXL | 2023/07 | 1024x1024 | 充実 | 高解像度、リファイナーモデル搭載 |
| SD3/3.5 | 2024/06 | 可変 | 制限付き | MMDiTアーキテクチャ、Stability AIがNSFW禁止 |
| Flux.1 | 2024/08 | 可変 | 基本は制限 | Black Forest Labs開発、コミュニティLoRAで解除 |

### 1.2 なぜ「無修正」が可能か

**根本原理**: オープンソースモデルはローカル実行可能で、安全フィルターの除去が技術的に容易。

1. **学習データに制限なし**: SD 1.5はLAION-5Bデータセット（50億画像）から学習。NSFWフィルタリングが不完全で、無修正画像を含む
2. **Safety Checker除去**: AUTOMATIC1111等のローカルUIではSafety Checker（NSFW検出フィルター）をデフォルトで無効化可能
3. **専用チェックポイント**: 無修正データセットでファインチューニングされた専用モデルが多数存在
4. **LoRAによる制限解除**: ベースモデルが制限されていても、LoRA重みで挙動を変更可能

### 1.3 主要フォトリアルチェックポイントモデル

| モデル名 | ベース | DL数 | 特徴 |
|----------|--------|------|------|
| RealVisXL V4.0 | SDXL | 300K+ | フォトリアル特化、NSFW対応 |
| Juggernaut XL | SDXL | 520K+ | 汎用フォトリアル、高品質 |
| epiCRealism XL | SDXL | 200K+ | 超リアル人物生成 |
| Realistic Vision V5.1 | SD1.5 | 400K+ | SD1.5最高峰のリアル系 |
| CyberRealistic | SD1.5 | 350K+ | 写実性とスタイルのバランス |

## 2. 技術パイプライン

### 2.1 生成フロー

```
[テキストプロンプト] → [チェックポイント + LoRA] → [Txt2Img生成]
                                                        ↓
                                              [Img2Img リファイン]
                                                        ↓
                                              [Inpainting 部分修正]
                                                        ↓
                                              [Upscale 高解像度化]
                                                        ↓
                                              [最終出力]
```

### 2.2 UI/ツール

#### AUTOMATIC1111 WebUI
- **リポジトリ**: github.com/AUTOMATIC1111/stable-diffusion-webui
- **特徴**: 最も普及したSD向けUI、拡張機能エコシステム
- **NSFW関連**: Safety Checkerデフォルト無効、negative promptでの制御
- **推奨設定**:
  - Sampler: DPM++ 2M Karras / Euler a
  - CFG Scale: 5-9（リアル系は低め推奨）
  - Steps: 20-40
  - CLIP Skip: 2（アニメ系）/ 1（リアル系）

#### ComfyUI
- **リポジトリ**: github.com/comfyanonymous/ComfyUI
- **特徴**: ノードベースのパイプライン、細かい制御が可能
- **利点**: カスタムワークフロー構築、複雑なパイプライン
- **NSFW利点**: ControlNet + Inpainting + LoRAの複合ワークフローが視覚的に構築可能

### 2.3 Inpainting（部分修正）技術

**仕組み**: マスク領域にのみノイズを付加し、テキストプロンプトに基づいて再生成

- **Denoising Strength**: 0.4-0.7が推奨（低すぎると変化なし、高すぎると不整合）
- **マスク**: 白=再生成、黒=保持
- **用途**: 衣服の除去、特定部位の詳細化、表情変更
- **専用モデル**: `stable-diffusion-v1-5/stable-diffusion-inpainting`（HuggingFace）

### 2.4 ControlNet

- **機能**: ポーズ・構図・深度を制御
- **OpenPose**: 人体ポーズの指定
- **Depth**: 奥行き情報の制御
- **Canny/Lineart**: 輪郭の維持
- **NSFW用途**: 特定のポーズや構図を正確に再現

## 3. LoRA（Low-Rank Adaptation）技術

### 3.1 技術的仕組み

- **原理**: 大規模モデルの全パラメータを再学習せず、低ランク行列の追加学習で挙動変更
- **パラメータ数**: 全体の0.1-1%程度（数MB〜数十MB）
- **メリット**: 少ないVRAM・学習時間、複数LoRA同時適用可能

### 3.2 Kohya_ss での学習設定

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| Learning Rate | 1e-4 (UNet), 1e-5 (TE) | テキストエンコーダーは低めに |
| Epochs | 10-30 | データセットサイズに依存 |
| Batch Size | 1-4 | VRAM依存（12GB: 1-2） |
| Network Rank | 32-128 | 高いほど表現力↑、サイズ↑ |
| Network Alpha | 16-64 | Rank以下推奨 |
| Optimizer | AdamW8bit / Prodigy | メモリ効率重視 |
| Resolution | 512 (SD1.5) / 1024 (SDXL) | モデルに合わせる |

### 3.3 学習データセット構成

- **必要枚数**: 最低50-100枚（多様性重視）
- **タグ付け**: WD14 Tagger / BLIP-2で自動キャプション
- **正則化画像**: 過学習防止のためクラス画像を用意
- **ディレクトリ構成**: `[繰り返し数]_[概念名]` 形式

### 3.4 Uncensored LoRA の実例

| LoRA名 | プラットフォーム | ベース | 説明 |
|--------|----------------|--------|------|
| Flux-uncensored | HuggingFace (enhanceaiteam) | Flux.1-dev | Fluxの制限解除LoRA |
| Flux-Uncensored-V2 | HuggingFace (Ryouko65777) | Flux.1-dev | V2改良版 |
| Flux Lustly.ai Uncensored | CivitAI | Flux | 男女ヌード対応 |
| Flux NSFW LoRA v2 | CivitAI | Flux | NSFW特化 |

## 4. モザイク除去AI

### 4.1 主要ツール

| ツール | 技術 | リポジトリ/ソース | 特徴 |
|--------|------|------------------|------|
| DeepMosaics | GAN (Pix2Pix系) | GitHub: HypoX64/DeepMosaics | 動画対応、自動検出 |
| hent-ai | Mask R-CNN + GAN | GitHub: natethegreate/hent-ai | 二次元特化、検閲バー除去 |
| JavPlayer | TecoGAN | 商用ソフト | 動画特化、リアルタイム処理 |
| Lama Cleaner | LaMa (Large Mask inpainting) | GitHub: Sanster/lama-cleaner | 汎用修復、モザイク除去にも |

### 4.2 技術的仕組み

1. **検出フェーズ**: モザイク/検閲領域を自動検出（物体検出モデル）
2. **セグメンテーション**: 検閲領域のマスク生成
3. **復元フェーズ**: GANによる元画像の推定・再生成
4. **注意**: 「復元」ではなく「推定生成」— 元の画像を完全に復元するものではない

## 5. AI動画生成

### 5.1 AnimateDiff

- **仕組み**: SD1.5の画像生成プロセスにモーションモジュールを追加
- **対応**: SD1.5モデルのみ（SDXL非対応）
- **フレーム**: 16-32フレームのショートアニメーション
- **統合**: AUTOMATIC1111拡張機能として利用可能
- **制限**: SD1.5チェックポイント使用のためNSFWモデルと互換

### 5.2 Stable Video Diffusion (SVD)

- **開発**: Stability AI
- **入力**: 静止画からの動画生成（img2vid）
- **フレーム**: 14-25フレーム、3-30fps
- **制限**: Stability AIポリシーによりNSFW制限あり

### 5.3 その他のAI動画ツール

- **Wan2.1**: オープンソース動画生成
- **CogVideoX**: テキストから動画
- **商用SaaS**: 複数のNSFW対応AI動画生成サービスが2025年に登場

## 6. プラットフォーム・エコシステム

### 6.1 CivitAI

- **役割**: AI画像生成モデル最大のハブサイト
- **コンテンツ**: チェックポイント、LoRA、VAE、Embedding、ワークフロー共有
- **NSFW政策変遷**:
  - 初期: NSFW比較的寛容
  - 2024: SD3系モデルでのNSFW禁止（Stability AI要請）
  - 2025: コンテンツ政策強化、一部NSFWモデルが制限/削除
- **収益化**: 「Buzz」仮想通貨 → クレジットカード決済停止後、暗号通貨に移行
- **オンライン生成**: 一部モデルはCivitAI上で直接生成可能

### 6.2 HuggingFace

- **役割**: 機械学習モデルのGitHub的存在
- **NSFW**: モデル自体のホスティングは許容（コンテンツポリシーは緩め）
- **主要NSFW資産**: Flux-uncensored、各種LoRA

### 6.3 代替プラットフォーム

- **Tensor.Art**: CivitAI代替のモデル共有サイト
- **Mage.Space**: オンラインNSFW生成対応
- **各種Discord**: モデル共有・ワークフロー共有コミュニティ

## 7. 収益化チャネル

| チャネル | NSFW対応 | 手数料 | 備考 |
|----------|----------|--------|------|
| Patreon | 限定的許容 | 5-12% | 2024年時点で58%がNSFWクリエイター |
| SubscribeStar | 許容 | 5% | Patreon代替として成長 |
| CivitAI Buzz | モデル配布 | 可変 | CC停止→暗号通貨 |
| Gumroad | 禁止 | - | 2024年にNSFWコンテンツ禁止 |
| Ko-fi | 制限的 | 5% | 明示的NSFW非推奨 |
| 暗号通貨直接 | 制限なし | ガス代のみ | 追跡困難、完全匿名可能 |

## 8. 法的フレームワーク

### 8.1 日本（刑法175条）

- **条文**: わいせつな文書、図画、電磁的記録の頒布、公然陳列を禁止
- **AI適用**: AI生成物であっても「わいせつ物」に該当すれば処罰対象
- **2024年改正**: 電磁的記録（デジタルデータ）を明文化

#### 逮捕事例（2025年）

| 時期 | 事案 | 容疑 | 備考 |
|------|------|------|------|
| 2025/04 | AI生成わいせつポスター販売 | わいせつ図画頒布 | 全国初の逮捕、売上約1000万円 |
| 2025/04 | 芸能人似わいせつ画像SNS掲載 | わいせつ図画公然陳列 | 約2万点、閲覧料120万円 |
| 2025/06 | AI生成抱き枕カバー販売 | わいせつ物頒布 | AI生成物への同容疑適用初 |

### 8.2 米国

- **TAKE IT DOWN Act** (2025/05 署名): 非同意親密画像（NCII）の配布を犯罪化、AI生成含む、最大2年の禁固刑
- **プラットフォーム義務**: 48時間以内の削除義務

### 8.3 Deepfake規制

- **問題**: 「Undress AI」アプリの氾濫 — 実在人物の画像から性的画像を生成
- **技術**: Stable Diffusion Inpaintingベースの衣服除去
- **デンマーク**: 肖像を知的財産として保護する著作権法改正
- **規模**: Telegram上だけで数百のDeepfakeボットが稼働

## 9. 市場規模・統計

### 9.1 AI x アダルト市場

- **AI活用率予測**: 2025年までに全アダルトコンテンツ制作の80%がAI活用
- **コスト削減**: AIにより制作コストが25%削減
- **デジタルアダルトコンテンツ市場**: 2024年 USD 455億 → 2032年 USD 788億（CAGR 7.1%）
- **AI NSFW対話プラットフォーム**: 44.4%のAIポルノサイトがAIエージェントとの対話機能を提供

### 9.2 CivitAIエコシステム

- **モデル数**: 数万のチェックポイント・LoRA
- **ユーザー**: 数百万のアクティブユーザー
- **2025年変化**: NSFW政策強化、一部クリエイターが代替プラットフォームへ移行

## 10. 具体的導入フロー（技術的ワークフロー）

### Step 1: 環境構築
```
1. Python 3.10.6 インストール
2. Git インストール
3. AUTOMATIC1111 WebUI クローン:
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
4. webui-user.bat 実行（自動セットアップ）
5. VRAM要件: 最低6GB（推奨12GB以上）
```

### Step 2: モデル配置
```
models/Stable-diffusion/ → チェックポイント (.safetensors)
models/Lora/ → LoRAファイル (.safetensors)
models/VAE/ → VAEファイル
```

### Step 3: 生成ワークフロー
```
1. チェックポイント選択（例: RealVisXL V4.0）
2. LoRA追加: プロンプト内に <lora:名前:強度>
3. プロンプト記述（品質タグ + コンテンツ指示）
4. Negative prompt設定（不要要素の除外）
5. パラメータ調整（Steps, CFG, Sampler, Size）
6. Generate → Img2Img → Inpainting → Upscale
```

### Step 4: LoRA学習（カスタムモデル作成）
```
1. Kohya_ss GUI インストール
2. 学習画像収集（50-100枚以上）
3. WD14 Taggerで自動タグ付け
4. 正則化画像準備
5. 学習設定（LR, Epochs, Rank等）
6. 学習実行（RTX 3090で数時間）
7. 出力LoRAをmodels/Lora/に配置
```
