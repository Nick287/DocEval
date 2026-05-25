# doceval — 调用方式与参数手册

本文档详细说明 `doceval` 的安装、配置、命令行参数与产物结构。
快速入门请看仓库根目录的 [README.md](../README.md)。

---

## 1. 安装

```bash
# 在仓库根目录
pip install -e .[dev]
```

环境要求：

- Python ≥ 3.11
- 依赖见 [pyproject.toml](../pyproject.toml)，关键项：
  - `agent-framework` ≥ 1.5（Microsoft Agent Framework，封装 Azure OpenAI Responses API）
  - `azure-ai-documentintelligence`
  - `azure-identity`（用于 `DefaultAzureCredential`）
  - `typer`、`pydantic-settings`、`Pillow`

---

## 2. Azure 认证

视觉验证 Agent 使用 **AAD（DefaultAzureCredential）** 而非 API key — 这是 Agent Framework 推荐的方式。

最简单的本地方式：

```bash
az login
az account set --subscription <your-sub-id>
```

也可设置任意一种 `DefaultAzureCredential` 支持的方式（环境变量、Managed Identity、VS Code 登录、Azure CLI 等）。

> ⚠️ **不要** 把 API key 写进配置 — 这个项目刻意没有 `api_key` 通道，统一走 AAD。

Document Intelligence 仍使用 key，写在 `.env` 的 `DOCEVAL_DI_KEY` 中。

---

## 3. 环境变量 / `.env`

所有配置变量都带前缀 `DOCEVAL_`，由 [config.py](../src/doceval/config.py) 中的 `Settings` 类加载（基于 `pydantic-settings`）。

复制模板：

```bash
cp .env.example .env
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DOCEVAL_AZURE_OPENAI_ENDPOINT` | `https://bowan-mk974k0g-swedencentral.services.ai.azure.com/` | Azure OpenAI 资源端点 |
| `DOCEVAL_AZURE_OPENAI_DEPLOYMENT` | `gpt-5.4` | **部署名（deployment alias）**，不是模型名 |
| `DOCEVAL_AZURE_OPENAI_API_VERSION` | `2025-04-01-preview` | Responses API 版本；当前 Agent Framework 默认会自动用 `preview` |
| `DOCEVAL_DI_ENDPOINT` | `https://bofoundry.cognitiveservices.azure.com/` | Document Intelligence 端点 |
| `DOCEVAL_DI_KEY` | _空_ | Document Intelligence 密钥（必填） |
| `DOCEVAL_DATA_ROOT` | 当前工作目录 | 数据根目录；`image/`、`MD/`、`output/`、`.cache/` 均相对于此 |
| `DOCEVAL_OUTPUT_DIR` | `output` | 产物输出目录（相对 `DATA_ROOT`） |
| `DOCEVAL_CLUSTER_EDIT_DISTANCE` | `1` | 编辑距离 ≤ 该值的 token 会归到同一簇 |
| `DOCEVAL_MIN_TOKEN_LENGTH` | `3` | 归一化后短于该长度的 token 丢弃 |
| `DOCEVAL_VERIFY_SINGLETONS` | `true` | 是否把孤立簇送给视觉 Agent 仲裁 |

---

## 4. 数据布局

`doceval` 严格按文件名 stem 匹配输入。**任何来源缺失对应 stem 都会被跳过。**

```
<DATA_ROOT>/
├── image/source/                # 原图
│   ├── 11_mosaic.jpg
│   ├── BOL1.png
│   └── ...
├── MD/                           # 各 MD 来源（子目录名即来源名）
│   ├── gemini/
│   │   ├── 11_mosaic.md
│   │   └── ...
│   ├── gpt/
│   │   ├── 11_mosaic.md
│   │   └── ...
│   └── gpt5_rot/                 # 想加新来源就再开一个目录即可
│       └── 11_mosaic.md
├── .cache/ocr/                   # Document Intelligence 结果自动缓存
└── output/                       # 默认产物目录（每次运行覆盖）
```

新增 MD 来源：在 `MD/` 下创建同名子目录，放入 `.md` 文件即可被自动发现，无需改代码。

---

## 5. 命令行参考

入口由 [pyproject.toml](../pyproject.toml) 注册为 `doceval`，对应 [`src/doceval/cli.py`](../src/doceval/cli.py)。

### 5.1 `doceval run`

跑一次完整的共识评估（OCR + 所有 MD 来源 + 视觉验证）。

```
doceval run [OPTIONS]
```

| 参数 | 简写 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| `--stem` | `-s` | `str`，可重复 | _空_ | 只跑指定的 stem，可重复传：`-s 11_mosaic -s BOL1`。省略时跑所有来源共有的 stem |
| `--source` | — | `str`，可重复 | _空_ | 限定 MD 来源子目录名，例如 `--source gemini --source gpt`。省略时自动发现 `MD/` 下所有子目录 |
| `--no-verify` | — | `bool` | `false` | 跳过视觉验证 Agent。**节省 token、更快**，但孤立簇得不到仲裁，只能保守标为 `ambiguous` |
| `--concurrency` | `-c` | `int` 1-8 | `1` | 同时处理多少张图。每张图内部仍是顺序 LLM 调用 |
| `--log-level` | — | `str` | `INFO` | `DEBUG / INFO / WARNING / ERROR` |

#### 典型调用

```bash
# 烟测：只跑一张
doceval run -s 11_mosaic

# 节省成本：关闭 LLM 验证
doceval run --no-verify

# 只比较 gemini 与 gpt（忽略其它来源）
doceval run --source gemini --source gpt

# 并发跑 4 张
doceval run -c 4

# 多 stem + 调试日志
doceval run -s 11_mosaic -s BOL1 --log-level DEBUG
```

#### 退出码

- `0` — 成功
- `1` — 找不到任何跨来源共有的 stem

### 5.2 `doceval ocr <stem>`

仅运行 Document Intelligence layout，把识别到的 token（规范化结果 + 原 surface + bbox）打印出来。**不调用 LLM**，用于排查 OCR 自身的读字错误。

```bash
doceval ocr 11_mosaic
```

---

## 6. 产物结构

每次 `run` 会写入：

```
output/
├── summary.md                 # 跨图汇总（markdown 表格 + 累计召回/准确）
├── summary.csv                # 同上的 CSV（每个 source × 每种判定的计数）
└── <stem>/
    ├── report.md              # 该图逐簇详细判定
    ├── clusters.json          # 全量簇 + 投票 + Agent 证据（可复现）
    └── annotated_<source>.jpg # 在原图上把该来源的错误用色框标出
```

### 6.1 `summary.md`

顶部新增的 **「评估配置」** 段会显示本次运行实际命中的视觉验证模型版本（Azure 通过 `x-ms-served-model` 响应头返回，例如 `gpt-5.4-2025-09-xx`），方便复现与审计：

```
## 评估配置

- 视觉验证模型：`gpt-5.4-2025-09-xx`
```

下方表格列：

- `stem` / `clusters` / `elapsed_s` — 基本信息
- `<source>_命中/看错/漏读/幻觉/不明确` — 每来源 × 每判定的计数
- **「各来源累计指标」** 表：
  - **召回率** = 命中 ÷ (命中 + 漏读 + 看错)
  - **准确率** = 命中 ÷ (命中 + 看错 + 幻觉 + 不明确)

### 6.2 `<stem>/report.md`

标题下会标注 **「视觉验证模型：`gpt-5.4-2025-09-xx`」**（与 `summary.md` 同源）。

### 6.3 `<stem>/clusters.json`

机读版完整快照，新增顶层字段：

```json
{
  "stem": "11_mosaic",
  "verifier_model": "gpt-5.4-2025-09-xx",
  "stats": {...},
  "clusters": [...],
  "judgements": [...]
}
```

判定取值：

| `verdict` | 含义 |
|---|---|
| `correct` | 该来源在该簇中写对了 canonical |
| `typo` | 写了但与 canonical 编辑距离 1（读错字符） |
| `omission` | 簇存在，但该来源没写 |
| `hallucination` | 该来源写了 canonical 不认可的 token（视觉验证后被判 `absent`） |
| `ambiguous` | 视觉验证无法定论，或验证被关闭 |

---

## 7. 工作流速查

| 目标 | 命令 |
|---|---|
| 第一次跑通环境 | `doceval run -s 11_mosaic --log-level DEBUG` |
| 全量评估 | `doceval run -c 4` |
| 只看某来源差异 | `doceval run --source gemini --source gpt` |
| 离线快速回归 | `doceval run --no-verify` |
| 排查 OCR 错读 | `doceval ocr <stem>` |
| 复现某次结果 | 检查 `clusters.json` 的 `verifier_model` 字段，固定到同一部署 + 同一服务端版本 |

---

## 8. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `Unable to handle auth` | 没有 `az login`，或当前账号无 Cognitive Services User 权限。本项目**不**支持 API key，必须 AAD |
| `API version ... is not supported` | `.env` 里的 `AZURE_OPENAI_API_VERSION` 太旧。Agent Framework 默认会用 `preview`，无须再手动覆盖 |
| `no stems shared across all sources` | `MD/<source>/` 与 `image/source/` 的 stem 没有交集。检查文件名（不带扩展名）是否一致 |
| `clusters.json` 里 `verifier_model` 为 `null` | 用了 `--no-verify`，或本次没有任何孤立簇需要送验 |
| OCR 反复请求 Document Intelligence | 检查 `.cache/ocr/<stem>.json` 是否被清空 — 缓存命中即跳过 API |

---

## 9. 编程式调用

如果不想走 CLI，也可以直接调度内部 API：

```python
import asyncio
from doceval.pipeline import build_default_evaluator, evaluate_many
from doceval.reporting import write_clusters_json, write_report, write_summary
from doceval.config import get_settings

settings = get_settings()
evaluator = build_default_evaluator(enable_verifier=True)

async def main() -> None:
    evals = await evaluate_many(evaluator, ["11_mosaic", "BOL1"], concurrency=2)
    out = settings.out_root
    for ev in evals:
        d = out / ev.stem
        d.mkdir(parents=True, exist_ok=True)
        write_clusters_json(ev, d / "clusters.json")
        write_report(ev, d / "report.md")
        print(ev.stem, ev.verifier_model)  # 实际服务模型版本
    write_summary(evals, out, sources=evaluator.all_sources)

asyncio.run(main())
```
