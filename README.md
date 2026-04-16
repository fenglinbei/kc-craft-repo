# KG-CRAFT 复现脚本（OpenAI 兼容 API 版本）

这是一个**模块化、配置驱动**的 KG-CRAFT 复现实现，目标是尽量忠实复现论文 *KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Enhancing Automated Fact-checking* 的主流程：

1. 从 claim 和 reports 中抽取知识图谱（KG）
2. 基于 KG 生成对比问题（contrastive questions）
3. 基于 reports 回答这些问题
4. 将问答对压缩成证据摘要
5. 基于摘要判断 claim 的 veracity label

同时支持以下要求：

- **KG 抽取阶段**和**后续推理阶段**分别使用两套独立的 OpenAI 兼容 API 配置
- API 参数均可通过 `config yaml` 控制：`api_key`、`api_base`、`model`
- Chat API 统一使用 **OpenAI 兼容 `/chat/completions`** 格式
- embedding 固定为本地模型 `./models/bge-small-zh-v1.5`
- 复现流程尽量**模块化**，且中间产物全部落盘，便于调试与复现实验
- 支持四种运行模式：
  - `full`：完整 KG-CRAFT
  - `naive_llm`：只用 claim + reports 做直接判别
  - `kg_only`：只做 KG 增强，不做 contrastive reasoning
  - `llm_questions`：不用 KG 生成问题，而是让 LLM 直接生成对比问题（对应论文的 ablation）

---

## 1. 目录结构

```text
kg_craft_repro/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── liar_raw.yaml
│   └── rawfc.yaml
├── data/
│   └── example_input.jsonl
├── scripts/
│   ├── run_pipeline.py
│   ├── evaluate.py
│   └── convert_raw_datasets.py
└── src/kg_craft/
    ├── __init__.py
    ├── api.py
    ├── config.py
    ├── contrastive.py
    ├── data.py
    ├── embeddings.py
    ├── evaluation.py
    ├── kg_extraction.py
    ├── pipeline.py
    ├── prompts.py
    ├── schemas.py
    ├── utils.py
    └── verification.py
```

---

## 2. 输入数据格式

默认输入为 JSONL，每行一个样本，推荐格式如下：

```json
{"id": "sample-1", "claim": "...", "reports": ["report text 1", "report text 2"], "label": "false"}
```

字段名也可以在配置里改：

- `data.id_field`
- `data.claim_field`
- `data.reports_field`
- `data.label_field`

其中：

- `claim`：待验证陈述
- `reports`：与 claim 关联的报告文本列表
- `label`：可选，评估时使用

---

## 3. 安装

建议 Python 3.10+

```bash
conda create -p ../conda/kg-craft-repo python=3.12
conda activate ../conda/kg-craft-repo
pip install -r requirements.txt
```

如果你的环境没有 GPU，也可以在 config 里把 embedding device 设为 `cpu`。

### 3.1 模型下载

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./models/Qwen3-Embedding-0.6B
```

## 4. 配置 OpenAI 兼容 API

`configs/base.yaml` 里已经预留了两套模型配置：

- `models.kg_llm`：用于 KG 抽取
- `models.reasoning_llm`：用于问题回答、摘要、最终判别

你可以直接写死，也可以使用环境变量：

```bash
export KG_API_KEY="..."
export KG_API_BASE="https://your-endpoint/v1"
export KG_MODEL="gpt-4o-mini"

export REASONING_API_KEY="..."
export REASONING_API_BASE="https://your-endpoint/v1"
export REASONING_MODEL="gpt-4o"
```

---

## 5. 运行

### 5.1 完整 KG-CRAFT

```bash
。
```

### 5.2 Naive LLM baseline

```bash
python scripts/run_pipeline.py \
  --config configs/liar_raw.yaml \
  --mode naive_llm \
  --input data/example_input.jsonl \
  --output outputs/example_naive.jsonl
```

### 5.3 KG only ablation

```bash
python scripts/run_pipeline.py \
  --config configs/liar_raw.yaml \
  --mode kg_only \
  --input data/example_input.jsonl \
  --output outputs/example_kg_only.jsonl
```

### 5.4 LLM question ablation

```bash
python scripts/run_pipeline.py \
  --config configs/liar_raw.yaml \
  --mode llm_questions \
  --input data/example_input.jsonl \
  --output outputs/example_llm_questions.jsonl
```

### 5.5 将 LIAR-RAW / RAWFC 转换为本项目输入格式

把原始数据放到以下目录：

- `data/raw/LIAR-RAW/{train,val,test}.json`
- `data/raw/RAWFC/{train,val,test}/*.json`

运行转换脚本（内置日志和进度条）：

```bash
python scripts/convert_raw_datasets.py \
  --dataset both \
  --input-root data/raw \
  --output-dir data/converted
```

如果你希望 RAWFC 使用 `original_label` 作为输出 `label`，可改成：

```bash
python scripts/convert_raw_datasets.py \
  --dataset rawfc \
  --label-field original_label
```

---

## 6. 评估

```bash
python scripts/evaluate.py \
  --predictions outputs/example_full.jsonl \
  --label-field label \
  --pred-field prediction
```

输出包含：

- accuracy
- macro precision / recall / f1
- weighted precision / recall / f1
- 每个标签的 classification report

---

## 7. 中间结果

每个样本都会保存这些中间字段，便于排查：

- `claim_kg`
- `report_kgs`
- `merged_kg`
- `claim_triples`
- `candidate_questions`
- `selected_questions`
- `qa_pairs`
- `contrastive_summary`
- `prediction`
- `raw_outputs`

---

## 8. 与论文逐项对齐

本实现对齐了论文的主流程：

- 论文将方法分为 **KG 抽取 → contrastive reasoning → veracity verification** 三阶段
- contrastive reasoning 内部又分为 **问题生成 → 问题回答 → 答案摘要**
- 问题选择使用 embedding + MMR 排序，默认 `K=5`

论文正文与附录给出了这些核心结构和 prompt 模板。

---

## 9. 需要你最终固定/确认的地方

这几点会影响你是否能做到“和论文几乎一模一样”的复现：

### A. 数据集与切分
论文实验用的是 LIAR-RAW 和 RAWFC，并且使用它们各自原始的 train/val/test 划分。你需要准备：

1. 原始数据文件
2. 与论文一致的 claim-report 关联格式
3. 与论文一致的标签集合

当前脚本已经支持通用 JSONL，但**不会自动下载论文数据集**。

### B. KG 抽取 prompt 的完整细节
论文附录 D.1 中展示的是**缩略版模板**，其中 `Extract nodes [...]`、`Label nodes [...]`、`Extract relationships [...]`、`Compliance criteria: [...]` 仍然保留了占位符。因此这里我提供的是一个**忠实于论文流程、但为了可运行而补全为 JSON 输出格式的 operational prompt**。

### C. LLM 直接生成对比问题的 few-shot 样例
附录 D.5 也保留了：

- `{claim example}`
- `{reports examples}`
- `{contrastive questions examples}`

这些 few-shot 实例并没有在 PDF 中展开给出。因此我在实现里把 few-shot 入口做成了配置项；如果你后面有作者原始样例，可以直接补进 config。

### D. 标签描述（label descriptions）
论文在 veracity verification 部分说 prompt 中包含“labels and their descriptions”，但附录 D.4 实际只明确展示了 `{veracity labels}` 模板，没有把描述全文写出来。所以这里把**标签描述交给 config 管理**。你可以：

- 用简单描述
- 用你已有的 benchmark 官方描述
- 或者后续替换成作者原始文本

### E. OpenAI 兼容服务差异
虽然接口统一用 `/chat/completions`，但不同供应商对这些参数的支持程度可能不一样：

- `response_format`
- `max_tokens`
- `temperature`
- 某些 JSON mode 特性

脚本已经做了容错；如果服务端不支持 JSON mode，可以把 `models.*.response_format` 设为 `null`。

---

## 10. 默认超参数

默认值与论文设置尽量对齐：

- `temperature = 0.0`
- `K = 5`
- MMR 使用 cosine similarity
- embedding 使用本地 `./models/bge-small-zh-v1.5`

---

## 11. 日志与进度条

- `scripts/run_pipeline.py`、`scripts/evaluate.py`、`scripts/convert_raw_datasets.py` 均支持 `--log-level`（`DEBUG/INFO/WARNING/ERROR`）。
- pipeline 样本处理和数据集转换均提供 `tqdm` 进度条，方便观察任务执行状态。

---

## 12. 一个最小可复现实验建议

你可以先这样跑：

1. 用 `data/example_input.jsonl` 跑通单条样本
2. 再接入你的完整 benchmark JSONL
3. 先比较 `naive_llm` 和 `full`
4. 再比较 `kg_only` 和 `llm_questions`

这样比较容易确认到底是哪一步带来了收益。

---

## 13. 说明

本仓库优先保证：

- 接口可替换
- 阶段可拆分
- 中间结果可检查
- 配置可调

如果你后面拿到了作者提供的原始 few-shot 示例、KG prompt 完整版、或数据预处理脚本，只需要替换：

- `src/kg_craft/prompts.py`
- `configs/*.yaml`
- 输入数据适配层

就能继续向“更严格的一比一复现”靠近。
