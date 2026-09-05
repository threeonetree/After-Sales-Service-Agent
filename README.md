# After-Sales Service Agent

An after-sales assistant for robot vacuums built with LangGraph, Qwen, RAG, and Streamlit.

## Features

- Answers troubleshooting, maintenance, and product questions from a local knowledge base.
- Accepts robot photos and App error screenshots alongside text; extracts visual
  observations, searches the existing text knowledge base, and displays cited passages.
- Supports image follow-ups with bounded text context and isolated conversations.
- Uses tools for weather, user profiles, and robot usage records.
- Resolves current, previous, or explicit-month record lookups deterministically,
  without spending chat-model quota or letting the model choose another month.
- Generates personalized reports only for one explicitly selected month that
  has a usage record.
- Includes tool-contract evaluations for checking tool order and required arguments.

## Project Structure

- `agent/`: LangGraph ReAct agent and tools.
- `rag/`: Chroma-based knowledge retrieval.
- `services/`: weather and user-data services.
- `prompts/`: system and report prompts.
- `evals/`: tool-contract cases and evaluation runner.
- `app.py`: Streamlit application entry point.

## Model configuration

The application calls Alibaba Cloud Model Studio only; it does not run a local
model and has no paid-model fallback.

- Chat: `qwen3.7-flash-2026-07-15`
- Embedding: `qwen3.7-text-embedding`
- Credential: one shared `DASHSCOPE_API_KEY`
- Chat protocol: Bailian's OpenAI-compatible multimodal endpoint
- Embedding integration: `DashScopeEmbeddings`

Keep "stop when free quota is exhausted" enabled for both models. When Bailian
returns `403 AllocationQuota.FreeTierOnly`, the UI shows a quota-exhausted
message and stops the request.

## Windows 10 setup

Python 3.10 or newer is supported. The example below uses Python 3.10 in
PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the `DASHSCOPE_API_KEY` already configured in Windows, or copy
`.env.example` to `.env` and replace the placeholder locally. Never commit the
real key.

The default chat endpoint is the mainland China endpoint. If the API Key belongs
to another Bailian region, set `DASHSCOPE_BASE_URL` to that region's
OpenAI-compatible `/compatible-mode/v1` endpoint as well.

Run three small live checks for chat, text embeddings, and tool calling:

```powershell
python -m scripts.probe_models
```

Expected result:

```text
[PASS] chat
[PASS] embedding
[PASS] tool_calling
```

The repository does not commit generated Chroma data. Build the local knowledge
index once after first install or whenever the embedding model changes:

```powershell
python -m rag.rebuild_index --yes
```

This command replaces only the generated `chroma_db` index; it keeps all source
files under `data/`. Start the application after the index is ready:

```powershell
streamlit run app.py
```

Run contract checks without calling the model:

```bash
python -m evals.run_contract_evals --dry-run
```

Install the small development-only dependency and run the offline tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Pytest also collects the repository's existing `unittest.TestCase` tests, so
both styles can coexist. Useful commands while learning:

```powershell
# Run one test file and show each case name
python -m pytest tests/test_personal_data_route.py -v

# Run one specific test method
python -m pytest tests/test_personal_data_route.py::test_current_month_without_record_stops_before_model -v
```

The new routing tests demonstrate three pytest basics:

- `@pytest.fixture`: prepares an isolated temporary CSV for each test.
- `@pytest.mark.parametrize`: runs the same assertion for several questions.
- Plain `assert`: compares the actual response with the expected behavior.

Start with `test_current_month_without_record_stops_before_model`: its comments
separate Arrange (prepare data), Act (call the code), and Assert (check output).
Then read `tests/test_react_agent_routing.py`, which uses the real LangGraph
execution with a scripted model to test tool guards, conversation memory and
user isolation. Pytest blocks network connections during tests, so testing does
not consume model quota or need a real API key.

Personal record and report behavior:

- "查询本月的使用记录" reads the selected user's current-month record directly.
- If that month has no record, the app lists the user's available months and
  does not generate a report.
- "生成使用报告" asks for one month before using the chat model.
- "生成2025年12月使用报告" enters report generation only if that record exists.
- Multi-month reports are not supported yet; this prevents silent date-range
  expansion and invented comparisons.
- An ambiguous date such as "12月" asks for a full year and month. After a
  month-selection prompt, replying "2025-12" resumes the record/report request.
- Switching the selected user starts a new conversation. Report generation uses
  only the approved user's month and does not read older reports as source data.

For an existing installation, follow the multimodal update steps below.

## 多模态售后：文字 + 图片

使用现有 `qwen3.7-flash-2026-07-15` 理解图片并生成答复，继续用
`qwen3.7-text-embedding` 和 `DashScopeEmbeddings` 检索知识库。
一个已有的 `DASHSCOPE_API_KEY` 即可，不部署本地 AI、不新增模型服务。
该 Flash 快照的图片输入及结构化输出能力见
[百炼模型说明](https://help.aliyun.com/zh/model-studio/qwen3-7-flash)，图片通过
[OpenAI 兼容接口的 Base64 输入](https://help.aliyun.com/zh/model-studio/vision)发送。

### 图片如何使用现有文本知识库

1. 校验上传的实际文件内容，纠正手机照片方向、移除 EXIF、缩小图片。
2. 视觉模型读取照片或截图，提取可见现象、能确认的报错文字和不确定项。
3. 把提问和图片观察组成文字查询，复用现有向量检索 + BM25 检索。
4. 用同一个聊天模型结合资料生成中文排查建议，页面展示实际采用的资料片段。

例如：滚刷照片 + “为什么扫不干净” → “滚刷可见毛发缠绕” →
检索文本里的滚刷维护内容 → 给出有资料支持的清理建议。
这属于**视觉理解 + 文本 RAG**。图片不会写入 Chroma；当前 TXT 和能提取文字的
PDF 知识库可以继续使用，尚不对知识库 PDF 内嵌的图片建立图像索引。

### 使用方式与边界

- 在聊天输入框点击附件按钮，可同时上传 1–3 张 JPG / PNG / WebP 静态图片。
  每张最多 5 MB、2000 万像素；发送前最长边缩小至 1600 像素。
- 支持仅图片提问，也支持“照片 + 问题”，例如“这处缠绕会影响清扫吗？”
  或“截图上的报错是什么意思？型号是……”。多图会按图1、图2、图3分析。
- 可展开“图片识别结果”核对现象和文字，展开“查看知识库依据”检查参考片段。
  引用编号由程序核验，只展示模型实际选择的已检索资料；引用存在不等于
  每项判断都已被事实验证，仍需结合型号和原始资料检查答复。
- 看不清或不相关的图片会先要求补充信息。没有相关知识依据时，应明确说明
  依据不足，不能把通用问答资料冒充特定型号的故障码说明书。
- 围绕当前图片的文字追问使用图片观察、最初的问题及最近两轮问答，不重复发送原图。
  若需要查看之前未描述的细节，请重新上传。新上传替换上一组图片的观察；
  新图分析失败时不会自动沿用旧图。
- 查询使用记录、生成报告继续走原来的月份/用户校验流程，并结束当前图片追问。
  图片内的账号或月份不会触发个人记录查询。
- “新对话”和切换用户会清空当前对话及图片上下文。页面只保留最近三组图片预览；
  应用没有把图片写入磁盘或向量库。图片会发送到配置的百炼服务，请先遮挡无关信息。
- 本版输入范围是文字和静态图片，输出为文字；语音、视频和自动创建维修工单尚未接入。

### 免费额度与失败行为

请继续为聊天模型和 Embedding 模型分别开启“免费额度用完即停”。
一次正常图片问答通常包含 **2 次聊天模型调用 + 1 次查询文本向量化**；
图片追问通常为 **1 次聊天模型调用 + 1 次查询向量化**。
模糊/无关图片通常在第一步停止；检索无资料时不调用第二次聊天模型。
这不是永久无限免费服务，能否继续运行取决于账户的剩余额度和有效期。

程序限制图片数量、尺寸及模型输出长度；畸形或截断 JSON 会提示重试，不自动
增加一次模型调用来修复。额度错误、网络错误或服务错误会结束当前请求，
不切换其他模型。SDK 原有的瞬时故障重试策略仍保留，实际请求数可能因此增加。
页面刷新不会重新提交已完成的图片问题。

### 已有项目的一次更新（Windows PowerShell）

先在运行 Streamlit 的终端按 `Ctrl+C`。在项目目录使用已有 `.venv`：

```powershell
cd D:\Pycharmfile\DLtest\machine
.\.venv\Scripts\Activate.ps1
git pull --ff-only origin main
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python -m scripts.probe_models --vision-only
python -m streamlit run app.py
```

不需要重新创建 `.venv`、更改 Key 或重建现有向量库。
依赖安装会复用已满足版本要求的包；本次明确声明了图片处理和 JSON 校验依赖。
若 `git pull` 提示本地修改冲突，先保留本地修改再处理，不要直接覆盖。

`--vision-only` 用一张程序生成的色块和文字图片检查模型是否真正识别图片、
以及 JSON 接口是否可用，只进行一次聊天模型检查，成功时输出 `[PASS] vision`。
它会使用真实 API 和免费额度。原来的 `python -m scripts.probe_models` 仍只跑
聊天、Embedding、工具调用三项；加 `--vision` 则运行全部四项。
图片自检仅验证视觉接口，不等同于售后答案准确率测试。

### 自动测试与人工验收

所有 pytest 测试阻止网络连接，不需要 Key，也不消耗模型额度。

```powershell
# 图片校验、图片到知识库流程：适合先读这些测试学习 fixture / 参数化 / Mock
python -m pytest tests/test_image_input.py tests/test_visual_support.py -v

# 真正运行 Streamlit 页面，使用模拟 Agent 回答
python -m pytest tests/test_multimodal_app.py -v
```

页面测试覆盖上传后预览、仅图片发送、来源展示、用户切换、失败提示及刷新不重复请求；
文件上传值通过 Streamlit 的返回对象注入，不包含浏览器文件选择器的端到端测试。
Agent 测试使用真实 LangGraph 和模拟模型，覆盖看图、追问、新旧图片隔离和原有查询回归。

连接真实模型后，用自己的照片完成以下验收：

| 输入 | 应检查的结果 |
| --- | --- |
| 滚刷缠绕照片 + “怎么清理？” | 正确指出可见缠绕；建议与展示的滚刷资料相符 |
| 清晰 App 报错截图 | 报错文字抄录准确；未知代码不会被编造解释 |
| 两张设备不同角度照片 | 图号与内容对应，不把不同画面合并成不存在的故障 |
| 仅上传图片 | 能开始分析或提出具体补充问题 |
| 模糊图片或无关图片 | 要求补拍/说明，不给肯定的故障诊断 |
| “已经清理了，接下来呢？” | 沿用当前现象和已做步骤，不声称看到新的图片细节 |
| 看图后查询本月使用记录 | 按当前用户和本月查询，不生成跨月份报告 |
| 新对话或切换用户后问“刚才图片呢？” | 不带入上一段对话的图片观察 |

这些人工检查验证实际识别与检索质量，不能用离线单元测试的通过数代替。

## Notes

Do not commit API keys, local vector databases, logs, or evaluation outputs.
An index manifest stored inside `chroma_db/` prevents vectors from different
embedding models from being mixed.
