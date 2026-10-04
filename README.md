# After-Sales-Service-Agent

**支持文字与图片问答的扫地机器人智能售后助手。**

[![Offline regression](https://github.com/threeonetree/After-Sales-Service-Agent/actions/workflows/tests.yml/badge.svg)](https://github.com/threeonetree/After-Sales-Service-Agent/actions/workflows/tests.yml)

基于 LangGraph、Qwen 和 RAG 构建。用户可以描述故障、上传设备照片或 App 报错截图，
由助手结合售后知识库给出排查建议，也可以查询使用记录、生成单月使用报告。

应用在本机运行，通过阿里云百炼调用模型，无需独立显卡或本地部署大模型。

## 功能

| 功能 | 使用方式 |
| --- | --- |
| 知识库问答 | 询问清扫故障、配件维护、保养方法和产品选购问题 |
| 图片辅助排查 | 上传设备、配件照片或报错截图，结合文字描述获取建议 |
| 多轮追问 | 在当前对话中继续补充现象、反馈已尝试的步骤 |
| 使用记录查询 | 查询本月、上月或指定月份；没有记录时提示可查询的月份 |
| 单月使用报告 | 根据已有记录生成指定月份的使用情况与保养建议 |
| 用户切换 | 从下拉框选择演示用户，切换时清空当前对话和图片上下文 |

图片支持 JPG、PNG、WebP，每次最多 3 张，单张不超过 5 MB。页面提供缩略图和放大查看，
回答直接面向客户，不展示内部识别清单或检索过程。

## 快速开始

### 1. 准备环境

- Python 3.10 或以上版本，以及 Git。
- 一个可调用所配置聊天模型与 Embedding 模型的百炼 API Key。
- 能连接模型服务的网络环境。

克隆项目：

```bash
git clone https://github.com/threeonetree/After-Sales-Service-Agent.git
cd After-Sales-Service-Agent
```

以下以 **Windows PowerShell + Python 3.10** 为例。已有合适的虚拟环境时，直接激活使用即可。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

<details>
<summary>Linux / macOS 安装命令</summary>

确保 `python3` 为 Python 3.10 或以上版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

</details>

### 2. 配置模型

在系统环境变量中设置 `DASHSCOPE_API_KEY`，或将 [.env.example](.env.example)
复制为项目根目录的 `.env`，填入自己的 Key：

```dotenv
DASHSCOPE_API_KEY=你的百炼_API_Key
```

不要将真实 Key 提交到仓库。模型名称在 [config/rag.yml](config/rag.yml) 中配置：

| 用途 | 默认模型 |
| --- | --- |
| 对话、图片理解与报告生成 | `qwen3.7-flash-2026-07-15` |
| 文本向量化与知识检索 | `qwen3.7-text-embedding` |

两个模型共用一个 Key。默认使用百炼中国内地服务配置；更换模型或接入地址前，
请先确认账号可用模型及其能力，详见[开发文档](docs/开发.md)。

若仅使用免费额度，请在百炼控制台为**两个模型分别开启“免费额度用完即停”**。
知识库初始化和模型调用都会消耗账户额度，剩余额度及有效期以控制台为准。
应用不自动切换到其他模型继续服务。

### 3. 初始化知识库并启动

仓库已附带示例知识资料。首次运行时创建本地索引，然后启动页面：

```bash
python -m rag.rebuild_index --yes
python -m streamlit run app.py
```

在浏览器打开终端显示的地址，默认是 **http://localhost:8501**。
以后启动只需激活原虚拟环境，再执行 `python -m streamlit run app.py`。

索引重建命令会替换生成的 `chroma_db/`，保留 `data/` 中的源文件；无需每次启动都重建。

## 试着这样提问

在页面顶部选择用户后，可以直接发送文字，或点击输入框的附件按钮上传图片。

| 场景 | 示例 |
| --- | --- |
| 常见故障 | “扫地机器人清扫时经常漏扫怎么办？” |
| 照片排查 | 上传滚刷照片：“这处缠绕会影响清扫吗？应该怎么处理？” |
| 报错截图 | 上传 App 截图：“这个报错是什么意思？设备型号是……” |
| 继续追问 | “已经清理了，接下来呢？” |
| 查询记录 | “查询 2025 年 12 月的使用记录” |
| 生成报告 | “生成 2025 年 12 月的使用报告” |

用户资料和使用记录来自 [data/external/](data/external/) 中的演示数据，未连接真实设备。
查询“本月”时若没有记录，助手会列出已有月份；生成报告需要选定一个有记录的月份。
讨论另一台设备或开始新的问题时，点击“新对话”即可。

## 使用自己的知识资料

将 TXT 或可提取文字的 PDF 文件放入 `data/`，重新运行：

```bash
python -m rag.rebuild_index --yes
```

完成后重启应用。分块和检索配置见 [config/chroma.yml](config/chroma.yml)。

图片问答会先提取照片中的可见现象和文字，再检索这些文本资料，最后生成答复。
因此现有文本知识库可以直接用于图片辅助排查；当前不会对 PDF 内嵌图片建立图像索引。

## 技术栈与目录

| 模块 | 实现 |
| --- | --- |
| 聊天界面 | Streamlit |
| Agent 与工具调用 | LangGraph、LangChain |
| 图片理解与文本生成 | Qwen，百炼 OpenAI 兼容接口 |
| 知识检索 | Chroma 向量检索 + BM25，RRF 融合排序 |
| 文本向量化 | DashScopeEmbeddings |
| 自动测试 | pytest、Allure、GitHub Actions |

| 路径 | 内容 |
| --- | --- |
| [app.py](app.py) | 页面入口 |
| [agent/](agent/) | Agent、工具与记录查询路由 |
| [services/](services/) | 图片处理、视觉问答与数据服务 |
| [rag/](rag/) | 知识库入库与检索 |
| [config/](config/)、[prompts/](prompts/) | 配置与提示词 |
| [data/](data/) | 示例知识资料和用户数据 |
| [tests/](tests/)、[evals/](evals/) | 自动测试与工具契约评估 |
| [docs/](docs/) | 开发说明、测试及排错指南 |

## 测试与更多文档

运行不调用真实模型的离线测试：

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.run_tests
```

- [开发文档](docs/开发.md)：实现原理、模型自检、配置细节与人工验收。
- [测试与 Allure 指南](docs/testing.md)：报告安装、查看失败用例与错误定位。
- [工具契约评估](evals/README.md)：Agent 工具调用的检查方法。

## 当前范围

本项目用于多模态售后场景的学习与演示。用户下拉框用于切换演示数据，尚未接入登录鉴权。
目前支持文字和静态图片输入、文字回答及单月报告；语音、视频、真实设备接入和维修工单尚未实现。
图片判断和建议的质量取决于照片清晰度、模型能力及知识资料的覆盖范围。
