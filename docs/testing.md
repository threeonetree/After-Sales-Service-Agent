# 测试、Allure 报告与出错定位

项目在 Windows 使用已有的 `.venv`。新开终端时只需要激活，不要重复创建：

```powershell
cd D:\Pycharmfile\DLtest\machine
.\.venv\Scripts\Activate.ps1
```

## 1. 先运行测试

首次使用测试工具，或更新了 `requirements-dev.txt` 时安装开发依赖：

```powershell
python -m pip install -r requirements-dev.txt
python -m scripts.run_tests
```

此命令运行所有离线测试，生成 `reports/pytest.log`、`reports/junit.xml` 和
`reports/allure-results/`。测试失败时仍保存日志和结果，并返回非零退出码。
每次运行清理旧的 Allure 结果，避免把上次成功的用例混在这次报告里。
不要同时运行两个报告命令，它们使用同一个输出目录。

普通 `python -m pytest -q` 仍可使用，但不会自动生成 Allure 数据。
自动测试模拟模型并阻止网络连接，不读取本地 `.env`、不使用真实 Key、不消耗模型额度。
测试不会清空 Windows 的用户目录环境变量，也不会把图片二进制作为用例名称。

## 2. 首次配置 Allure 页面工具（只做一次）

`allure-pytest` 负责收集数据；生成网页需要单独安装 **Allure 2 命令行工具**。
本项目脚本使用 Allure 2 的参数，不是 Allure 3。

1. 在 PowerShell 执行 `java -version`。若找不到 Java，先安装 JDK（例如 JDK 17），
   配置 `JAVA_HOME` 为 JDK 安装目录，并把 `%JAVA_HOME%\bin` 加入用户 `Path`。
2. 按 [Allure 2 Windows 官方安装说明](https://allurereport.org/docs/v2/install-for-windows/)
   下载 2.x 的 zip，解压到固定目录（例如 `D:\Tools\allure-2.x.x`），
   把其中的 `bin` 目录加入 Windows 用户环境变量 `Path`。
3. 关闭并重新打开 PowerShell（PyCharm 内终端可能需要重启 PyCharm），重新激活 `.venv`。
   执行 `java -version` 和 `allure --version`，确认均能显示版本。

无需为了 Allure 安装 Node.js，也不用修改模型 Key。
如果暂时不安装 Allure，照样可以运行测试，并直接看 `reports/pytest.log`。

## 3. 运行并打开报告

```powershell
python -m scripts.run_tests --open-report
```

即使测试失败，脚本也会继续生成并打开报告。生成位置为
`reports/allure-report/index.html`，这是 Allure 2 的单文件报告，以后可直接双击打开。
若浏览器没有自动弹出，手动打开这个文件即可。
报告生成失败时查看 `reports/allure.log`；不要把旧报告当作当前测试结果。

只运行一个文件或一个用例：

```powershell
python -m scripts.run_tests --open-report -- tests/test_image_input.py
python -m scripts.run_tests --open-report -- tests/test_multimodal_app.py::test_user_switch_clears_history
```

这些命令会覆盖之前的报告，页面显示的是此次选定的用例。

## 4. 怎么知道 error 在哪里

在 Allure 左侧选择 **Suites**，展开对应功能分组，选择红色 **Failed** 或黄色 **Broken** 的用例。

- `Failed` 常见于断言不满足：查看 expected / actual，弄清楚哪里与预期不同。
- `Broken` 常见于准备环境、依赖或执行过程中的异常。以实际异常和 traceback 为准。
- 打开用例中的错误详情 / **Traceback**，查看异常类型、测试文件和行号。
  从调用栈末尾往上看，找到本项目代码的位置。
- **Attachments** 中可以查看 pytest 捕获的 stdout、stderr 和 logging；
  也可直接打开 `reports/pytest.log` 搜索 `FAILED`、`ERROR`、`AssertionError`。

假设报错指出 `tests/test_multimodal_app.py::test_user_switch_clears_history`，可单独重跑：

```powershell
python -m pytest tests/test_multimodal_app.py::test_user_switch_clears_history -vv --tb=long
```

只有需要查看参数和局部数据、并确定不包含敏感信息时，才额外使用 `--showlocals`。
不需要为了看报错再次安装全部依赖。

## 5. 网页使用时报错，看运行日志

**Allure 展示测试结果，不会自动收集你在网页实际聊天时发生的错误。**

客户页面只显示简短提示和“问题编号”。维护时在 `logs/support_YYYYMMDD.log`
搜索该编号，可以找到对应的错误调用栈、文件与行号。日志会隐藏已配置的 API Key
和图片 Base64，不记录会话图片；也不要把完整日志直接公开发布。
API 自检等命令仍保留详细的开发诊断。

## 6. GitHub 自动测试与验证范围

仓库的 **Actions → Offline regression** 在 PR 和 main 更新时运行：
Windows + Python 3.10、Linux + Python 3.12。每个任务保存 `reports/` 作为 artifact，保留 7 天。
下载解压后可以阅读日志；要生成页面，对其中的 `allure-results` 运行：

```powershell
allure generate .\allure-results --clean --single-file -o .\allure-report
```

自动测试覆盖输入校验、图片到检索流程、对话隔离、页面交互、额度失败、报告生成和日志脱敏。
页面测试使用 Streamlit AppTest 和模拟 Agent，上传通过返回对象注入，
不等于真实浏览器上传控件的端到端测试。Windows CI 也不等于已在你的 Win10 电脑实测。
实际照片识别、知识检索质量和真实模型答复，仍需要在本机做人工验收。

参考：[Allure pytest](https://allurereport.org/docs/pytest/)、
[单文件报告查看方式](https://allurereport.org/docs/v2/view-report/)。
