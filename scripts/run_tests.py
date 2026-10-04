"""Offline pytest + Allure results, with logs preserved even when tests fail."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def run_logged(command: list[str], log_path: Path) -> int:
    env = dict(os.environ, PYTHONUTF8="1")
    with log_path.open("w", encoding="utf-8") as log:
        try:
            with subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return process.wait()
        except OSError as error:
            log.write(f"Unable to start command: {error}\n")
            print(f"命令启动失败，详情见 {log_path}")
            return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-report", action="store_true", help="生成并打开 Allure 2 单文件报告")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="在 -- 后指定测试路径或 pytest 参数")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("allure_pytest") is None:
        print("请先运行：python -m pip install -r requirements-dev.txt")
        return 2
    extra = args.pytest_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if any(value.startswith(("--alluredir", "--clean-alluredir", "--junitxml", "--junit-xml")) for value in extra):
        parser.error("报告输出位置由此脚本统一管理，请勿覆盖。")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    results = reports / "allure-results"
    html_dir = reports / "allure-report"
    log = reports / "pytest.log"
    # Remove our old HTML first so a failed generation cannot look like a new success.
    if args.open_report and html_dir.exists():
        shutil.rmtree(html_dir)
    code = run_logged([
        sys.executable, "-m", "pytest", "-q", "--tb=short",
        f"--alluredir={results}", "--clean-alluredir",
        f"--junitxml={reports / 'junit.xml'}", *extra,
    ], log)
    results.mkdir(exist_ok=True)
    (results / "environment.properties").write_text(
        f"Python={platform.python_version()}\nOS={platform.system()}\nMode=offline\n",
        encoding="utf-8",
    )
    print(f"\n测试退出码：{code}；日志：{log}；Allure 数据：{results}")
    if not args.open_report:
        return code
    allure = shutil.which("allure")
    if not allure:
        print("未找到 Allure 2 命令行工具。测试日志已保存；安装步骤见 docs/testing.md。")
        return code or 2
    report_code = run_logged([
        allure, "generate", str(results), "--clean", "--single-file", "-o", str(html_dir),
    ], reports / "allure.log")
    index = html_dir / "index.html"
    if report_code or not index.is_file():
        print("报告生成失败，请查看 reports/allure.log；测试结果仍保留。")
        return code or report_code or 2
    print(f"Allure 报告：{index}")
    webbrowser.open(index.resolve().as_uri())
    return code  # Failed tests still return nonzero after opening the report.


if __name__ == "__main__":
    raise SystemExit(main())
