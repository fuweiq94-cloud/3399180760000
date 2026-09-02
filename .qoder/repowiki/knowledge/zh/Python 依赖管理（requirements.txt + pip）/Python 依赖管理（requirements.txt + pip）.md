---
kind: dependency_management
name: Python 依赖管理（requirements.txt + pip）
category: dependency_management
scope:
    - '**'
source_files:
    - requirements.txt
---

## 1. 使用的系统/方法

本项目采用 Python 生态中最基础的依赖管理方式：**`requirements.txt` + pip**。仓库根目录仅包含一个 `requirements.txt`，没有使用 `pyproject.toml`、`setup.py`、`Pipfile`、`poetry.lock`、`conda` 环境文件或任何 vendoring 策略。所有第三方库通过 `pip install -r requirements.txt` 安装。

## 2. 关键文件

- `requirements.txt`：声明全部运行时依赖及最低版本约束。
- 各 `.py` 模块中的 `import` 语句（`d3qn_agent.py`、`d3qn_network.py`、`snake_env.py`、`train.py`、`demo.py`）是依赖的实际消费点，用于反向验证清单是否完整。

## 3. 架构与约定

- **集中式清单**：所有依赖集中在根目录的 `requirements.txt`，无子目录级依赖文件。
- **最小版本约束**：每个依赖使用 `>=` 指定最低版本，例如 `torch>=2.0.0`、`numpy>=1.24.0`、`gymnasium>=0.29.0`、`pygame>=2.5.0`、`matplotlib>=3.7.0`，不锁定精确版本，也不使用 `<=` 上限。
- **可选依赖以注释形式标注**：TensorBoard（`torch.utils.tensorboard`，随 PyTorch 自带）和 tqdm（`tqdm>=4.65.0`）被注释掉，作为可选功能保留在清单中但不强制安装。
- **无虚拟环境或锁文件**：仓库未包含 `venv/`、`.venv/`、`requirements.lock`、`poetry.lock` 等锁定文件；也没有 Dockerfile 或 CI 脚本固定构建环境。
- **私有注册表/代理**：未发现任何 `pip.conf`、`~/.pip/pip.conf`、环境变量（如 `PIP_INDEX_URL`、`PIP_TRUSTED_HOST`）或 `--index-url` 参数引用，表明项目默认使用 PyPI 官方源。

## 4. 约定与约束

- **依赖来源**：全部为 PyPI 上的公开包，无本地 vendored 代码或私有 wheel。
- **版本策略**：仅声明下限（`>=`），由 pip 在安装时解析并选择满足条件的最新版本；这意味着不同机器上可能安装到不同大版本，存在潜在兼容风险。
- **导入与清单一致性**：代码实际 import 的包（torch、numpy、gymnasium、pygame、matplotlib）与 `requirements.txt` 一一对应，未见未声明的外部依赖。
- **可选依赖处理**：TensorBoard 和 tqdm 通过注释保留在清单中，使用时需手动取消注释后重新安装，属于“按需启用”的约定而非自动检测。
- **无依赖升级流程**：仓库未提供自动化更新脚本、CI 任务或 PR 模板来定期升级依赖版本，升级完全依赖人工编辑 `requirements.txt`。
- **无依赖冲突解决机制**：由于缺少 lockfile，不存在强制统一依赖树的手段；若多人协作，建议配合 `pip freeze > requirements.freeze.txt` 生成冻结文件以固化环境。

## 总结

这是一个极简风格的 Python 项目，依赖管理停留在最基础的 `requirements.txt` 阶段——适合个人实验或教学演示，但在团队协作、可复现构建和长期维护方面缺乏锁文件、CI 集成和私有源配置等工程化保障。