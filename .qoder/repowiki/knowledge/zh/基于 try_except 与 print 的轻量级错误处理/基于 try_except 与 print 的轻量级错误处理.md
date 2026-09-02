---
kind: error_handling
name: 基于 try/except 与 print 的轻量级错误处理
category: error_handling
scope:
    - '**'
source_files:
    - demo.py
    - snake_env.py
    - train.py
    - d3qn_agent.py
    - d3qn_network.py
---

## 1. 使用的系统/方法

该仓库是一个小型 PyTorch + Gymnasium 强化学习演示项目，**没有定义任何自定义异常类型、错误码或统一的错误处理框架**。错误处理完全依赖 Python 内置的 `try/except` 块配合 `print` 输出进行“静默降级”和交互式中断处理。

- 未使用 `logging` 模块（无日志级别管理）。
- 未抛出任何自定义 `Exception` 子类。
- 未使用 `raise` 主动抛出异常。
- 未使用 `assert` 做运行时断言。
- 未使用 `panic/recover`（Python 无此概念）。

## 2. 关键文件与位置

| 文件 | 错误处理片段 | 行为 |
|---|---|---|
| `demo.py` | L19–L27：`try: agent.load(...) except Exception as e:` | 加载预训练模型失败时打印警告并回退到未训练智能体 |
| `demo.py` | L80–L99：`try: while True: ... except KeyboardInterrupt:` | 用户按 Ctrl+C 时优雅退出 pygame 环境 |
| `snake_env.py` | L227–L238：`try: while True: ... except KeyboardInterrupt:` | 独立运行环境测试脚本时捕获中断并调用 `env.close()` |

## 3. 架构与约定

- **容错式加载**：`demo.py` 中模型加载被包裹在 `try/except Exception` 中，捕获所有异常后以“可恢复”的方式继续执行——即打印 `⚠ Could not load trained model: {e}` 并重新构造一个未训练的 `D3QNAgent()`。这是一种“尽力而为”的策略，保证 demo 脚本不因缺少权重文件而崩溃。
- **交互式中断处理**：所有包含无限循环（pygame 渲染主循环）的地方都显式捕获 `KeyboardInterrupt`，并在退出前调用 `env.close()` 释放 pygame 资源。这是本项目中唯一一处有明确清理逻辑的错误处理路径。
- **训练/评估流程零异常**：`train.py` 的 `Trainer` 类、`d3qn_agent.py` 的 `optimize_model`、`d3qn_network.py` 的前向传播均**不捕获异常也不抛出异常**。任何底层异常（如张量维度不匹配、CUDA OOM、键缺失等）都会直接向上冒泡至进程，导致程序终止。`train_one_episode` 也没有 try/except 包裹。
- **无返回值错误信号**：`optimize_model` 在 replay buffer 不足时返回 `None`（见 `train.py` L68 的 `if loss is not None`），但这不是异常，而是通过返回值表示“本步未更新”。

## 4. 约定与约束

- **约定**：对外部依赖（磁盘 I/O 的模型加载）使用宽泛的 `except Exception` 做降级；对交互循环使用 `except KeyboardInterrupt` 做优雅退出。
- **约束**：核心训练与推理路径上没有任何错误检查或防御性代码——如果传入的 state 形状不正确、PyTorch 计算出错、Gymnasium step 返回格式不符合预期，程序会直接崩溃。
- **约束来源**：从代码实现看，该项目属于“最小可用演示”，作者有意省略了健壮的错误处理，将重点放在 D3QN 算法本身。

## 总结

该仓库的错误处理是**极简且非结构化的**：仅存在两处 `try/except`（模型加载降级 + 用户中断处理），其余代码路径假设输入与环境始终合法。对于生产用途，需要补充自定义异常类型、参数校验、结构化日志以及训练循环中的异常捕获与恢复机制。