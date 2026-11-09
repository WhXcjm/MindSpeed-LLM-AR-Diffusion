# d-OPD 50 轮 A/B 实验执行手册

训练和导出在现有训练容器内执行，评测在现有评测容器内执行。两个容器使用同一套挂载路径，共享代码、原始模型和 `RUN_ROOT`。沿用容器原有的 CANN 环境和 Python，下面的命令均在容器内运行。

## 1. 准备仓库和路径

在两个容器都能访问的工作目录下，并排克隆两个仓库。

```bash
git clone --branch dopd-50-ab --single-branch https://github.com/WhXcjm/MindSpeed-LLM-AR-Diffusion.git
git clone --branch dopd-50-ab --single-branch https://github.com/WhXcjm/lm-eval-harness.git
cd MindSpeed-LLM-AR-Diffusion
cp examples/fsdp2/qwen3_dopd/ab/paths.env.example paths.env
```

打开 `paths.env`，填三个绝对路径。

| 变量 | 填写内容 |
| --- | --- |
| `QWEN3_MODEL` | 原始 `Qwen/Qwen3-1.7B` 本地 HF 权重目录，包含 safetensors、配置、tokenizer 和 chat template。用带聊天模板的 Qwen3-1.7B。 |
| `OPDLM_TRAIN_DATA` | 之前 d-OPD 训练使用的官方 `opdlm_train.json`，共 61,816 条。 |
| `RUN_ROOT` | 本次实验专用的共享输出目录，训练和评测容器可读写。 |

两个仓库使用其他摆放方式时，再填写 `EVAL_REPO`。正式评测默认用 8 张卡，可用 `EVAL_NPUS` 指定评测容器里可见的卡数。训练沿用单机 8 卡配置。路径含空格时，在 `paths.env` 中用引号包住路径值。

## 2. 准备初始化权重

在训练容器里进入训练仓库。

```bash
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env prepare
```

脚本给同一份原始权重生成两个加载目录。训练入口使用 `models/a2d-init` 的扩散模型配置，评测入口使用 `models/a2d-init-hf` 的 HF 配置。两个目录都引用同一份 safetensors，`<|MASK|>` 使用已有的预留词表行 151669。训练初始化和两个 AR 对照均从这套原始参数出发。

## 3. 先跑三个初始化对照

在评测容器里进入同一个训练仓库目录。可先做一题烟测，确认 tokenizer、NPU 和扩散后端加载成功。烟测结果单独存放。

```bash
EVAL_NPUS=1 bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval original-ar --limit 1
EVAL_NPUS=1 bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval init-diffusion --limit 1
```

再跑正式的 200 题。

```bash
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval original-ar
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval init-ar
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval init-diffusion
```

`original-ar` 与 `init-ar` 预期逐题生成同样的 token。它们使用相同的 HF 因果解码、tokenizer 和原始参数，汇总会对比生成 token 与停止原因。用 `init-ar` 和 `init-diffusion` 对照，可以单看同一初始化权重切换解码方式之后的差异。

## 4. 训练 50 轮并导出

回到训练容器，顺序运行。

```bash
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env train baseline
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env train combined
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env export baseline
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env export combined
```

| 训练组 | KL 监督位置 `dopd.loss_mask_mode` | 扩散块边界 `dopd.block_alignment` |
| --- | --- | --- |
| `baseline` | `newly_unmasked`，只监督本轮揭示的响应 token | `response`，从回答起点划块 |
| `combined` | `remaining_mask`，监督所选状态中仍被遮盖的响应 token | `prompt`，按完整序列划块，题面尾部与回答开头共享首块 |

两组都保留块内双向注意力、原训练数据、训练种子、教师、验证器和 483 轮训练日程。`stop_after_iteration=50` 控制何时停下来保存。每轮 128 条轨迹，优化器全局 batch 为 8，第 50 轮对应 800 次优化器更新。成功的 `train` 命令会检查检查点记录与本组参数。

完整 DCP 位于 `$RUN_ROOT/train/<组>/checkpoints/global_step_50`，含模型、优化器、调度器和迭代状态。供评测加载的 HF 权重位于 `$RUN_ROOT/exports/<组>50`，其中 `source_dcp.json` 记录来源。续训使用完整 DCP。

## 5. 评测两个训练组并汇总

在评测容器执行。

```bash
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval baseline50
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval combined50
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env compare
```

五组都直接读评测仓库中已抽好的 GSM8K 文件。它来自官方 test split 的 1,319 题，seed 42 无放回抽 200 题，再按原题号排序。0-shot 题面明确要求 `\boxed{}`，统一使用 Qwen3 模板的 `enable_thinking=False`，每题贪心解码，最多生成 1024 个新 token。256 准确率按同一次生成的前 256 token 评分。评分取最后一个完整的 `\boxed{…}`，用 Math-Verify 比较其数学值。

- `eval/<组>/summary.json` 分别汇总 256 前缀和完整预算的准确率、EOS 结束比例、EOS 条件准确率、完整盒子率、生成长度中位数。另有旧式首个简单盒子的字符串精确匹配作对照。
- `eval/<组>/samples.jsonl` 保留逐题输入、原始生成 token、有效回答、首次闭合盒子的位置、盒子之后的续写长度、首末答案是否改值、实际 EOS 或预算停止。扩散日志还保留 EOS 所在块内的原始尾部和解码计数。生成出来的思考标签单独统计。
- `comparison.json` 先验证五组题号、输入 token、评分配置一致，再给出逐题配对的增益、损失和两个 AR 组的 token 一致率。训练日志在 `train/<组>/logs`，评测进度日志在 `eval-logs`。

正式结果目录要求为空。遇到中断时，先保留失败目录和日志，将该组的结果目录改名后再重跑同一条命令。可选的同权重换边界评测写入 `eval-cross`，与正式五组分开。

```bash
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval baseline50 --block-alignment prompt
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env eval combined50 --block-alignment response
```

首个盒子晚于 256 token，说明短前缀尚未给出可评分的完整答案。首个盒子早、盒后继续写很长时，结合首末值变化和 EOS 区分重复、改答和未及时结束。EOS 后的同一块尾部与有效续写分别保存。两处修复同时启用的 A/B 差值代表组合效果。换边界评测可进一步看推理时边界的影响。

## 6. 后续需要续训时

确认继续训练后，在训练容器运行所需组。

```bash
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env resume baseline
bash examples/fsdp2/qwen3_dopd/ab/run.sh paths.env resume combined
```

默认从第 50 轮继续至 483 轮，并保持所属组的开关。如果续训过程中已经保存了第 100 轮或更晚的检查点，把最新完整周期检查点的轮次写在组名后，例如 `resume combined 100`。首次训练在第 50 轮前中断时，保留并改名该组的训练目录，再从初始化重跑该组。
