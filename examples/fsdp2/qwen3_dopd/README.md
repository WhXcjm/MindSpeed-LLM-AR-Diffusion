# Qwen3-1.7B d-OPD：论文对齐的单机八卡训练

本目录与现有的 Nemotron_Diffusion 实验相互隔离。这里提供的 YAML 不再读取七个 Megatron CPT 数据前缀，而是直接使用公开的 OPDLM 指令混合数据，并启用 d-OPD 正确性门控。

## 原始训练数据

论文使用 `divelab/opdlm_train_data`，它由 61,816 条提示组成，包含以下七个数据源：

- 代码：TACO、KodCode-Light-RL、AceCode；
- 数学：DAPO、Nemotron-v2-Math；
- 科学：Nemotron-v2-STEM；
- 对话：Nemotron-v2-Chat。

每条 JSON 数据都包含 `question`、`ground_truth_answer`、`domain`、`source` 和 `tests_json` 字段。数据加载器会拒绝记录数不等于 61,816 的数据版本，并按照原项目为不同领域追加提示后缀，然后应用 Qwen 对话模板，其中 `add_generation_prompt=true`、`enable_thinking=false`。不会额外执行过滤或去重。

在共享文件系统上，从仓库根目录执行一次下载：

```bash
bash examples/fsdp2/qwen3_dopd/download_opdlm_train_data.sh
pip install -r examples/fsdp2/qwen3_dopd/requirements_dopd.txt
```

如果两个节点不共用 Python 环境，需要在两个节点上都安装依赖。如果共享数据路径不同，可通过 `DATA_DIR` 覆盖默认路径。

## 论文对齐约束

启用 `dopd.strict_paper_alignment: true` 后，训练器会在启动时校验以下共用的数据、模型和训练超参数。各组分别配置块边界、KL 监督范围和 EOS 监督。

- 对数据随机打乱并只训练一个 epoch；共 61,816 条提示，每个外层迭代采样 128 条，共执行 483 次外层迭代；最后一个数据池不足时，从打乱后数据的开头补齐；
- 学生从官方 `Qwen3-1.7B-a2d-init` 初始化，教师从普通 `Qwen3-1.7B` 初始化；a2d-init 保留原始 Qwen3 的全部参数（包括 `lm_head.weight`），只把模型类型改为 `a2d-qwen3`，训练器再加载扩散注意力实现；不加载 Nemotron CPT checkpoint，也不重新初始化输出头；
- block size 为 4，每个 block 执行 4 个去噪步骤；使用阈值 0.9 的动态低置信度重掩码；temperature 为 1、top-p 为 1，并禁用 top-k；
- 最大生成长度在前 100 次外层迭代中按余弦曲线从 100 增长到 4,000 token，最大提示长度为 2,000 token；
- 每条 prompt 生成一条 rollout。训练按重建边界分块，再根据每个 token 的首次揭示时间为每块独立抽样；
- d-OPD 修正强度为 0.5，教师与学生候选 token 取 top-16 并集，教师 top-16 前向 KL，评分分块大小为 128，修正要求存在可见未来 token。
- 使用 AdamW：学习率 `1e-5`、betas `(0.9, 0.999)`、epsilon `1e-8`、weight decay 为 0、梯度裁剪为 1.0；使用 BF16 和激活重计算；前 5 次外层迭代进行 warmup，之后使用余弦衰减，最低学习率为峰值的 10%；
- `correctness_gate: true`：正确的数学、科学和代码 rollout 使用未来感知目标；错误样本以及无法验证的对话样本使用普通因果蒸馏目标。

论文在单机 8 张 H100 上训练，并使用带 CPU offload 的 ZeRO-3。本实现使用单机 8 张 Ascend A2 和 FSDP2，因此硬件、并行后端和吞吐无法做到逐比特一致；但优化器全局 batch 仍为 8，每个包含 128 条 rollout 的外层迭代执行 16 次优化器更新。

## 正确性验证器与安全说明

数学领域使用 `math_verify==0.9.0`；科学领域沿用原项目的最终选项判定规则；代码领域支持公开数据中的四种 `tests_json` 格式：`pytest`、`stdio`、`fn_call` 和断言列表。对话领域按设计不进行正确性验证。

代码验证会在限制运行时间、进程数、文件大小和文件描述符数量的条件下执行模型生成的 Python 代码，但这些限制不能视为完整的安全边界。请在断网的容器或虚拟机中运行训练，环境内不要放置密钥、访问令牌、SSH 密钥或敏感挂载目录。不要直接在有特权的宿主机环境中启动启用了论文对齐代码验证器的训练。

## 模型初始化

默认使用以下两个互相独立的本地目录：

- 学生：`/mnt/pipeline-data/f50058111/model_weights/Qwen3-1.7B-a2d-init`，其 `config.json` 的 `model_type` 必须是 `a2d-qwen3`；
- 教师：`/mnt/pipeline-data/f50058111/model_weights/Qwen3-1.7B`，其 `config.json` 的 `model_type` 必须是 `qwen3`。

严格模式还会比较 vocab、隐藏维度、层数和注意力头数，防止把不匹配的教师/学生组合启动起来。MindSpeed 会在加载学生前注册官方的 A2D 配置类型，`trust_remote_code` 保持关闭。

## 单机启动

在 `80.5.5.133` 上从仓库根目录直接执行：

```bash
cd /path/to/MindSpeed-LLM
bash examples/fsdp2/qwen3_dopd/train_dopd_qwen3_1.7b_fsdp2_A2.sh
```

启动脚本默认加载 `baseline(RRR)`，可用 `TRAIN_CONFIG` 选择其他 YAML。可选环境变量还包括 `STUDENT_MODEL_PATH`、`TEACHER_MODEL_PATH`、`DOPD_DATA_PATH`、`OUTPUT_DIR` 和 `MASTER_PORT`。启动器固定要求 `NPUS_PER_NODE=8`。控制台日志保存在 `$OUTPUT_DIR/logs/node_0_<timestamp>.log`，训练指标写入 `metrics.jsonl`。

前 10 个外层迭代的 128 条全局 rollout 会逐条写入 `$OUTPUT_DIR/rollout_audits/iteration_0001.jsonl` 到 `iteration_0010.jsonl`。每条记录含原始问题、完整模型回答、是否生成 EOS、是否撞到长度上限、gold answer、verifier 是否适用、正确与否、答案提取结果，以及代码任务的 return code/stdout/stderr 尾部。聚合失败原因写入同目录的 `summary.jsonl`。

六份配置都每 50 个外层轮次留一份完整历史档，并在本次训练终点保存完整 `latest`。483 轮训练的目录如下：

```text
checkpoints/global_step_50, ..., checkpoints/global_step_450
latest/global_step_483
```

历史档由 `training.save_steps`、`training.save_only_model` 和 `training.save_total_limit` 控制。间隔为 `0` 时关闭历史档。已有轮次目录会拒绝覆盖。`save_total_limit` 只清理历史档。

`dopd.save_latest_steps: null` 关闭滚动保存，`0` 只在终点保存，正整数按间隔和终点保存。终点是 `max_steps` 或 `stop_after_iteration`。用 `dopd.save_latest_only_model` 单独选择是否只存模型。两种策略在同一轮触发时会各写一份。

新版本通过 DCP 校验后才在共享文件系统上发布，随后清理旧版本。续训和推理可直接传入 `<output_dir>/latest`。导出用下面的命令，通过同一入口读取最高的已发布轮次。也可传具体版本目录或已有的 `global_step_*_final`。模型档可推理和导出，续训要求完整状态。

```bash
export PYTHONPATH="${PYTHONPATH:-}"
source examples/fsdp2/env_config.sh
python examples/fsdp2/qwen3_dopd/ab/assets.py export-weights /path/to/run/latest /path/to/hf_export
```

## 实验配置

`dopd.block_alignment` 控制 rollout，`dopd.reconstruct_block_alignment` 控制状态抽样、学生 attention 和教师的同块未来评分。省略重建项或设为 `null` 时，重建继承 rollout 的设置。训练 checkpoint 保存两种边界，旧 checkpoint 从保存的 rollout 边界恢复重建设置。

文件名括号内的三位字母依次表示 rollout、重建（含反传）和扩散推理的块边界。`P` 表示 `prompt`，`R` 表示 `response`。六组均测 AR 和 diffusion，AR 按 token 自回归生成。各组使用同一训练日程和独立输出目录。

| 配置 | KL 监督 | EOS 位置可参与 KL |
| --- | --- | --- |
| [baseline(RRR)](dopd_qwen3_1.7b_fsdp2_A2_baseline%28RRR%29.yaml) | `newly_unmasked` | 否 |
| [kl_only(RRR)](dopd_qwen3_1.7b_fsdp2_A2_kl_only%28RRR%29.yaml) | `remaining_mask` | 否 |
| [prompt_only(PPP)](dopd_qwen3_1.7b_fsdp2_A2_prompt_only%28PPP%29.yaml) | `newly_unmasked` | 否 |
| [kl+prompt(PPP)](dopd_qwen3_1.7b_fsdp2_A2_kl+prompt%28PPP%29.yaml) | `remaining_mask` | 否 |
| [EOS_loss_only(RRR)](dopd_qwen3_1.7b_fsdp2_A2_EOS_loss_only%28RRR%29.yaml) | `newly_unmasked` | 是，首个 EOS 在被抽中时参与 |
| [official(PRP)](dopd_qwen3_1.7b_fsdp2_A2_official%28PRP%29.yaml) | `remaining_mask` | 否 |

三个 `only` 组以 `baseline` 为对照。`prompt_only` 和 `kl+prompt` 沿用已有实验的 `PPP`。`official` 从回答第一个 token 起重建状态，rollout 和扩散推理则将未凑满一块的 prompt 尾部带入首块。 [d-OPD rollout](https://github.com/mit-han-lab/d-OPD/blob/7939f8266934a05096f37b1ed00ae9fef46a7218/dopd/defaults.py#L11-L22) · [d-OPD 训练重建](https://github.com/mit-han-lab/d-OPD/blob/7939f8266934a05096f37b1ed00ae9fef46a7218/dopd/trainer.py#L950-L1004) · [OPDLM JetEngine 首块](https://github.com/divelab/OPDLM/blob/3bb8bf60a882131aedccf43575f896be95eda0f0/sample/jetengine_ext/engine/sequence.py#L23-L38)

在已有 MindSpeed 训练环境中，从仓库根目录启动。模型、数据和输出路径可在 YAML 中填写或用 `--组名.字段=值` 覆盖。括号属于文件名，命令中引用路径时加引号。

```bash
export PYTHONPATH="${PYTHONPATH:-}"
source examples/fsdp2/env_config.sh
torchrun --standalone --nproc_per_node=8 train_dopd_fsdp2.py \
  'examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2_official(PRP).yaml'
```

`dopd.inference` 中的 `modes: [ar, diffusion]` 在同一权重、同一题面上跑两种解码。五个对照组沿用既有评测的贪心解码和 1024 新 token 上限。`official` 使用附录 C.4 的扩散设置，block size 4、每块最多 4 步、静态低置信度重掩码、temperature 1、top-p 1、top-k 1、最多 4096 个新 token。六组均使用 Qwen3 chat template 并关闭 thinking。 [d-OPD 附录 C.4](https://arxiv.org/html/2609.35362v1#A3.SS4)

本地单条 DCP 推理可显式读取同一份 YAML。直接调用 Python 推理入口时，也可从 checkpoint 对应的 `dopd_config.json` 自动读取推理设置，命令行参数优先。输出 JSON 中的 `ar_result` 记录 AR 回答，`results` 记录扩散回答和轨迹。

```bash
CHECKPOINT_PATH=/path/to/run/latest \
INFERENCE_CONFIG='examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2_official(PRP).yaml' \
  bash examples/fsdp2/qwen3_dopd/inference_dopd_qwen3_1.7b_fsdp2_A2.sh
```

固定 GSM8K 200 题的同口径对比用 [`ab/run.sh … eval-group <组名>`](ab/README.md#5-评测两个训练组并汇总)，每组依次跑 AR 和该组边界下的扩散。

论文六项基准使用 OPDLM 的零样本任务模板与评分。导出 HF 权重后，可在 OPDLM 仓库中使用以下评测入口和附录 C.4 的解码参数。[OPDLM 评测入口](https://github.com/divelab/OPDLM/blob/3bb8bf60a882131aedccf43575f896be95eda0f0/pure_inference/eval.py)

```bash
for mode in qwen bd3lm; do
  python pure_inference/eval.py --models /path/to/exported-model --model_bases "$mode" \
    --datasets MMLU MMLU_Pro GPQA_Diamond_shuffled GSM8K MATH500 AIME2025 \
    --block_size 4 --denoising_steps_per_block 4 --remasking_strategy low_confidence_static \
    --temperature 1.0 --top_p 1.0 --top_k 1 --max_token 4096 --num_response_per_task 1 \
    --tag "$mode"
done
```

本地解码诊断可用 `PAPER_ALIGNED=0` 调整 block size 或阈值。旧配置默认使用 `response` 推理，输出 JSON 将它标为解码消融。JSON 还记录实际边界、`TPF_total`、`TPF_decode` 和 `TPF_denoise_only`。论文的 1.35–1.58 倍加速衡量达到 OPDLM 准确率所需的训练时间。

正式训练前，建议使用单独的输出目录进行一次单机冒烟测试。严格论文校验在这个缩短实验中必须关闭：

```bash
export OUTPUT_DIR=/share/dataset/x00840191/model_train/ckpt/dopd_smoke
torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 \
  --master_addr 127.0.0.1 --master_port 6503 train_dopd_fsdp2.py \
  'examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2_baseline(RRR).yaml' \
  --training.output_dir=/share/dataset/x00840191/model_train/ckpt/dopd_smoke \
  --training.max_steps=1 --dopd.strict_paper_alignment=false \
  --dopd.prompts_per_iteration=16 --dopd.min_response_tokens=16 \
  --dopd.max_response_tokens=16 --dopd.response_schedule_steps=1 \
  --dopd.score_chunk_size=16
```

本仓库的 YAML 参数解析器要求命令行覆盖项使用 `=` 形式。重复执行冒烟测试前，请先删除或移动原来的冒烟测试输出目录；训练器会拒绝覆盖已经存在 checkpoint 的目录。
