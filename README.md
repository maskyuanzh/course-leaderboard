# 实训课题：在 8 卡 H200 上从零训练 dense 语言模型

在统一训练预算内，跑通基线并改进模型、数据处理或训练配方。每组交一份可复现的 TXT 日志和一个最终 checkpoint。课程组统一盲评，以 Test Loss 从低到高排名，同分并列。

[打开 Model Lab 排行榜](https://maskyuanzh.github.io/course-leaderboard/)

## 1. 公共起点

使用 Ai2 的 OLMo 2 1B **架构和 tokenizer**。公开配置为 16 层、隐藏宽度 2048，实际约 14.85 亿参数。通过 `AutoModelForCausalLM.from_config` 随机初始化。课程允许改模型结构、数据处理和优化器；每个正式提交都应说明相对基线的改动，并提供源码。

| 项目 | 建议开课配置 |
| --- | --- |
| 硬件 | 单机 8 × NVIDIA H200，BF16，PyTorch DDP |
| 模型 | dense decoder-only，含 embedding 和输出头合计不超过 16 亿参数 |
| 数据 | 教师发布的共同训练池和 tokenizer，记录版本及预处理方法 |
| 训练 | 2048 tokens/样本，每卡 micro-batch 4，累积 4 次，2000 optimizer steps |
| 预算 | 每次正式训练最多 524,288,000 个模型输入 token，改 batch 或步数时重新计算 |
| 基线 | AdamW，峰值学习率 `3e-4`，5% warmup + cosine，seed 42 |

H200 SXM 每卡 141 GB。基线按单机数据并行设计，每卡各持有一份模型。开课前先跑 20 步记录峰值显存与稳态 tokens/s，用实测吞吐决定正式机时。首次小跑可用 `--steps 20 --micro-batch 1 --accumulation 1`。

基线由课程重新限定数据、预算与盲评目标。建议选一个有因果假设的改进，用一次同预算对照说明效果，例如调整残差或注意力、改变样本去重与混合比例、调整优化器与学习率。

[Ai2 官方配置](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json) · [OLMo 2 训练配置](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml) · [NVIDIA H200 规格](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)

## 2. 启动基线并生成一份完整 TXT

使用有 CUDA 版 PyTorch 的课程环境。基线接口按 Transformers `4.57.1`、Datasets `4.4.1`、Accelerate `1.11.0` 编写。`COURSE_ASSETS` 指向老师发放的 `config.json` 和 tokenizer 目录。`COURSE_TRAIN` 指向老师发放的 `datasets.save_to_disk` 目录，其中每条 `input_ids` 恰好 2048 个 token。

```bash
python -m pip install 'transformers==4.57.1' 'datasets==4.4.1' 'accelerate==1.11.0'

export COURSE_ASSETS=/path/from-instructor/olmo2-assets
export COURSE_TRAIN=/path/from-instructor/course-train-v1
cp course/run-header.txt run-header.txt

LOG=kernel-studio__model-base__data-base__params-base__run-001.txt
python course/run-log.py --header run-header.txt --source course --output "$LOG" -- \
  torchrun --standalone --nproc-per-node=8 course/main.py \
    --assets "$COURSE_ASSETS" --train "$COURSE_TRAIN" --out model \
    --steps 2000 --sequence-length 2048 --micro-batch 4 --accumulation 4
```

用自己的队伍 slug 替换 `kernel-studio`。改模型、数据或参数时，更新 `run-header.txt` 和文件名。`Title` 写一句核心改进，例如 `Grouped-query attention + 语料去重`。紧接着在 `Description` 分别写模型、数据和参数相对基线的变化，续行缩进。

文件名格式为 `队伍__model-改动__data-改动__params-改动__run-001.txt`。每段用英文小写、数字和单连字符，沿用基线填 `base`。

采集脚本依次写入日志头、实际命令、`nvidia-smi`、全量项目源码、训练标准输出和标准错误、退出码。`[Code]` 每行一个 JSON 对象，`path` 和 `contents` 可逐文件恢复。把 `--source` 指向包含本次运行全部项目源码和配置的目录，其中只放用于复现的 UTF-8 文本。token、环境变量文件、数据和权重放在该目录之外。若采用其他训练器，只需在每次 optimizer update 后由 rank 0 输出一行 `step:1/2000 train_loss:3.123456`，全程编号连续，且与日志头 `Parameters.steps` 一致。loss 用该 optimizer step 的批次平均，包含累积的 micro-batch 和各 rank。调试时也可以使用同一个采集脚本，正式提交按课程发布的预算执行。

[PyTorch `torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) · [NVIDIA `nvidia-smi`](https://docs.nvidia.com/deploy/nvidia-smi/) · [日志头模板](course/run-header.txt)

## 3. 在平台预检并交付

1. 打开 [Model Lab](https://maskyuanzh.github.io/course-leaderboard/)，进入 **提交日志**，选本次 `.txt`，点 **校验日志**。校验会显示标题、描述和连续记录的训练步数。可点 **载入示例** 试用。
2. 按课程通知，在课程共享存储的队伍目录下创建一个提交目录。把 **预检通过的原始 TXT** 和同一次运行的 **Hugging Face checkpoint** 一起放入。每队指定一个最终版本，由课程组截止后锁定。

   ```text
   submissions/<team-slug>/run-001/
     <team-slug>__model-base__data-base__params-base__run-001.txt
     model/
       config.json
       model.safetensors   # 或所有分片与索引
       tokenizer.json
       tokenizer_config.json
   ```

3. 确保课程组可读取目录，并在课程通知指定的渠道登记该路径和短标题。网页是浏览器格式预检，队伍登记和正式文件以课程侧收件记录为准。若使用训练框架原生分片，先导出成老师可以直接加载的 `safetensors`。自定义架构提前与老师确认加载方案。

## 4. 教师盲评与成绩

课程组从若干常见 LLM 评测中抽题，使用固定题目、固定选择顺序和同一 tokenizer，在教师环境加载每队指定 checkpoint。每道题读取各选项字母的模型概率，在该题选项内归一化，取正确选项的负对数概率。先分别计算每个来源的题目平均，再对来源等权平均，得到 Test Loss。提交阶段发放共同训练池与调试样例。正式排名在截止后的同一评测轮次统一公布。

提交使用课程允许的训练数据和随机初始化的模型。训练和调参依据训练数据及其公开的开发集。最终日志记录方法和训练过程，榜单统计教师侧实测成绩。

## 5. 实训报告

提交一页说明，包含实验假设、相对基线的变化、同预算对照结果、运行成本和复现命令。日志 `Title` 与 `Description` 对应最终 checkpoint。
