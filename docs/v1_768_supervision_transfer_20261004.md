# V1_768_SUPERVISION_TRANSFER_20261004

本段检验上一段指定的768监督迁移假设：在同一个已训练的v1 DEV视觉塔上，
重新由官方冻结768特征产生的train_dev目标/可靠度，能否改善原512监督的决策错误。
现役继续为v2平台74.89317380621728%，未经新候选平台实测严格超过不替换。
当前研究不重复A机同监督LoRA续训，不恢复已关闭阈值、增强或margin扫描。

## 结果前冻结协议

- 基点`origin/main@b312b72`；独立分支`codex/v1_768_supervision_transfer_20261004`，
  工作目录`/home/lux1/noise/worktrees/v1_768_supervision_transfer_20261004`。
- 只用当前20260921的133,815张train_dev构建新监督；14,880张val_dev只评价，
  内容组和路径均与训练隔离。官方冻结768缓存含全train像素前向，但teacher/kNN只索引train_dev，
  不读取full_train拟合的teacher/targets。原20轮teacher与全部去噪超参数不变。
- 冻结CRT已训练视觉塔及其448中心768特征；两头均从原父512头乘官方投影转置初始化。
  原512目标对照与新768目标候选各20轮，沿用CRT的AdamW、lr .001→.0001、WD .01、
  batch8192、dropout .1、LS .1、seed42；只改变targets/weights。
- 两臂共享旧/新正权重人口的并集、逐轮顺序和dropout RNG。零权重行保留在批次、损失贡献为零。
  因此重拟合对照头；旧CRT原来只采样旧正权重人口，直接复用会混入样本顺序差异。
  原CRT末轮头仍保留为第三个只读参照，不训练第三臂。
- 完整val无bias/448中心固定评价，冻结共同错误2,875张、尾75类和跨标签内容冲突代理；
  同时报告原标签micro/macro及每组修正/退化，不将代理当成干净真值。
- 推进复核门须同时满足：候选相对新配对控制全量净≥75、相对旧CRT净≥75、
  相对新控制共同错误净≥25、全量macro差≥0。否则关闭该固定监督迁移配方。
  通过也只支持复核下一步投入，不自动full、LoRA、第二seed或阈值扫描；平台收益未知。
- 目标构建上限600秒；拟合与交付上限5,400秒。真实4 batch/臂成本探针丢弃权重，
  完整拟合重新初始化；若成本不支持，停止不放宽。测试推理沿固定448/512/576 resize后crop448、
  原图及flip六视图logits求和、200步强度1均匀bias，无bias对照同时保留。
  每臂独立checkpoint和raw/bias CSV/ZIP，不融合两头预测。

资产复用、14,880张原CRT逐图重放和内容组隔离通过后才执行；所有输出使用独立目录。
新监督变更样本量只表示干预范围，不代表可修复的平台错误数。

## 入口与产物

固定配置：[目标构建YAML](../configs/v1_768_supervision_targets_20261004.yaml)、
[配对JSON](../configs/v1_768_supervision_transfer_20261004.json)。从方案目录执行：

```bash
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 scripts/probe_v1_768_supervision_transfer.py prepare \
  --config configs/v1_768_supervision_transfer_20261004.json
# 同一绑定下依次执行targets、fit、deliver，不能覆盖已生成资产。
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -m pytest tests/test_v1_768_supervision_transfer.py tests/test_v1_post768_jobs.py -q
```

输出根`outputs/codex/v1_768_supervision_transfer_20261004`，新目标在`targets768/targets.pt`；
两臂`control512/`、`candidate768/`各含`selected.pt`、`submission/`（bias）、`submission_raw/`。
实现与结果分开；没有正式delivery_report及独立核验时不能声称候选交付完成。

现役包：`/mnt/c/Users/lqh22/Desktop/noise_top4_20261003/01_v2_full_sixview_logit_sum_bias.zip`，
ZIP SHA `51e372c390f65702ef15af82b7de5a6cc7a382cfaf6f54b7c44be418517aec49`；
37,444行/九项检查和CSV摘要来源为[现役核验](../results/v1_priority_20261004/validation.json)。

## 已核验的目标差异与独立DEV结果

协议在候选生成前提交为`ed3ad8f`。11项配对/边界测试通过；资产预检逐张重放原CRT
14,880条预测。首次目标构建在沙箱内因CUDA不可见而停止，没有生成目标；原日志保留。
随后获准在沙箱外使用本机CUDA，仍执行同一配置，没有更换设备或连接远端。

768目标构建内部计时13.22秒（不含前置资产读取/核验）：旧有效监督119,074条，新122,262条；
新增4,086、移除898、共享118,176，共同采样并集123,160。共享有效行中目标标签改变4条，
权重逐位不同110,178条；伪标签由78条变为173条。这些不是干净标注或可恢复错误数。

4 batch/臂成本检查0.882秒，权重丢弃；完整两臂从同一初始头重开。
20轮×123,160条样本的共享顺序、dropout RNG和初始化摘要相同；每臂320次AdamW更新。
拟合含成本检查及评价内部计时5.96秒。保守拟合预计70.54秒，加原CRT完整交付耗时2,495.61秒，
合计低于5,400秒成本门。以上内部计时不包含每阶段重复的输入读取和摘要校验。

| 同448中心/无bias/14,880张val | micro | macro | 正确数 |
|---|---:|---:|---:|
| 旧CRT普通768头（只读参照） | 76.5188% | 75.6027% | 11,386 |
| 本次512监督控制 | 76.4180% | 75.4773% | 11,371 |
| 本次768监督候选 | 76.6465% | 75.7192% | 11,405 |

候选相对同轨迹控制：修正151、退化117、净+34（micro+0.2285pp / macro+0.2419pp）；
相对旧CRT：修正165、退化146、净+19。新控制相对旧CRT净−15，说明新的公共采样顺序
本身足以改变逐图结果；不能用候选对旧CRT的差值替代严格配对差值。

| 冻结组 | 图数 | 候选对新控制：修正/退化/净 | 候选对旧CRT：修正/退化/净 |
|---|---:|---:|---:|
| 原四候选共同错误 | 2,875 | 44 / 10 / +34 | 54 / 0 / +54 |
| 尾75类 | 1,003 | 10 / 7 / +3 | 9 / 11 / −2 |
| 跨标签内容冲突 | 367 | 7 / 6 / +1 | 7 / 9 / −2 |

共同错误门和macro门通过；两个全量净≥75门均未通过。决定`close_fixed_supervision_transfer`，
不扩full/LoRA、不改采样、阈值或teacher设置继续追分。不把这次头部小幅提升外推为平台收益，
也不据该结果否定所有监督改进方法。

CPU独立核验已复算133,815条监督变化、20轮公共顺序、29,760条FP64头预测、12份群体指标及
12份配对，全部一致；30份源文件摘要通过。两份导出checkpoint的视觉参数逐张量与原父一致。
此处确认DEV结果；最终候选交付以独立提交核验记录为准。

实际计算命令（每条均在方案目录执行，环境相同）：

```bash
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -u scripts/probe_v1_768_supervision_transfer.py targets --config configs/v1_768_supervision_transfer_20261004.json
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -u scripts/probe_v1_768_supervision_transfer.py fit --config configs/v1_768_supervision_transfer_20261004.json
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -u scripts/probe_v1_768_supervision_transfer.py deliver --config configs/v1_768_supervision_transfer_20261004.json
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python3 scripts/verify_v1_768_supervision_transfer.py --config configs/v1_768_supervision_transfer_20261004.json \
  --validation-only --output outputs/codex/v1_768_supervision_transfer_20261004/validation_independent.json
```

## 最终交付与收尾

两臂各自raw/bias四包全部交付，正式九项校验、ZIP内外CSV字节、完整预测重放通过。
每臂64张真实图像冷加载与共享前向的logits最大误差均为0，预测与缓存相同；
固定200步bias的独立float64重拟合各37,444张硬判决全部一致。
推理及出包内部计时1,266.53秒，拟合加交付1,272.49秒，未超5,400秒成本门。
40项相关CPU测试通过；最终独立核验重算30份源摘要、133,815条监督、20轮×123,160条公共顺序、
29,760条FP64验证预测、12份群体指标与12份配对，并重跑四包正式提交检查。
平台分未知，现役v2包SHA未变，本固定配方关闭。

输出根绝对路径：
`/home/lux1/noise/worktrees/v1_768_supervision_transfer_20261004/outputs/codex/v1_768_supervision_transfer_20261004`。
下表目录内均有`pred_results.csv`和`submission.zip`；每个臂的checkpoint为其`selected.pt`。

| 相对输出根目录 | ZIP SHA256 | 行数/校验 |
|---|---|---|
| `control512/submission_raw` | `b97c72db522d6e0db18a4047265640fae09f4edd3e40694385726b79bba407d5` | 37,444 / 九项通过 |
| `control512/submission` | `6a5db487c7995c6fe96bdbe7dcc48048968c8226ed1d1c7a1f48691288a3ca20` | 37,444 / 九项通过 |
| `candidate768/submission_raw` | `9e1f20f40b5a32d3a7c251cdf32d1c87588adbe530147abf867033a94bcc0712` | 37,444 / 九项通过 |
| `candidate768/submission` | `11cd92fc2534ce85c25c1e3ad58762d8e1ca6b1e3fd18e762a13115a5ded45e4` | 37,444 / 九项通过 |

未复制到桌面或上传平台。四包为可复用的固定配对产物，不建议仅凭本地小涨自动占用平台名额。
结果：[监督差异](../results/v1_768_supervision_transfer_20261004/target_comparison.json)、
[DEV与推进门](../results/v1_768_supervision_transfer_20261004/fit_report.json)、
[提交与冷加载](../results/v1_768_supervision_transfer_20261004/delivery_report.json)、
[独立核验](../results/v1_768_supervision_transfer_20261004/independent_verification.json)。
最终核验命令：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python3 scripts/verify_v1_768_supervision_transfer.py --config configs/v1_768_supervision_transfer_20261004.json \
  --output outputs/codex/v1_768_supervision_transfer_20261004/independent_verification.json
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -m pytest tests/test_v1_768_supervision_transfer.py tests/test_v1_post768_jobs.py \
  reproducibility/aegis_f1/tests/test_v1.py reproducibility/aegis_f1/tests/test_v1_preprojection_test_bias.py -q
git diff --check
```

改动为两个固定配置、探针/独立核验脚本、配对测试、私有输出ignore、本记录、当前执行入口及聚合结果。
方案验证后立即commit/push；main采用自动`pull --rebase --autostash`模式同步，
合并后重新核验并push，在本检查点收尾。v1尚无平台实测超过现役的新候选，目标保持active。
