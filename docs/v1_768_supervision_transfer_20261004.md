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
