# V1_WFT_FIXED_BIAS_20261004

## 冻结协议

后续优化主线仍为 v1，提交基准保留 v2 **74.89317380621728%**。
本段只复用已完成的 WFT448 FULL 第4轮 last EMA，补齐同协议的固定先验解码比较。
无训练、无权重平均、无参数搜索，不恢复 WFT 训练队列。

可观察缺口：WFT 原 raw 包用户反馈 **70.43585087063347%**（26,374/37,444），
未使用后来既定的六视图 SUM + 测试均衡 bias；v1 512 full 在该解码下为
73.99049246875335%，768 full 为74.41512658903963%。解码不同使旧 raw 分数不能回答
WFT 在固定先验下的表现。WFT 需比自身旧 raw 净增至少1,670张才严格超过现役28,043张；
其他模型的 bias 增益不能直接外推给 WFT，本段不预言收益或把分布均衡当作正确率。
此前 WFT DEV 相对父模型 raw 净+87、FULL raw 平台反而−206的结论保留。
FULL 已用原 val 训练，禁止用旧 val 选择 checkpoint 或解码。

唯一候选：checkpoint SHA `9561fbb463d37c8ff576098fee04dc696aa71baa80f377c52fe40bb3bc8e9823`；
448/512/576短边缩放、中心裁剪448、各原图/翻转，FP32前向；按翻转对求和、再累加三尺度。
固定 uniform soft-mass bias 200次、strength=1、damping=1，不做硬配额。
同时输出 raw 控制。许可来自[既有官方确认转述](v1_test_prior_authorization_20261001.md)。
测试图只读，无标签、梯度更新或分布驱动的参数挑选。

停止条件：来源摘要变化、原六视图均值或新SUM的raw预测不等于旧37,444条、
64张冷加载逐视图复算不一致、独立FP64重拟合不一致、提交九项失败均拒绝交付。
本机CUDA仅空闲时启动，前向总预算45分钟；不自动重跑或扩训练。
只有未来平台实测严格超过74.89317380621728%才晋级；否则保留现役。

当前主线模型接口与原WFT归档不同，原计划的兼容性检查按设计拒绝跨版本加载。
推理使用原WFT代码，原计划 verify 不改、不绕过；解码使用现行固定bias代码。
配置绑定两侧源码及模型/计划/原raw/训练报告SHA，两个进程间通过带绑定的六视图缓存衔接。

## 路径与命令

分支 `codex/v1_wft_fixed_bias_20261004`，基线 `29c95d8`。
工作目录 `/home/lux1/noise/worktrees/v1_wft_fixed_bias_20261004`。
配置 [v1_wft_fixed_bias_20261004.json](../configs/v1_wft_fixed_bias_20261004.json)。
输出根为工作目录下 `outputs/codex/v1_wft_fixed_bias_20261004/`。

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export PYTHONPATH=/home/lux1/noise/worktrees/wft448_full_20261001/reproducibility/aegis_f1
python3 scripts/infer_v1_wft_fixed_bias.py preflight --config configs/v1_wft_fixed_bias_20261004.json
python3 scripts/infer_v1_wft_fixed_bias.py infer --config configs/v1_wft_fixed_bias_20261004.json
export PYTHONPATH=reproducibility/aegis_f1
python3 scripts/deliver_v1_wft_fixed_bias.py decode --config configs/v1_wft_fixed_bias_20261004.json --report outputs/codex/v1_wft_fixed_bias_20261004/delivery.json
python3 scripts/deliver_v1_wft_fixed_bias.py verify --config configs/v1_wft_fixed_bias_20261004.json --report outputs/codex/v1_wft_fixed_bias_20261004/independent_verification.json
python3 -m pytest tests/test_v1_continuation.py reproducibility/aegis_f1/tests/test_v1_preprojection_test_bias.py reproducibility/aegis_f1/tests/test_prior_alignment.py reproducibility/aegis_f1/tests/test_submission.py -q
```

现役桌面包 `/mnt/c/Users/lqh22/Desktop/noise_top4_20261003/01_v2_full_sixview_logit_sum_bias.zip`，
SHA `51e372c390f65702ef15af82b7de5a6cc7a382cfaf6f54b7c44be418517aec49`，
九项校验见 [平台登记验证](../results/top4_platform_20261004/validation.json)。

## 执行状态

协议已冻结，CPU预检进行中。暂无新候选结果或平台分数。
