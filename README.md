# FPC 四参考动作识别代码与结果

本仓库收录四参考最佳配置的代码、锁定参数、汇总结果和验证记录。每套参考各做一次六选一前向，使用冻结 StartLux-Decision-v1.1-4B-AutoThink、think_mode=never，四套概率经温度 0.9276 调整后等权融合，再做阈值及离线软时序解码。

四套参考为 V1、V2、V3、V6，每类参考三帧。Query 每 0.5 秒判断一次，输入两个历史上下文面板加三个目标面板。参考图的 BG 包含两处动作间过渡和一处循环间背景。四条 Demo 来源录像额外提供 320 个校准窗，分别用于确定阈值和软时序参数。

| 评测范围 | 窗口数 | Accuracy | 动作 Macro-F1 | 动作召回 | BG 召回 | 去 BG F1@50 |
|---|---:|---:|---:|---:|---:|---:|
| 六条 Query | 4,786 | 66.42% | 68.96% | 76.07% | 58.32% | 71.80 |
| V4 录像隔离子集 | 799 | 66.96% | 71.95% | 74.11% | 60.19% | 79.76 |

六条完整 Query 包含同一次录制的不同片段及相同外观组，数据审计核查到的重复窗口为 0。六条结果用于跨片段比较。V4 为录像来源独立子集，操作者身份仍待确认。历史 Query 已反复用于探索性研究，监督预算需同时统计参考和校准标签。

锁定阈值按 A01–A05、BG 排列为 [0.10, 0.35, 0.20, 0.45, 0.35, 0.20]，切换惩罚为 0，顺序惩罚为 0.5。

## 查看结果

results/four_worker_full_query_v1/REPORT.md 是六条 Query 主报告，evaluation_summary.json 含各参考组合的汇总指标和混淆矩阵。
results/four_worker_references_v1/ 保存 V4 子集结果。
config/LOCK.json 保存全部参考组合的校准参数，config/PROMPT.json 是准确的模型输入提示模板。

## 运行代码

run.py 提供可迁移的拼图重建、GPU 推理、Support 重校准和缓存复核入口。numerics.py 从原脚本原样提取数值函数。original_source/ 保留实验原代码，便于审查数据拆分、参考选择和评分来源。vendor/ 是带 Apache-2.0 许可的 StartLux 提示格式工具。

本次上传范围为代码和汇总结果。工业参考图、逐窗标注、输入清单、概率缓存、模型权重由有权访问者在本地准备。可将本地完整数据包指定为 --data。该目录采用 cache/、inputs/、evaluation_labels/、references/、config/、results/ 布局，具体读取格式见 run.py。

安装 CPU 依赖并复核缓存，运行
```bash
python -m pip install -r requirements.txt
python run.py --data /path/to/private-data replay
python run.py --data /path/to/private-data calibrate
```

重建已锁定帧清单的拼图并重新推理，运行
```bash
python run.py --data /path/to/private-data build-sheets --split query --frames /path/to/ffpmeg384fps25
python -m pip install -r requirements-inference.txt
CUDA_VISIBLE_DEVICES=0 python run.py --data /path/to/private-data infer --split query --model /path/to/StartLux-Decision-v1.1-4B-AutoThink
```

使用多卡时指定相同 --world N 与不同 --rank。推理输出支持按响应 ID 断点续跑，默认写入 output/inference/。
输入拼图和参考图均校验 SHA-256。缓存复核会与归档逐窗预测一致性及汇总指标比较。

## 验证范围

使用本地完整素材执行了以下检查，记录在 verification/EXPORT_VALIDATION.json。

- 21 组完整 Query 预测重放一致，独立重算分类、混淆及分段指标一致。
- 七组参考组合的阈值和软时序参数重算一致。
- 320 个 Support 拼图和 4,786 个 Query 拼图重建后逐一通过归档 JPEG SHA-256 校验。
- 四套参考的模型提示与原推理入口逐字一致。

本次打包新增的便携 GPU 推理入口经过提示和输入重建验证；GPU 神经前向结果采用已验证的原实验缓存。当前导出时环境及历史日志中的 vLLM 版本保存在 provenance/environment.json。模型权重摘要保存在 provenance/model_weight_sha256.json。StartLux 权重按原模型许可单独获取。
