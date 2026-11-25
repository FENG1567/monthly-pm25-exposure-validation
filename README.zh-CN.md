# 跨国家 PM₂.₅ 暴露验证：复现包

本仓库为论文 **“Limits of monthly PM₂.₅ exposure assignment across space and time and implications for gallbladder and bile duct disease research”** 整理代码与作者自有汇总结果。

**作者：** Shuo Feng、Chao Shi、Tianming Liu、Haoran Ding、Fenghe Hang、Zhenghao Huang、Xianzhi Meng。

公开仓库：https://github.com/FENG1567/monthly-pm25-exposure-validation 。对应论文的软件发布版本为 v0.1.0，尚未分配 DOI。

## 包含内容

- 带发布校验值的模型和辅助分析代码快照，以及便携式输入适配器。
- 作者自有的汇总结果：21 张来源汇总表、17 张论文表格，以及供 14 张图使用的 35 个 panel CSV。补充图 S7 以无损压缩 CSV 提供，约 19 MB，包含全部 7,716,800 行。
- 使用已保存汇总结果复现表格和绘制图片的代码。

本包验证 PM₂.₅ 暴露估计，不含胆道或胰腺疾病结局数据，也不含疾病关联估计代码。

## 快速开始

在仓库根目录安装复现依赖并运行一级复现：

```bash
python -m pip install -r requirements-replay.txt
PYTHONPATH=src python -m pm25_repro all
```

Windows PowerShell 用户先设置源码路径：

```powershell
python -m pip install -r requirements-replay.txt
$env:PYTHONPATH = "src"
python -m pm25_repro all
```

该命令验证文件校验和及随包汇总结果检查，重新生成参考表格并绘制图表到 `results/generated/`。这是对已保存汇总结果的确定性复现，不会重新拟合模型、执行站点 bootstrap 或独立验证。复现层级和限制见[复现说明](docs/REPRODUCIBILITY.md)。

## 使用外部输入重新运行分析

便携式适配器支持历史验证、葡萄牙验证、未来年份评估和留一国家外推（LOCO）。运行需要用户明确提供输入文件和 JSON 配置。先检查命令计划，再添加 `--execute` 执行归档 runner：

```bash
PYTHONPATH=src python -m pm25_repro analysis --config configs/your-run.json --component historical
PYTHONPATH=src python -m pm25_repro analysis --config configs/your-run.json --component historical --execute
```

`configs/your-run.json` 是由使用者根据自己的输入创建的配置占位路径；这不表示包内含外部数据。归档中还保留站点稀疏化、多重性分析、月度比较和葡萄牙尾部诊断脚本，它们各自需要相应输入。详见[复现说明](docs/REPRODUCIBILITY.md)。

## 新增来源核验

EPA 2018—2025年原始ZIP到日/月目标值的处理链已重建核对；重放入口需要显式提供原始文件和保存目标表，见[EPA来源说明](docs/EPA_LINEAGE.md)。新增的年度指标表和[年度导出核验](docs/ANNUAL_EXPORT_CHAIN.md)支持全部704项对比的指标对应和种子检查。上述核验未重新训练模型或重新计算年度bootstrap区间；剩余来源缺口仍见复现说明。

## 数据与许可

原始项目代码（包括归档代码快照和便携式适配器）采用 MIT 许可。作者自有的汇总结果、参考表格和图片采用 CC BY 4.0。第三方源数据仍受各自条款约束，以上许可不会重新许可第三方数据。本包不含第三方原始数据。详见 [LICENSE-DATA.md](LICENSE-DATA.md) 和[第三方数据说明](docs/THIRD_PARTY_DATA.md)。

## 引用

论文正式发表后，请按最终出版记录引用论文，并在适当情况下引用本软件。仓库地址和软件版本用于标识本软件，论文尚无正式出版记录，软件尚未分配 DOI。作者和软件元数据另见 [CITATION.cff](CITATION.cff)。

14幅图已采用大写18号Arial加粗标签与全图粗体文字，并提供600 dpi TIFF；字体配置及输出说明见[图件版式说明](docs/FIGURE_LAYOUT.md)。
