# 论文绘图

在项目根目录运行：

```powershell
pixi run python src/figures/reproduce_figures.py --all
```

PDF 输出到 `outputs/figures/`；逐图代码、输入与命令见[图表来源](../../docs/图表来源.csv)。

安装 SimSun 和 Times New Roman，或使用 `paper/fonts/` 中对应字体文件。涉及原视频截图的图需要原视频和 FFmpeg。图 1.1 的 TikZ 源码位于 `paper/figures/总体技术路线.tex`。

单独修改绘图后，同步图库：

```powershell
pixi run python src/figures/sync_paper_gallery.py
```
