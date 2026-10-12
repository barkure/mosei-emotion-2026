# common —— 共享层

被两个以上任务引用的代码放在这里；问题一/二/三的脚本都通过 `src/` 引导路径导入本目录。

| 文件 | 作用 | 产物 |
| --- | --- | --- |
| `prepare_resources.py` | 下载并校验固定版本公开模型（`pixi run resources`） | `models/` |
| `data_io.py` | 附件 2 两版（aligned／unaligned）统一读取：结构掩码、观测掩码、声明／结构长度、越界统计 | 供其它脚本 import |
| `audit_data.py` | 附件 2 对齐版各划分 + 附件 3／4 两版文件数与字段结构 | `outputs/data_audit/data_audit.json` |
| `audit_attachment2.py` | 附件 2 两版 train/valid 逐样本审计（ID、形状、长度、非有限值、标签对应） | `outputs/data_audit/attachment2_audit.json` |
| `verify_data_io.py` | 统一读取接口的两版抽样验证 | 终端输出，可重定向保存日志 |
| `prep.py` | 问题二共享工具：`BERT`／`DATA` 常量、`make_split`、`slice_batch`、`train_normalizer`、`apply_missing`、`augment`、`missing_span`、`seed_all`、`consistent_strength`（强度符号一致性投影）；包含统计融合对照模型训练入口 | 供问题二／三 import |
| `encoders.py` | `AVEncoder`（掩码感知的音视频编码器）；包含时序融合对照模型训练入口 | 供问题二 import |

运行入口：

```bash
pixi run audit                # → outputs/data_audit/data_audit.json
pixi run audit-attachment2    # → outputs/data_audit/attachment2_audit.json
pixi run verify-data-io       # 在终端显示检查结果
```

统计融合与时序融合的训练入口分别为`prep.py`、`encoders.py`，运行时会创建 `outputs/q2_prediction/comparisons/` 保存新结果。正式三种子模型及已归档的对照汇总位于 `outputs/q2_prediction/q2_systematic/`；对照证据范围见[问题二方法说明](../../docs/问题二模型与实验.md)。

## 数据接口与掩码

附件二两版均为train=3395、valid=728、test=727，读取方式为 `data[split][field][index]`。

| 数据 | 文本 | 音频/视频 | 说明 |
| --- | --- | --- | --- |
| 附件二aligned | text 50×768、text_bert 3×50 | 50×74 / 50×35 | 正式训练与验证版本 |
| 附件二unaligned | 同上 | 500×74 / 500×35 | 有独立音视频长度 |
| 附件三aligned | 仅text_bert，无text | 50×74 / 50×35 | 无标签及id，以文件编号01—30关联结果 |
| 附件三unaligned | 仅raw_text | 500×74 / 500×35 | 缺少词元与音视频有效长度字段 |
| 附件四aligned | raw_text、text、text_bert | 50×74 / 50×35 | 有id与01—20原视频 |
| 附件四unaligned | 同上 | 500×74 / 500×35 | 另有音视频有效长度 |

类别编码0=负、1=中、2=正；强度在[-3,3]，中性严格为0。附件一自提取维度为文本768、语音17、面部52；它与附件二特征定义不同，尚未验证直接迁移预测模型。

### 掩码与时间契约

- `text_bert`三路为词元编号、注意力掩码和分段编号。结构掩码含CLS/SEP，观测掩码只包含内容词元。填充位置的text向量可能非零，不能据其数值判定有效长度。
- aligned音视频结构长度沿用文本注意力有效前缀，首尾特殊词元音视频位置为零；内部全零位置标为未观测，成因单独解释。
- unaligned音频长度已核实自洽；视频保留原始声明长度，并以 `min(500, max(声明长度, 最后非零位置+1))` 作为保守结构长度。这是附件二train/valid上制定的读取规则，字段真实含义尚未查明。
- unaligned视频声明长度后仍有非零值：train 618/3395、valid 141/728；越界计数按原始声明长度保留，不因结构扩展归零。细节保存在数据审计JSON。
- 原始媒体时间来自解码音视频时基。对齐版特征索引不能按视频总时长均分成秒；文本词—子词映射本身不提供词起止时间。

公共读取模块 `src/common/data_io.py`，审计 `outputs/data_audit/attachment2_audit.json`；预测管线公共处理为 `src/common/prep.py`，编码器为 `src/common/encoders.py`。
