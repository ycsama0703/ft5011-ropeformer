# 改动方案：预测占位 token（方案 A）

状态：**提案，尚未实现**。需要小组确认后再改代码。本文件中的结论都是设计预期，不是实验结果。

## 1. 为什么要改

当前骨架为：历史 token → encoder → mean/last pooling → 线性头一次输出 H 步。
RoPE 只作用于历史窗口内部，未来各步没有位置，预测的相位只能从池化向量间接推出。
已做的合成实验显示：所有可学习 log-scale 仅移动到 0.7–1.2 倍；各模式差距小于种子波动。

文献给出一致解释：

- LeRoPE：无周期结构的相对位移上，频率梯度的各项互相抵消，期望接近 0。
- ElasTST：预测窗口短时，可调 RoPE 学不到有效周期系数；其设计让未来位置直接参与注意力。
- Why Attention Fails：PatchTST 中把注意力换成零或单位矩阵，误差不升反降，时序 Transformer 常退化为 MLP。

因此本提案让**未来每一步都拥有 RoPE 位置，并通过注意力直接读取历史**。这时位置频率决定
"未来第 h 步应该看历史中哪些相对距离"，成为外推机制的一部分，分组假设才有可检验的作用路径。

## 2. 模型改动

### 2.1 序列构造

```
tokens:   [ h_0 ... h_{L-1} | m m ... m ]        位置 0 .. L+H-1
           历史：Linear(x_t)    H 个占位 token：同一个可学习向量 m（d_model 维）
```

- 占位 token 不用"数值 0 经输入投影"：标准化后 0 是合法观测值，会与真实数据混淆。
  使用独立的可学习向量 m，所有占位 token 共享，彼此只靠 RoPE 位置区分。
- 不加任何绝对位置编码。RoPE 仍是唯一的位置来源。

### 2.2 注意力掩码

```
                key: 历史(L)          占位(H)
query 历史      全可见（causal=True 时下三角）   不可见
query 占位      全可见                          仅自身（对角）
```

- 历史 token 不看占位 token：历史表示与 H 无关。
- 占位 token 只看历史和自己，不看其他占位：第 h 步的预测与总预测长度 H 无关。
  这使"训练 H=96、测试 H=336"成为严格定义的外推，而不是分布改变。
- 不存在未来数据泄露：占位 token 不包含任何未来观测。
- 备选（需小组决定，见 §6）：占位之间相互可见。表达力更强，但失去 H 无关性。

### 2.3 输出头

- 取最后一层占位 token 的输出，经 output norm 后用逐 token 共享的 `Linear(d_model, output_dim)`。
- 参数量与 H 无关，推理时可传入不同的 horizon。

### 2.4 RoPE 部分

- GroupedRotaryEmbedding 数学不变：仍按 `position * base_frequency * exp(log_scale)` 直接计算，
  位置 0..L+H-1，无缓存插值。分组、初始化、per-layer 参数完全不变。
- 五种 rope_mode 共用同一新骨架。另加诊断模式 `rope_mode="none"`（NoPE，0 参数），
  明确作为诊断对照，不属于主比较。预期 NoPE 下所有占位 token 输出相同、预测在 horizon 上为常数，
  可直观说明位置在新骨架中的作用。

### 2.5 配置与 API

```python
ForecastConfig(
    ...,
    forecast_head="horizon_tokens",   # 新增："horizon_tokens" | "pooled"
    pooling=None,                     # 仅 pooled 可设 "mean"/"last"；None 表示 pooled 时用 mean
    rope_mode="grouped",              # 新增可选值 "none"（诊断用）
)
model(x)                    # [B, L, C] -> [B, cfg.horizon, out]
model(x, horizon=336)       # 仅 horizon_tokens 允许；用于预测长度外推
```

- 保留 `pooled` 头作为消融：同一 RoPE 模式在两种头下的差别，回答"RoPE 是否只在参与外推时才重要"。
- horizon_tokens 下设置 pooling、或 pooled 下传 horizon 参数，均报错，不静默忽略。
- 默认 head 改为 horizon_tokens 与否需小组决定（§6）。
- `return_attention=True` 返回含掩码后的 [B, heads, L+H, L+H] 注意力图；SDPA 路径与显式路径用同一掩码。

## 3. 必须保留或新增的测试

保留：初始等价于标准 RoPE、tied group 退化为 per-head、范数保持、相对位置性质、梯度有限、
保存恢复、配置校验、CPU autocast、pooled 头原有行为。

新增：

1. **H 无关性**：同一输入下，`horizon=H1` 的预测等于 `horizon=H2 (>H1)` 预测的前 H1 步。
2. **历史不受占位影响**：中间层历史 token 表示与 H 无关。
3. **掩码正确**：显式注意力图中，历史→占位、占位→其他占位的权重全为 0；causal 时历史部分为下三角。
4. **SDPA 与显式路径一致**（带掩码）。
5. **NoPE 诊断**：rope_mode="none" 时各占位输出相同。
6. **参数计数**：horizon_tokens 头的参数量与 horizon 无关；各模式只在 log_scales 上不同。

## 4. 训练与实验协议调整

- 公平性不变：所有模式同一骨架、同一初始化种子、同一 batch 顺序、同一步数和 base 搜索预算。
- base 网格改为 {1000, 10000}（ElasTST 报告 1–1000 的周期范围更适合时序；我们的合成实验也多选 1000）。
- 学习率：warmup + cosine；log_scales 单独参数组、无 weight decay。是否给 log_scales 更大学习率作为
  单独记录的设置，所有可学习模式一致（§6）。
- 记录 log-scale 轨迹（每次验证时保存），用于判断尺度是否真正移动。

## 5. 实施步骤

| 步骤 | 内容 | 产出 |
| --- | --- | --- |
| 1 | model.py：占位 token、掩码、逐 token 头、`horizon` 参数、`none` 模式；测试 | 单元测试全过 |
| 2 | 同步 README、docs/implementation.md、tools、examples（AGENTS 要求 API 改动一起更新） | 文档一致 |
| 3 | 合成实验：random_period、regime_switch；先比较 standard 在 pooled vs horizon_tokens，再比较四种模式；记录尺度轨迹 | 结果表 + 尺度轨迹 |
| 4 | 预测长度外推：训练 H=48，测试 H=48/96/144 | 外推曲线 |
| 5 | 诊断：注意力置零/单位矩阵、逐频带置零（参考 LeRoPE 的 band importance） | 分析章节 |
| 6 | 真实数据（数据集另定），沿用上述协议 | 正式结果 |

步骤 1–2 约半天；步骤 3–4 每轮约 1–3 小时 GPU。先看步骤 3 中 horizon_tokens 是否让尺度明显移动、
模式间差距是否超过种子波动，再决定是否投入真实数据。

## 6. 需要小组决定的问题

1. 默认 head：horizon_tokens（推荐）还是保留 pooled 为默认？
2. 占位 token 之间是否相互可见？（推荐不可见，保留 H 无关性与严格外推）
3. 是否加入 NoPE 诊断模式？（推荐加，仅作诊断）
4. log_scales 是否使用更大的独立学习率？（推荐先不加，若尺度仍不动再作为单独记录的实验）
5. 是否同时引入 patch token？（推荐暂不引入：会改变位置单位，日周期 24 步在 patch=16 下变成 1.5 个 token，
   与当前机制分析混在一起；可作为后续独立实验）

## 7. 风险

- 改进可能来自新头本身而非 RoPE：用 pooled vs horizon_tokens、NoPE 对照区分。
- 外推到训练中未见过的相对距离（> L+H_train）可能失败；这正是步骤 4 要测量的，不预设结论。
- 序列变长（L+H）使注意力成本增加；L=336、H=96 时为 432 个 token，GPU 上可接受。
- 文献依据中 LeRoPE、AdaRoPE 为 2026 年预印本，结论来自语言模型；ElasTST 对 tunable vs fixed RoPE
  的比较我们只看到图和文字描述，没有核对数值。
