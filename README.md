# GroupedRoPE

**面向时间序列预测的分组可学习旋转位置编码。** 这是 Deep Learning 课程小组项目，目标是做清楚一个模型改动及其对照实验，不要求绝对创新，也不预设一定提升预测性能。

> 给组员和 AI 助手：先读本 README，再读 [AGENTS.md](AGENTS.md)、[实验方案](docs/experiments.md)，最后看 [核心实现](src/grouped_rope/model.py)。当前已有模型、数学性质测试和合成训练示例；真实数据实验尚未实现。不要把示例输出写成方法有效的证据。

## 1. 我们研究什么

时间序列可能同时有较快变化和较慢变化。Transformer 的多个注意力头提供并行建模途径，但标准 RoPE 给每个 head 相同的固定位置频率表。

原方案是每个 head 学一个缩放参数，把该 head 的全部位置频率一起调快或调慢。现在的问题是：**是否应该让同一个 head 的不同频率组分别调整，而不是一起缩放？**

主方法将每个 head 的 RoPE 坐标对按基础频率顺序分为两组，每组学习一个正值缩放。这样，高频组和低频组可以分别调整。每层 H 个 head 只新增 2H 个参数。

这里的高低频是**位置旋转频率**。输入没有做 FFT，没有被分解或过滤，head 也不是天然的带通滤波器。预测对象是通用连续时间序列；项目不再绑定原油数据、波动率或涨跌分类。

## 2. 直觉：为什么不是每头一个尺度就够了

假设某 head 有两个基础位置频率 (0.1, 1.0)。如果希望它们变为 (0.1, 0.5)，一个共同乘数无法实现：保留第一个频率需要乘 1，调整第二个需要乘 0.5。

按组缩放放宽了这一限制。它允许不同频段作不同调整，同时通过组内共享控制参数量。我们的假设是：这种额外自由度可能有助于拟合任务需要的时间依赖。

它不意味着标准 Transformer 完全不能捕捉多尺度结构，也不意味着更多频率参数必然更好。我们用消融实验检验这一假设。

## 3. 数学定义

设层数为 L，head 数为 H，每头维度为 d_h，旋转坐标对数量为 m=d_h/2。标准基础频率为：

$$\omega_j=b^{-j/m},\qquad j=0,\ldots,m-1.$$

频率按 j 递减：**较小的 j 对应较高频率**。默认 G=2，用连续、尽量等长的坐标对分组，前一组高频，后一组低频；若不能整除，组大小相差至多一。组归属 g(j) 固定，不参与学习。

### 原方法：每头统一缩放

$$\widehat\omega_{\ell,h,j}=\omega_j\exp(a_{\ell,h}).$$

### 主方法：每头分组缩放

$$\widehat\omega_{\ell,h,j}=\omega_j\exp(a_{\ell,h,g(j)}).$$

所有 a 默认初始化为 0，因此各可学习版本从标准 RoPE 开始。每一层独立拥有自己的参数；不是全模型只共享一套。学习的是 log-scale，用指数得到正缩放。

将一对坐标看成复数，位置旋转写为：

$$\widetilde q_{t,h,j}=q_{t,h,j}e^{i t\widehat\omega_{\ell,h,j}},\qquad
\widetilde k_{u,h,j}=k_{u,h,j}e^{i u\widehat\omega_{\ell,h,j}}.$$

实际实现使用实数 sin/cos，与半拆分配对约定：第 j 个通道与第 j+m 个通道配对，而不是相邻通道配对。Q、K 都旋转，V 不旋转。直接计算角度，不从有限缓存插值。

## 4. 理论依据与可说到的程度

### 欧拉公式与相对位置

欧拉公式 e^(i theta)=cos(theta)+i sin(theta) 将二维旋转与复指数联系起来。RoPE 的注意力内积（略去缩放因子）为：

$$\operatorname{Re}\sum_j q_{t,h,j}\overline{k_{u,h,j}}\,
 e^{i\widehat\omega_{\ell,h,j}(t-u)}.$$

因此频率控制位置项随时间距离的变化。系数本身也依赖输入内容，所以这不是普通傅里叶变换或固定卷积核。

### 原参数化的约束

每头只有一个尺度时，任意两个频率的比例不变：

$$\widehat\omega_{h,j}/\widehat\omega_{h,k}=\omega_j/\omega_k.$$

分组允许跨组比例改变。当所有组的 a 相等时，新方法严格退化为原方法。这证明新的位置参数化包含旧参数化，不证明测试误差必然下降。

### 保留的性质

在同一层、同一 head 内，频率固定于模型参数，不随位置或输入变化，因此旋转仍满足：

$$R(t)^\top R(u)=R(u-t).$$

R 为正交矩阵，故旋转保持 Q/K 向量范数。新增可学习参数为 LHG，不改变注意力 O(T^2) 的复杂度阶数；具体耗时仍需测量。

### 文献背书

- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)：相对位置旋转的数学基础。
- [LeRoPE，2026 预印本](https://arxiv.org/html/2607.10134v1)：直接学习频率缩放、log-space 参数化及其优化分析。我们的分组方法是更低自由度的借鉴，不是该论文的原样复现。该文研究语言模型，不直接证明时间序列收益；其独立层/head 参数也未显示明确优势。
- [TimesNet](https://arxiv.org/abs/2210.02186)、[Pathformer](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2be6705de7412adf107900add727a795-Abstract-Conference.html)：多周期和多尺度时序建模背景，不是当前代码的组成部分。

我们没有理论保证“学出的周期就是信号周期”“分组必然改善性能”或“各 head 必然分工”。这些都不能当成既定结论。

## 5. 代码支持的对照

| rope_mode | 每层可学习尺度数 | 含义 |
| --- | --- | --- |
| none | 0 | 诊断用 NoPE：不旋转，不属于主比较 |
| standard | 0 | 标准固定 RoPE |
| fixed_head | 0 | 固定 per-head 尺度，手动给初值 |
| head | H | 原方案，每头一个可学习尺度 |
| shared_group | G | 分组尺度在同层各 head 间共享 |
| grouped | H x G | 主方法，每头每组分别学习 |

各模式使用同一套编码器、输入投影、预测头，只在 log_scales 上不同。G=1 时 grouped 等价于 head。G>1 的组在训练后可能交叉，high/low 只表示初始基础频率分组，不保证有效频率始终有序。

默认所有尺度为 1，包括 fixed_head。因此非平凡固定对照应显式使用 `head_scale_init=(0.5, 0.8, 1.25, 2.0)`（四个 head）。该初始化只接受 head/fixed_head；不要把固定对照与学习对照的初值混为一谈。

### 预测头

| forecast_head | 结构 | 角色 |
| --- | --- | --- |
| horizon_tokens（默认） | 历史后追加 H 个共享的可学习占位 token（位置 L..L+H-1），逐 token 线性读出 | 主骨架：RoPE 直接决定未来各步看历史的哪些相对距离 |
| pooled | mean/last pooling 后一次性线性输出 H 步 | 消融：RoPE 只作用于历史内部 |

horizon_tokens 的注意力掩码：历史只看历史（causal=True 时为下三角）；占位只看全部历史和自身，
不看其他占位。因此历史表示与 H 无关，第 h 步预测与总预测长度无关，`model(x, horizon=H')`
可直接用于预测长度外推。占位不含任何未来观测，没有标签泄露。设计动机与依据见
[docs/plan_horizon_tokens.md](docs/plan_horizon_tokens.md)。

## 6. 安装与快速验证

Python >= 3.10，PyTorch >= 2.0。可在独立虚拟环境中运行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python tools/smoke_test.py
python examples/train_synthetic.py --mode grouped --steps 100
```

已有 PyTorch 时，脚本会自动导入本目录 src，可直接运行，不必先安装项目。CI 文件已配置，但远程 GitHub Actions 尚未运行。

## 7. 使用模型

```python
import torch
from grouped_rope import GroupedRoPEForecaster

model = GroupedRoPEForecaster(
    input_dim=7,
    horizon=24,
    d_model=64,
    n_heads=4,
    num_layers=2,
    d_ff=128,
    rope_mode="grouped",
    n_freq_groups=2,
)
x = torch.randn(8, 96, 7)
target = torch.randn(8, 24, 7)  # 仅演示形状；实际使用真实未来标签
forecast = model(x)
loss = (forecast - target).square().mean()
loss.backward()
```

接口为 `[batch, history, input_dim] -> [batch, horizon, output_dim]`。output_dim 默认等于 input_dim；可显式配置其他输出维度，调用者负责正确构造标签。

默认 forecast_head="horizon_tokens"；`model(x, horizon=96)` 可在推理时改变预测长度。pooled 头需显式设置 `forecast_head="pooled"`，可选 `pooling="mean"`（默认）或 `"last"`；在 horizon_tokens 下设置 pooling 会报错。历史部分默认不使用因果 mask，因为输入全部是预测起点之前已知的历史；可设置 causal=True 做因果对照，所有实验应保持该设置一致。无独立 decoder、外生 cross-attention、方向或分布预测头。

输入/目标的归一化与反归一化由外部数据流程负责，模型内部不偷偷改变目标定义。

```python
model.eval()
with torch.no_grad():
    forecast, maps = model(x, return_attention=True)
    info = model.rope_diagnostics()
# maps: 每层一个 [batch, heads, T, T]，已 detach；horizon_tokens 下 T=history+horizon，pooled 下 T=history
# info: 每层的 scales、group_index、frequencies、periods
```

正常前向使用 PyTorch SDPA，不存储 attention maps；显式请求时计算并返回 dropout 前的图。报告的 period=2*pi/frequency，以输入采样步为单位，不等于估计出的真实周期。

## 8. 训练约定与当前限制

- 通用预测从 MSE loss 开始，报告 MSE/MAE。先控制骨架不变，再比较 RoPE 模式。
- 建议 log_scales 单独分 optimizer group，不加 weight decay；合成训练示例已这样处理。默认不硬性限制尺度；监测其增长、梯度和有效频率，必要时再做有记录的约束实验。
- 指数参数化保证正值，不保证优化永远稳定；过大频率还可能产生离散采样混叠。
- 不默认添加尺度分离损失。尺度相似可能是合理解，不能为了好看的图强迫 head 分工。
- 未引入 FFT 条件网络、动态尺度、可学习分组边界或滤波器。不要把讨论过的扩展说成已实现。
- 单元测试在 CPU 上运行；合成实验脚本在本机 CUDA GPU 上跑过，但没有 GPU 单元测试或 CI 结果，也没有真实数据预测结果。

合成示例生成独立的训练/验证/测试序列，用验证集选择最佳状态，最后只评估测试集。不用于方法优劣结论：周期固定且任务较简单。

```bash
python examples/train_synthetic.py --checkpoint results/demo.pt
python tools/check_rope_spectrum.py --checkpoint results/demo.pt
```

多尺度机制检查（固定、随机周期、漂移、状态切换四种合成任务；所有模式同一 base 搜索预算；
记录 log-scale 轨迹；可评估训练长度以外的预测步）。同样只是机制检查，不是基准结果：

```bash
python examples/synthetic_multiscale.py --task regime_switch --eval-horizons 48 96 144
```

## 9. 仓库结构与下一步

```text
GroupedRoPE/
  README.md                 项目动机、数学、实现边界与使用说明
  AGENTS.md                 给 AI 助手的协作约定
  pyproject.toml            可安装的 Python 包
  requirements.txt
  src/grouped_rope/
    __init__.py
    model.py                通用预测器、两种预测头与位置编码配置
  tests/test_model.py       数学性质、梯度和模型接口测试
  tools/
    smoke_test.py
    check_rope_spectrum.py
  examples/
    train_synthetic.py      最小合成训练示例
    synthetic_multiscale.py 多尺度合成机制检查与预测长度外推
  docs/
    experiments.md          假设、对照、数据划分与评价方案
    implementation.md      具体实现与旧代码迁移说明
    plan_horizon_tokens.md  占位 token 预测头的设计依据
  .github/workflows/tests.yml
```

下一阶段是实现真实数据 loader、训练集标准化、共享训练入口和基线。建议先选择两个公开预测数据集，完成 standard/head/shared_group/grouped 的同预算对照，再考虑更多频率组。完整实验设计见 [docs/experiments.md](docs/experiments.md)。

## 10. 项目来源

本项目由早期 RopeFormer 方法原型重新整理，只保留相关模型思路。新包名为 grouped_rope，接口有意不兼容旧 oil 任务。旧 proposal、结果与历史设计文档没有带入此仓库，也不能作为本项目的实验依据。

这是一个本地 Git 项目骨架，尚未连接远程仓库。发布或确定开源许可前由小组确认。
