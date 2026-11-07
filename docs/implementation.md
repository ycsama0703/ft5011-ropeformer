# 实现与迁移说明

## API

```python
from grouped_rope import ForecastConfig, GroupedRoPEForecaster
cfg = ForecastConfig(input_dim=3, horizon=24, rope_mode="grouped", n_freq_groups=2)
model = GroupedRoPEForecaster(cfg)
```

可传 cfg 或同名关键字，不能混用。output_dim=None 表示与 input_dim 相同。
未知关键字、无效组数、不整除的 head 维度等会报错。输入必须是非空时间轴的
三维浮点张量。模型不处理外部归一化、缺失值填补或设备迁移。

model.py 中按顺序包含：

- ForecastConfig：集中配置与校验；默认两层、四头、d_model=64、d_ff=128。
- GroupedRotaryEmbedding：基础频率 buffer、固定组归属、log-scale 与直接旋转。
- MultiHeadAttentionRoPE：QKV 投影、旋转、SDPA；显式诊断时返回 attention。
- EncoderBlock：标准 pre-norm 残差块，Q/K/V 都来自同一个归一化后的输入。
- GroupedRoPEForecaster：输入投影、encoder、输出 norm、mean/last pooling、直接预测。

没有 decoder。历史窗口内默认双向注意力是合法的，因为所有输入都在预测起点
之前；严格因果 encoder 通过 causal=True 启用。这不代替正确的数据切分。

## 参数化

低组号对应较高基础频率。组数不能超过每头的旋转坐标对数；仅 grouped/
shared_group 使用 n_freq_groups，其余模式只有一组。组内坐标对尽量均分，
不按绝对频率阈值分组。重新划分方案属于后续独立实验。

head_scale_init 仅适用于 head/fixed_head，必须每头一个正有限值。
固定参数用 buffer 注册，可学习参数用 nn.Parameter 注册，都会进入 state_dict。

尺度不硬性截断，因此没有全局数值稳定保证。优先监测与梯度裁剪；需要有界
参数化时，应显式记录这个方法变化。没有自动施加 Nyquist 约束或周期下限。
基础频率范围仍由 rope_base 决定；极慢旋转不自动判定为无用。

## 诊断与精度

直接用 position * effective_frequency 计算 sin/cos，不使用有限长度缓存插值。
三角计算在 autocast 关闭的 float32 中完成，然后转回 Q/K dtype。推荐通过
AMP 混合精度而不是直接 model.half()，以保留参数和 buffer 精度。

return_attention=True 返回 dropout 前且已 detach 的图；正常训练不存图。
rope_diagnostics() 返回 detached 副本。频率单位 rad/step；period 单位输入步。

## 清理了哪些旧实现

- 替换每头统一尺度为可选择的统一/分组尺度，保留公平对照。
- 去掉 mu/log_sigma/方向分类头，改为通用连续多步预测输出。
- 去掉无用的 x_dec、overlap_k、task_mode 和外生 cross-attention。
- 去掉懒创建的 mem_proj，防止 optimizer 初始化后漏掉新参数。
- 去掉静默忽略关键字及 ff_mult 宽度歧义。
- 修复 Q/K/V 归一化不一致和未使用的输出 LayerNorm。
- 用直接角度计算取代超出缓存范围时的插值/外推。
- 不兼容旧任务脚本是有意选择，没有隐式别名。

## 验证范围

测试覆盖初始标准 RoPE 等价、原方法退化、范数保持、相对位置、超窗口尺度、
每组梯度、组数参数量、保存恢复、诊断路径与 SDPA 路径一致、因果 mask、
输入配置校验和 CPU autocast。

CPU bf16 测试关闭 oneDNN 后运行，避免本机 CPU 对 bf16 backward 的指令支持
限制；不代表默认 oneDNN 路径或 CUDA 已通过。没有 GPU 或真实 benchmark 结果。
