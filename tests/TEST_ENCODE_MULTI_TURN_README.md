# encode_multi_turn 测试说明

## 概述

`test_encode_multi_turn.py` 是一个**完全独立**的测试脚本，用于测试 `encode_multi_turn` 函数的核心逻辑。

- ✅ **无外部依赖** - 内置 MockTokenizer，可直接运行
- ✅ **完整的 decode 输出** - 便于检查编码结果
- ✅ **10 个全面的测试用例** - 覆盖各种序列长度和数据格式

## 快速开始

```bash
python toolkits/tests/test_encode_multi_turn.py
```

## 测试用例说明

| # | 测试名称 | 目的 | 重点 |
|---|---------|------|------|
| 1 | 简单单轮对话 | 基础功能 | User + Assistant 消息 |
| 2 | 多轮对话 | 多个 assistant 回复 | 多个 assistant 标签区域 |
| 3 | 批处理不同长度 | 序列打包 | 多个样本合并 |
| 4 | 序列长度边界 | 边界情况 | seq_length = 64/128/256 |
| 5 | 旧格式 | 后向兼容 | instruction/input/output 格式 |
| 6 | Padding 验证 | Padding 处理 | 短序列 Padding 到 seq_length |
| 7 | End-of-text 验证 | 特殊 token | <\|endoftext\|> 令牌位置 |
| 8 | 极短序列 | 最小化测试 | 最短可能的输入 |
| 9 | 多轮 (3 轮) | 复杂对话 | 3 个独立的 assistant 回复 |
| 10 | 超长截断 | 超出 seq_length | 超长序列正确截断 |

## 输出解读

### 完整输出 (Full Sequence)

```
完整输出:

Full Sequence:
  Doc IDs:   [151644, 151643, 5, 6, 9, 151645, ...]  # 实际的 token IDs
  Label IDs: [-100, -100, -100, -100, -100, -100, ...]  # 标签掩蔽
  Decoded:   <|im_start|> <|endoftext|> what is python ...  # 解码后的文本
```

- **Doc IDs**：实际编码的 token 序列
- **Label IDs**：掩蔽标签（-100 表示掩蔽，其他值表示用于损失计算）
- **Decoded**：解码后的人类可读文本

### 标签分析 (Label Analysis)

```
Label Analysis:
  总 tokens: 256
  掩蔽 (-100): 247 tokens
  非掩蔽: 9 tokens
  掩蔽比例: 96.5%

  掩蔽模式 (前 100 个 tokens):
    MMMMMMUUUUUUUUUMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM...
    M = Masked (-100)
    U = Unmasked (用于损失计算)

  非掩蔽区域 (assistant 内容):
    [6:14] (9 tokens):
      Tokens: [151644, 77091, 9, 6, 101, 10, 11, -151646, 151643]
      Decode: <|im_start|> assistant python is [101] programming language [-151646] <|endoftext|>
```

- **掩蔽比例**：表示被掩蔽的 token 占比
- **掩蔽模式**：可视化的 M/U 模式，快速检查掩蔽是否正确
- **非掩蔽区域**：列出所有用于损失计算的 assistant 内容

## 关键特性

### 1. 自动 Padding

```
  实际内容: 9 tokens
  Padding: 503 tokens
  总长度: 512 (目标: 512)
  ✓ 正确 Padding 到 seq_length
```

### 2. 多 Assistant 回复识别

测试 9 展示了如何正确识别和标记多个 assistant 回复：

```
Assistant 回复区域数: 3
  回复 1: [5:9] (5 tokens)
  回复 2: [15:19] (5 tokens)
  回复 3: [25:30] (6 tokens)
```

### 3. 超长序列截断

```
  ⚠️  Extreme long sequence, truncated... (406 -> 128)
  输出长度: 128
  是否截断到 seq_length: True
  包含 end-of-text token: True
```

## 常见检查清单

运行测试后，检查以下几点：

- [ ] 所有测试都完成且无错误
- [ ] Decode 输出看起来合理（包含 `<|im_start|>`, `assistant`, `<|im_end|>`)
- [ ] 掩蔽模式显示正确的 M/U 分布
- [ ] 非掩蔽区域确实对应 assistant 内容
- [ ] Padding 处理正确（短序列补充到 seq_length）
- [ ] 超长序列被截断到 seq_length

## 技术细节

### Token IDs

- `151644`: `<|im_start|>`
- `151645`: `<|im_end|>`
- `151643`: `<|endoftext|>`
- `77091`: `assistant`
- `151650`: `<|pad|>` (Padding token)

### 标签掩蔽规则

- `-100`：掩蔽 token（不计入损失）
- 其他值：实际 token ID（计入损失）
- `-1 - token_id`：最后一个 token 的特殊标记

### 序列打包

- 每个对话后添加 `<|endoftext|>` 分隔符
- 多个对话连续排列，然后 padding 到 seq_length
- 每个 seq_length 的数据会生成一个独立的输出

## 修改和扩展

如果需要修改测试：

1. 修改 `MockTokenizer` 中的 token IDs 来适配不同的 tokenizer
2. 在 `encode_multi_turn` 函数中调整 seq_length 参数
3. 添加新的测试用例到 `main()` 函数

## 相关文件

- `build_idxmap_sft_dataset.py` - 原始实现（使用真实 tokenizer）
- `test_chat_template.py` - Chat template 测试

