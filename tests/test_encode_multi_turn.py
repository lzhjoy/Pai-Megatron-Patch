"""
测试真实的 encode_multi_turn 函数
验证 encode_multi_turn 的编码逻辑正确性 (适配负数 Boundary 逻辑)
"""

import numpy as np
import json
from transformers import AutoTokenizer


def encode_multi_turn(datas, tokenizer, seq_length):
    """
    Encode multi-turn conversations for SFT with proper masking and sequence packing.
    Fixed version: correct document boundary marking with EOT tokens.
    """
    if isinstance(datas, dict):
        datas = [datas]

    ids = {}
    lens = {}
    doc_ids = []
    sentence_lens = []
    label_ids = []

    pad_token_id = tokenizer.pad_token_id
    end_of_text_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_token = tokenizer.convert_tokens_to_ids("assistant")

    for data in datas:
        if isinstance(data, dict):
            if 'messages' in data:
                text = data['messages']
            else:
                text = [{
                    'role': 'user',
                    'content': data.get("instruction", "") + data.get('input', '')
                }, {
                    'role': 'assistant',
                    'content': data.get('output', '')
                }]
        else:
            text = data

        all_ids = tokenizer.apply_chat_template(text)
        
        if len(all_ids) >= seq_length:
            print('Extreme long sequence, truncated...')
            all_ids = all_ids[:seq_length]

        y_ids = [-100] * len(all_ids)
        all_ids_arr = np.array(all_ids)

        im_start_mask = all_ids_arr == im_start_id
        im_end_mask = all_ids_arr == im_end_id
        im_start_pos = np.where(im_start_mask)[0]
        im_end_pos = np.where(im_end_mask)[0]

        for start_idx in im_start_pos:
            if start_idx + 1 < len(all_ids) and all_ids[start_idx + 1] == assistant_token:
                end_positions = im_end_pos[im_end_pos > start_idx]
                if len(end_positions) > 0:
                    end_idx = end_positions[0]
                    for i in range(start_idx + 2, min(end_idx + 1, len(y_ids))):
                        y_ids[i] = all_ids[i]
                else:
                    for i in range(start_idx + 2, len(y_ids)):
                        y_ids[i] = all_ids[i]

        # --- Fixed Logic ---
        if len(all_ids) != seq_length:
            all_ids.append(end_of_text_id)
            y_ids.append(-100) # EOT 不计算 loss

        if len(all_ids) > seq_length:
            all_ids = all_ids[:seq_length]
            y_ids = y_ids[:seq_length]

        # Boundary marking
        all_ids[-1] = -1 - all_ids[-1]

        if sum(sentence_lens) + len(all_ids) > seq_length:
            if seq_length > sum(sentence_lens):
                doc_ids = doc_ids + [pad_token_id] * (seq_length - sum(sentence_lens))
                label_ids = label_ids + [-100] * (seq_length - sum(sentence_lens))
            
            ids['text'] = doc_ids + label_ids
            lens['text'] = [len(doc_ids) * 2]
            yield ids, lens, len(json.dumps(ids))
            
            ids = {}
            lens = {}
            doc_ids = []
            sentence_lens = []
            label_ids = []

        doc_ids.extend(all_ids)
        label_ids.extend(y_ids)
        sentence_lens.append(len(all_ids))

    if sum(sentence_lens) > 0:
        if seq_length > sum(sentence_lens):
            doc_ids = doc_ids + [pad_token_id] * (seq_length - sum(sentence_lens))
            label_ids = label_ids + [-100] * (seq_length - sum(sentence_lens))
        ids['text'] = doc_ids + label_ids
        lens['text'] = [len(doc_ids) * 2]
        yield ids, lens, len(json.dumps(ids))


# ==========================================
#  Helper Functions for Display
# ==========================================

def recover_ids(doc_ids):
    """
    将包含负数边界标记 (-1 - id) 的 doc_ids 还原为正常的 token ids
    """
    clean_ids = []
    boundary_indices = []
    for i, token_id in enumerate(doc_ids):
        if token_id < 0:
            original_id = -1 - token_id
            clean_ids.append(original_id)
            boundary_indices.append(i)
        else:
            clean_ids.append(token_id)
    return clean_ids, boundary_indices

def print_section(title: str):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

def analyze_output(doc_ids, label_ids, tokenizer, title: str = "结果分析"):
    """分析输出，自动处理负数 ID"""
    print(f"\n{title}:")
    print(f"  序列总长度: {len(doc_ids)}")

    # 1. 还原负数 ID 以便解码
    clean_doc_ids, boundaries = recover_ids(doc_ids)
    
    if boundaries:
        print(f"  检测到文档边界 (Negative IDs) 索引: {boundaries}")
        # 验证边界处的 token 是否真的是 EOT (或最后截断的词)
        boundary_tokens = [tokenizer.decode([clean_doc_ids[i]]) for i in boundaries]
        print(f"  边界对应的 Token 内容: {boundary_tokens}")

    masked_count = sum(1 for y in label_ids if y == -100)
    unmasked_count = len(label_ids) - masked_count

    print(f"  掩蔽 tokens (label=-100): {masked_count}")
    print(f"  用于学习 tokens (label!=-100): {unmasked_count}")

    # 2. 打印全文解码
    print(f"\n  全文解码 (已还原负数):")
    full_text = tokenizer.decode(clean_doc_ids, skip_special_tokens=False)
    print(f"{full_text}")

    # 3. 找到非掩蔽区域 (Label 对应的区域)
    unmasked_regions = []
    start = None
    for i, y in enumerate(label_ids):
        if y != -100:
            if start is None:
                start = i
        else:
            if start is not None:
                unmasked_regions.append((start, i - 1))
                start = None
    if start is not None:
        unmasked_regions.append((start, len(label_ids) - 1))

    if unmasked_regions:
        print(f"\n  Assistant 学习内容 (Check Labels):")
        for idx, (start, end) in enumerate(unmasked_regions, 1):
            # 同样需要使用 clean_doc_ids 来解码
            segment_ids = clean_doc_ids[start:end + 1]
            decoded = tokenizer.decode(segment_ids)
            print(f"    [{idx}] (pos {start}-{end}): {decoded}")

# ==========================================
#  Test Cases
# ==========================================

def test_case_1():
    print_section("测试 1: 简单单轮对话")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    data = {
        'messages': [
            {'role': 'user', 'content': 'What is Python?'}, 
            {'role': 'assistant', 'content': 'Python is code.'}
        ]
    }

    for ids, lens, size in encode_multi_turn([data], tokenizer, seq_length=512):
        doc_ids = ids['text'][:len(ids['text']) // 2]
        label_ids = ids['text'][len(ids['text']) // 2:]
        analyze_output(doc_ids, label_ids, tokenizer)

def test_case_2():
    print_section("测试 2: 多轮对话")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    data = {
        'messages': [
            {'role': 'user', 'content': 'Hi'}, 
            {'role': 'assistant', 'content': 'Hello!'}, 
            {'role': 'user', 'content': 'Bye'}, 
            {'role': 'assistant', 'content': 'See you.'}
        ]
    }

    for ids, lens, size in encode_multi_turn([data], tokenizer, seq_length=512):
        doc_ids = ids['text'][:len(ids['text']) // 2]
        label_ids = ids['text'][len(ids['text']) // 2:]
        analyze_output(doc_ids, label_ids, tokenizer)

def test_case_3():
    print_section("测试 3: Sequence Packing (多条数据打包)")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    datas = [
        {'messages': [{'role': 'user', 'content': 'A'}, {'role': 'assistant', 'content': '1'}]},
        {'messages': [{'role': 'user', 'content': 'B'}, {'role': 'assistant', 'content': '2'}]},
        {'messages': [{'role': 'user', 'content': 'C'}, {'role': 'assistant', 'content': '3'}]}
    ]

    print(f"输入: {len(datas)} 条短数据")

    # 验证是否都打包进了一个序列
    count = 0
    for ids, lens, size in encode_multi_turn(datas, tokenizer, seq_length=512):
        count += 1
        doc_ids = ids['text'][:len(ids['text']) // 2]
        label_ids = ids['text'][len(ids['text']) // 2:]
        
        analyze_output(doc_ids, label_ids, tokenizer, f"Packed Sequence {count}")
        
        # 验证是否包含多个负数边界
        clean, boundaries = recover_ids(doc_ids)
        print(f"  \n验证 Packing: 此序列包含 {len(boundaries)} 个文档边界 (预期是 3 个)")
        assert len(boundaries) == 3, "Packing 逻辑可能有误，未检测到3个边界"

def test_case_4():
    print_section("测试 4: 截断测试 (Truncation)")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    data = {
        'messages': [
            {'role': 'user', 'content': 'Tell me a long story.'}, 
            {'role': 'assistant', 'content': 'Once upon a time ' * 50} # 很长
        ]
    }

    # 强制极短的 seq_length
    seq_len = 50
    print(f"设置 seq_length = {seq_len}")

    for ids, lens, size in encode_multi_turn([data], tokenizer, seq_length=seq_len):
        doc_ids = ids['text'][:len(ids['text']) // 2]
        label_ids = ids['text'][len(ids['text']) // 2:]
        
        # 检查 doc_ids 最后一个是否是负数
        last_token = doc_ids[seq_len - 1] # 最后一个非 padding
        # 注意：如果有 padding，要找非 padding 的最后一个。这里刚好满了或被截断，通常在 index -1
        # 但 packing 逻辑可能在最后加了 padding
        
        clean_ids, boundaries = recover_ids(doc_ids)
        
        # 找到最后一个非 pad token
        real_len = 0
        for t in clean_ids:
            if t != tokenizer.pad_token_id:
                real_len += 1
        
        last_real_token_idx = real_len - 1
        last_val = doc_ids[last_real_token_idx]
        
        print(f"\n验证截断边界:")
        print(f"  最后一个有效 Token 原值: {last_val}")
        print(f"  是否为负数边界: {last_val < 0}")
        
        analyze_output(doc_ids, label_ids, tokenizer, "截断后分析")


def main():
    print("\n" + "=" * 80)
    print(" ENCODE_MULTI_TURN 修正版测试")
    print(" 重点：验证 EOT 插入及负数边界处理")
    print("=" * 80)

    try:
        test_case_1()
        test_case_2()
        test_case_3()
        test_case_4()

        print("\n" + "=" * 80)
        print(" ✅ 所有测试完成")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()