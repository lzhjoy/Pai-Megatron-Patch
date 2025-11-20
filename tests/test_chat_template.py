from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained("/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct")

text = [
    {'role': 'user', 'content': "Hello"},
    {'role': 'assistant', 'content': "Hello, how are you?", 'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Beijing'}}, {'name': 'get_news', 'arguments': {'topic': 'politics'}}]},
    {'role': 'tool', 'content': 'The weather in Beijing is sunny.'},
    {'role': 'assistant', 'content': "The weather in Beijing is sunny."}
]

input_ids = tokenizer.apply_chat_template(text)
print(input_ids)
print(tokenizer.decode(input_ids))


assistant_pattern = re.compile(r"<\|im_start\|>assistant(.*?<\|im_end\|>)", re.DOTALL)
for match in re.finditer(assistant_pattern, tokenizer.decode(input_ids)):
    print("Labels to predict: ", match.group(1))


print(tokenizer.special_tokens_map)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)

# end_of_text_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
# print(end_of_text_id)
# print(tokenizer.decode(end_of_text_id))
