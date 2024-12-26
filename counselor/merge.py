'''
土木项目真伪判别模块，包含:
1.多个来源查缺补漏 + 检查是否冲突 
2.规则判别
3.源文追溯
'''
import os
import json
import re
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
import warnings
from transformers import logging
warnings.filterwarnings("ignore")  # 暂时屏蔽所有警告
logging.set_verbosity_error()      # 降低 Transformers 的日志级别

def find_files_by_extension(folder_path, file_extension):
    """
    查找指定文件夹及其子文件夹中的所有特定类型文件。

    :param folder_path: 文件夹路径
    :param file_extension: 文件类型（如 '.txt' 或 '.json'）
    :return: 包含所有匹配文件路径的列表
    """
    matched_files = []  # 用于存储找到的文件路径
    for root, _, files in os.walk(folder_path):  # 遍历文件夹及其子文件夹
        for file in files:
            if file.endswith(file_extension):  # 检查文件扩展名
                matched_files.append(os.path.join(root, file))  # 拼接完整路径
    return matched_files

def merge(extract_folder_path):
    """
    多个来源查缺补漏+检查是否冲突
    """
    def merge_json(base_json, new_json):
        """
        合并两个 JSON 对象。如果某个字段在两个 JSON 中不同，合并为一个列表显示。
        """
        for key, value in new_json.items():
            value = str(value)  # 确保是字符串类型
            try:
                if key in base_json:
                    if '未提及' in base_json[key]:
                        # 如果 base_json 中的值为 "未提及"，直接替换
                        base_json[key] = value
                    elif '存在冲突' in base_json[key]: # 如果已经存在冲突，补充
                        base_json[key] = str(base_json[key]) + " / " + str(value) 
                    elif '未提及' in value:
                        continue
                    else: # 判断是否存在冲突
                        ratio = difflib.SequenceMatcher(None, base_json[key], value).ratio() # 通过相似度判断是否存在冲突
                        if ratio > 0.6:
                            continue
                        else: # 如果存在冲突，合并
                            base_json[key] = "存在冲突：" + str(base_json[key]) + " / " + str(value) 
                else:
                    # 如果 base_json 中没有该字段，直接添加
                    base_json[key] = value
            except:
                pass
        return base_json
    # final_json
    # 读取多个抽取json结果
    # 1. 互相补充
    # 2. 检查是否冲突
    final_json = {}
    files_list = find_files_by_extension(extract_folder_path, '.json')
    for file in files_list:
        with open(file, 'r', encoding='utf-8') as f:
            if final_json == {}:  # 如果 final_json 为空，直接加载第一个 JSON 文件
                    final_json = json.load(f)
            else:
                try:
                    current_json = json.load(f)  # 加载当前 JSON 文件
                    final_json = merge_json(final_json, current_json)  # 合并 JSON
                except:
                    continue
    print("多个来源结果合并完成！")
    return final_json
    # 在这里编写第一个函数的具体实现

def init_llm(model_dir, device_list):
    model_name = "Qwen1.5-14B-Chat"
    # model_dir = os.path.join(model_dir, model_name)
    tokenizer = AutoTokenizer.from_pretrained(
    model_dir, use_fast=False, trust_remote_code=True,)

    model_dtype_dict = {
        "Qwen1.5-7B-Chat": "auto",
        "Qwen1.5-14B-Chat": torch.float16,
        "Qwen1.5-MoE-A2.7B": "auto",  # 需要32GB显卡
    }

    mm_dict = {0: "24576MiB", 1: "24564MiB", 2: "24564MiB"}
    for i in mm_dict.keys():
        if i not in device_list:
            mm_dict[i] = "0MiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=model_dtype_dict[model_name],
        max_memory=mm_dict)

    return model_name, model, tokenizer

def get_llm_result(content, model, tokenizer):
    messages = [{"role": "user", "content": content}]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(
        model_inputs.input_ids.to('cuda'),
        max_length=8192,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return str(response[0])

def trace_llm(final_json, spider_folder_path, model, tokenizer):
    """
    源文追溯，输出对应的句子。
    调用大模型,源文需要分段处理
    存在冲突的不处理
    """
    def merge_data(raw_data, MAX_L=2048, RED_L=100):
        # 按照最大长度(MAX_L)拆分数据, 每条数据的前面附带上一条数据的冗余(RED_L), 防止信息丢失
        if len(raw_data) <= MAX_L:
            return [raw_data]
        # 第一条数据前面不附带冗余, 直接取MAX_L长度
        data = [raw_data[:MAX_L]]
        # 有效数据长度
        ACT_L = MAX_L - RED_L
        
        for i in range(MAX_L, len(raw_data), ACT_L):
            # 从第二条数据开始, 每条数据前面附带上一条数据的冗余
            # 实际有效长度为ACT_L
            data.append(raw_data[i-RED_L:i+ACT_L])

        return data

    def find_value_in_text(value, spider_path, model, tokenizer, prompt):
        files_list = find_files_by_extension(spider_path, '.txt')
        for file in files_list:
            with open(file, 'r', encoding='utf-8') as f:
                source_text = f.read()
                # 分段处理
                data= merge_data(source_text, MAX_L=800, RED_L=100)
                for text in data:
                    content = prompt.format(text=text, value=value)
                    trace_result = get_llm_result(content, model, tokenizer)
                    if "追溯失败" not in trace_result:
                        return trace_result
        # 如果没找到
        return "追溯失败"

    prompt = "请帮我在以下段落中“{text}”，找到“{value}”所在的句子，请直接回答句子，如果找不到请回答“追溯失败”。"
    trace_json = {} # 初始化一个json保存溯源结果
    # 遍历 final_json 中的每个字段和值，查找源文中是否存在
    for key, value in final_json.items():
        if "未提及" in value:
            trace_json[key] = "未提及，无需追溯"
        elif "存在冲突" in value:
            trace_json[key] = "存在冲突，无法追溯"
        else:
            result = find_value_in_text(value, spider_folder_path, model, tokenizer, prompt)
            trace_json[key] = result
    print("源文追溯完成！")
    return trace_json
    

def save_final_json(final_json, trace_json, show_folder_path):
    """
    保存最终结果到文件
    """
    if "来源" in final_json and isinstance(final_json["来源"], str):
        final_json["来源"] = final_json["来源"].replace("存在冲突：", "")
    # 保存 final_json 到文件
    final_json_path = os.path.join(show_folder_path, "final.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
    # 保存 trace_json 到文件
    trace_json_path = os.path.join(show_folder_path, "trace.json")
    with open(trace_json_path, 'w', encoding='utf-8') as f:
        json.dump(trace_json, f, ensure_ascii=False, indent=4)
    print(f"已保存最终结果到文件：{final_json}")

def check_llm(merge_json, model, tokenizer):
    """
    输入建筑数据，让llm根据已有信息，补充年代、结构体系、结构材料信息
    """
    # TODO 这部分后续需要微调prompt，以及输出需要指明的LLM输出的，并考虑后续怎么和「下一个模块」规则判别的结构结合
    prompt_year = '''
        ### 输入建筑信息：{building}
        ### 输出要求：请帮我根据建筑信息，判断建筑的完工年代.
    '''
    content_year = prompt_year.format(building=merge_json)
    year = get_llm_result(content_year, model, tokenizer)
    print("年代：", year)
    merge_json['完工年代'] = year

    prompt_cailiao = '''
        ### 输入建筑信息：{building}
        ### 输出要求：请帮我根据建筑信息，判断建筑的结构材料.
    '''
    content_cailiao = prompt_cailiao.format(building=merge_json)
    cailiao = get_llm_result(content_cailiao, model, tokenizer)

    promt_tixi = '''
        ### 输入建筑信息：{building}
        ### 输出要求：请帮我根据建筑信息，判断建筑的结构体系.
    '''
    content_tixi = promt_tixi.format(building=merge_json)
    tixi = get_llm_result(content_tixi, model, tokenizer)

    merge_json['结构材料'] = cailiao
    merge_json['结构体系'] = tixi
    print("结构材料：", cailiao)
    print("结构体系：", tixi)
    print("大模型年代、结构材料、结构体系判别完成！")
    return merge_json

def main(spider_path, extract_path, show_path, model, tokenizer):
        merge_json = merge(extract_path)
        final_json= check_llm(merge_json, model, tokenizer) # 用大模型判别规则
        trace_json = trace_llm(merge_json, spider_path, model, tokenizer)
        save_final_json(final_json, trace_json, show_path)

if __name__ == "__main__":
    '''
    给定一个建筑id，及对应的spider_data和extract_data文件夹路径，执行真伪判别模块
    输出：final.json和trace.json
    '''
    parser = argparse.ArgumentParser(description="Run information extraction on a specified directory.")
    parser.add_argument('--spider_path', type=str, required=True, help="")
    parser.add_argument('--extract_path', type=str, required=True, help="")
    parser.add_argument('--show_path', type=str, required=True, help="")
    parser.add_argument('--llm_model_path', type=str, required=True, help="")
    # parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, required=True, help="")

    args = parser.parse_args()

    spider_path = args.spider_path #'/nfs-data/spiderman/content/temp/'
    extract_path = args.extract_path #'/nfs-data/spiderman/result/temp/'
    show_path = args.show_path #'/nfs-data/spiderman/show/temp/'
    llm_model_path = args.llm_model_path #"/nfs-data/user31/finetuneLLM/LLaMA-Factory/saves/lora/export"
    main(spider_path, extract_path, show_path, llm_model_path)  # 执行主函数