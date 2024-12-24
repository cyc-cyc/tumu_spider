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
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, TextStreamer
from transformers import AutoTokenizer
import torch

import os

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
                import ipdb; ipdb.set_trace()
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
                    print(file)#,"不是标准的json格式"
    return final_json
    # 在这里编写第一个函数的具体实现


def check(merge_json):
    """
    规则判别, 存在冲突怎么处理？
    """
    def extract_max_number(text):
        numbers = re.findall(r'\d+\.?\d*', str(text))  # 匹配整数和小数
        numbers = [float(num) for num in numbers]  # 转换为浮点数
        return max(numbers) if numbers else None  # 返回最大值

    # 替换标点符号为 '.'
    def normalize_date_format(date_text):
        normalized_date = re.sub(r'[^\d.]', '.', str(date_text))  # 替换非数字和点为点
        return normalized_date
    
    def handle_field_value(value, extract_function=None):
        """
        处理字段值，检查是否为'未提及'或'存在冲突'，并根据需要进行额外处理。
        :param value: 原始字段值
        :param extract_function: 用于处理正常值的函数（可选）
        :return: 处理后的字段值
        """
        value = str(value)  # 确保是字符串类型
        if "未提及" in value:
            return "未提及"
        elif "存在冲突" in value:
            return "存在冲突"
        elif extract_function:
            return extract_function(value)
        return value

    def data_process(df):
        year = handle_field_value(df["完工年代"], normalize_date_format)
        floor_up = handle_field_value(df["地上层数"], extract_max_number)
        h = handle_field_value(df["层高"], extract_max_number)
        
        return year, floor_up, h

    fields_to_extract = ["建筑名称", "地上层数", "层高", "完工年代"]
    # 提取所需的字段
    extracted_data = {field: merge_json.get(field, "未提及") for field in fields_to_extract}
    year, floor, h = data_process(extracted_data)
    # import ipdb; ipdb.set_trace()
    # 直接根据条件判断输出对应规则
    # 年代
    # 1999.2.21前
    if year == '未提及' or year == '存在冲突':
        rule = '年份信息缺少或存在冲突，无法判断'
    elif year < '1999.2.21': # 根据层数判断
        if floor == '未提及' or floor == '存在冲突':
            rule = '1999.2.21前的建筑，若层数小于10，则大概率为框架结构，小概率为框架-剪力墙；若层数大于10，则大概率为框架-剪力墙结构，小概率为框架结构。'
        elif floor < 10:
            rule = '1999.2.21前的建筑，若层数小于10，则大概率为框架结构，小概率为框架-剪力墙。'
        else:
            rule = '1999.2.21前的建筑，若层数大于10，则大概率为框架-剪力墙结构，小概率为框架结构。'
    else: # 根据房屋高度判断
        if h == '未提及' or h == '存在冲突':
            rule = '1999.2.21后的建筑，若层高小于20m，建筑材料为混凝土结构，则为框架结构；建筑材料为钢结构，则为框架结构(含阻尼器)。若层高大于等于20m且小于等于50m，建筑材料为混凝土结构，则为框架-剪力墙；建筑材料为钢结构，则为框架-支撑(含BRB、阻尼器)。若层高大于50m，建筑材料为混凝土结构，则为采用减(隔)震技术的框架-剪力墙；建筑材料为钢结构，则为采用减(隔)震技术的框架(含SRC)+支撑体系(BRB)。'
        elif h < 20:
            rule = '1999.2.21后的建筑，若层高小于20m，建筑材料为混凝土结构，则为框架结构；建筑材料为钢结构，则为框架结构(含阻尼器)。'
        elif h <= 50:
            rule = '1999.2.21后的建筑，若层高大于等于20m且小于等于50m，建筑材料为混凝土结构，则为框架-剪力墙；建筑材料为钢结构，则为框架-支撑(含BRB、阻尼器)。'
        else:
            rule = '1999.2.21后的建筑，若层高大于50m，建筑材料为混凝土结构，则为采用减(隔)震技术的框架-剪力墙；建筑材料为钢结构，则为采用减(隔)震技术的框架(含SRC)+支撑体系(BRB)。'
    # 保存在新的列中
    merge_json["规则"] = rule
    return merge_json # 返回更新后的json


def trace_llm(final_json, spider_folder_path, llm_model_path):
    """
    源文追溯，输出对应的句子。
    调用大模型？源文需要分段处理
    或者先把完工年代反处理一下，然后直接查找
    存在冲突的不处理
    """
    def init_llm(model_dir):
        model_name = "Qwen1.5-14B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True,)

        model_dtype_dict = {
            "Qwen1.5-7B-Chat": "auto",
            "Qwen1.5-14B-Chat": torch.float16,
            "Qwen1.5-MoE-A2.7B": "auto",  # 需要32GB显卡
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=model_dtype_dict[model_name],)
        prompt = "请帮我在以下段落中“{text}”，找到“{value}”所在的句子，请直接回答句子，如果找不到请回答“追溯失败”。"
        return model_name, model, tokenizer, prompt
    
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
                    response = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                    trace_result = str(response[0])
                    if "追溯失败" not in trace_result:
                        return trace_result
        # 如果没找到
        return "追溯失败"

    # 读取大模型
    model_name, model, tokenizer, prompt = init_llm(llm_model_path)
    trace_json = {} # 初始化一个json保存溯源结果
    # 遍历 final_json 中的每个字段和值，查找源文中是否存在
    print(final_json)
    for key, value in final_json.items():
        if key == "规则":
            continue # 规则不需要追溯
        if "未提及" in value:
            trace_json[key] = "未提及，无需追溯"
        elif "存在冲突" in value:
            trace_json[key] = "存在冲突，无法追溯"
        else:
            result = find_value_in_text(value, spider_folder_path, model, tokenizer, prompt)
            trace_json[key] = result
    return trace_json
    

def save_final_json(final_json, trace_json, show_folder_path):
    """
    保存最终结果到文件
    """
    # 保存 final_json 到文件
    final_json_path = os.path.join(show_folder_path, "final.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
    # 保存 trace_json 到文件
    trace_json_path = os.path.join(show_folder_path, "trace.json")
    with open(trace_json_path, 'w', encoding='utf-8') as f:
        json.dump(trace_json, f, ensure_ascii=False, indent=4)
    print(f"已保存最终结果到文件：{final_json_path}")


def main(spider_path, extract_path, show_path, llm_model_path):
        merge_json = merge(extract_path)
        final_json= check(merge_json) # 在merge_json中加上规则
        trace_json = trace_llm(final_json, spider_path, llm_model_path)
        save_final_json(final_json, trace_json, show_path)

if __name__ == "__main__":
    '''
    给定一个建筑id，及对应的spider_data和extract_data文件夹路径，执行真伪判别模块
    输出：final.json和trace.json
    '''

    spider_path = '/nfs-data/spiderman/content/temp/'
    extract_path = '/nfs-data/spiderman/result/temp/'
    show_path = '/nfs-data/spiderman/show/temp/'
    llm_model_path = "/nfs-data/user31/finetuneLLM/LLaMA-Factory/saves/lora/export"
    main(spider_path, extract_path, show_path, llm_model_path)  # 执行主函数
