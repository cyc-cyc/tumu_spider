# %%
import csv
import difflib
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model, load_checkpoint_and_dispatch
from transformers.generation.utils import GenerationConfig
from transformers import pipeline
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, TextStreamer
from transformers import AutoTokenizer
import torch
import os
import json
import json5

import tqdm
import rich

import datetime

from typing import Union

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEBUG = False

# def read_file(filename):
#     with open(filename, 'r', encoding='utf-8') as file:
#         # 按行读取文件内容到列表中
#         raw_data = file.read().splitlines()
#     # print(f"raw_data: {raw_data}")
#     return raw_data

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        # 按行读取文件内容到列表中
        raw_data = file.read()
    # print(f"raw_data: {raw_data}")
    return raw_data


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

# def merge_data(raw_data, MAX_STR_LENGTH=2048):
#     text = ""
#     data = []
#     for line in raw_data:
#         print(f"line: {len(line)}, text: {len(text)}")
#         if len(line) + len(text) >= MAX_STR_LENGTH:
#             data.append(text)
#             print(f"text: {len(text)}")
#             text = line + '\n'
#         else:
#             text += line + '\n'
#     data.append(text)
#     print(f"last text: {len(text)}")
#     # print(f"data: {data}")
#     return data


# %%
def init_LLM(model_name, model_dir):
    # model_name = "Qwen1.5-14B-Chat"
    use_english_materials = True
    
    # model_dir = f"/nfs-data/zhengliwei/Projects/SHLP/LLMs/{model_name}"
    # model_dir = LLM_dir
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
    
    prompt = '''我会给你一段文本，对于这段文本中提到的建筑及其详细信息返回一个JSON对象，包括但不限于以下字段：建筑名称、来源、详细地址、建筑功能、建筑结构、经纬度、街区、街道、坪数、地上层数、地下层数、层高、屋龄、完工年代等。
                其中建筑结构字段从以下专有名词中挑选符合描述的作为回答：土木工程结构、木结构、石砌结构、砖砌结构、木柱、木桁架、木屋盖、圈梁、构造柱、砌块墙体、钢板墙、钢结构、冷弯薄壁型钢结构、钢筋混凝土结构、混凝土结构、预制装配式混凝土结构、钢筋混凝土结构连接、钢筋混凝土柱、钢筋混凝土桁架、素混凝土结构、组合结构、组合连接、混合结构、复合材料筋混凝土结构、剪力墙、杆、桁架、柱、板、梁、连梁、支撑、拱、穹顶、壳、结构体系、梁板结构、墙板结构、巨型结构、悬吊结构、筒体结构、悬挑结构、排架结构、框架核心筒结构、框筒结构、框架剪力墙结构、框架结构、预应力结构、薄壳结构、空间结构、悬索结构、特种工程结构、高耸结构、风电塔架、塔式结构、地下管道、荷载、活荷载、地基、桩基础、桩承台、热轧型钢、压型钢板、钢管、梁式桥、实腹梁桥、桁架梁桥、拱桥、悬索桥、刚架桥、斜拉桥、组合体系桥、立交桥、桥台、桥墩、隧道、铁路隧道、人防工程、围岩、明洞、盾构
                你的回答必须遵循以下格式：json
                                        {
                                        "建筑名称": "……",
                                        "来源": "……",
                                        "详细地址": "……",
                                        "建筑功能": "……",
                                        "建筑结构": "……",
                                        "经度": "……",
                                        "纬度": "……",
                                        "地上层数": "……",
                                        "地下层数": "……",
                                        "层高": "……",
                                        "屋龄": "……",
                                        "完工年代": "……",
                                        "……": "……"
                                        }
                你可以根据给定文本中的信息补充一些字段放在JSON格式的数据中，越详细越好。文本中未出现的字段只返回"未提及"。
                注意区分屋龄和完工年代，完工年代是指建筑完工的年份，屋龄是指房屋自建成至今(2024年)的年数。
                层数选取最大层数，如"7、12层"则层数为"12层"。
                层高指的是楼层高度。
                若出现完工年代只有三位数，则表示使用民国纪年法，应加上1911换算成公元纪年法，如"103年"应换算成"2014年"。
                你的回答必须只包含上述JSON格式的数据，以"json{"开始，以"}endgen"结束，不要返回其他的说明，不要给任何字段增加注释。
                如果给定文本不包含建筑信息，则直接返回"NONE"。
                文本:'''
    
    return model_name, model, tokenizer, prompt

# %%
def use_LLM_predict(data, model_name, model, tokenizer, prompt):

    def predict_line(line, prompt):
        content = prompt + line
        messages = [{"role": "user", "content": content}]
        if model_name == "Qwen1.5-7B-Chat" or model_name == "Qwen1.5-14B-Chat":
            # Qwen1.5 疑似没有实现 model.chat ？
            # 以下的代码改自 Qwen1.5 的 Readme 文件。
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
        else:
            raise NotImplementedError
        return str(response[0])
    
    merged_dict = {}
    for i, line in tqdm.tqdm(enumerate(data)):
        print(f"line: len: {len(line)}, {line}")
        err_msg = ""
        while True:
            res = predict_line(line, err_msg + prompt)
            print(f"res: {res}")
            # 删除前面的提示信息
            if "json" in res:
                res = res[res.index("json")+len("json"):]
            if "```<|im_end|>" in res:
                res = res[:res.index("```<|im_end|>")]
            if "<|im_end|>" in res:
                res = res[:res.index("<|im_end|>")]
            if "endgen" in res:
                res = res[:res.index("endgen")]
            if "NONE" in res:
                break
            if "```" in res:
                res = res[:res.index("```")]
            print(f"res_after: {res}")
            try:
                res_json = json5.loads(res)
                # print(f"res: {res}")
                # merged_dict.update(res)
                for key, value in res_json.items():
                    # 只有当 key 不在 merged_dict 中时，才更新（合并）它们的值
                    if key not in merged_dict:
                        merged_dict[key] = value
                    elif merged_dict[key] == "未提及" and value != "未提及":
                        ratio = difflib.SequenceMatcher(None, merged_dict['建筑名称'], res_json['建筑名称']).ratio()
                        # print(f"merged_dict: {merged_dict['建筑名称']}, res: {res_json['建筑名称']}, ratio: {ratio}")
                        if ratio > 0.6:
                            merged_dict[key] = value
                # print(f"merged_dict: {merged_dict}")
                break
            except Exception as e:
                print("ERROR:", e)
                err_msg = "你上次的回答不符合标准的json格式，json解析器报错：" + str(e) + "，请重新抽取。"
                # return res
    return merged_dict

# %%
# json数据写入文件, 缩进为4, 自动换行
def write_json(filename, data_json):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_json, f, indent=4, ensure_ascii=False)


def write_txt(filename, data_txt):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data_txt)


# # %%
# # file_dir = '/nfs-data/spiderman/content/2024-05-18/'
# # file_dir = '/nfs-data/spiderman/content/2024-05-20/'
# # file_dir = '/nfs-data/spiderman/content/2024-10-09/'
# file_dir = '/nfs-data/spiderman/content/2024-10-16/'
# # file_dir = '/data/gaozhicheng/GenPrompts/journeys/'
# # file_dir = '/nfs-data/spiderman/content/2024-05-28/'
# # save_dir = '2024-05-18/'
# # save_dir = '2024-10-15/'
# save_dir = '2024-10-16/'
# # save_dir = '2024-05-28/'
# # 检查该路径是否存在
# if not os.path.exists(save_dir):
#     # 如果不存在，则创建文件夹
#     os.makedirs(save_dir)

# def main():
#     # 初始化模型
#     model_name, model, tokenizer, prompt = init_LLM()
#     entries = os.listdir(file_dir)
#     all_file_names = [f for f in entries if os.path.isfile(os.path.join(file_dir, f))]
#     print(f"file_names: {all_file_names}")

#     # 检查已经抽过的文件
#     already_read_list = os.listdir(save_dir)
#     # 去除文件名中的"_result.json"
#     already_read_list = [f[:f.index("_result.json")] for f in already_read_list]
#     # 检查文件中是否有已经抽取过的文件
#     file_names = []
#     for file_name in all_file_names:
#         if file_name not in already_read_list:
#             file_names.append(file_name)
#         else:
#             print(f"already read: {file_name}")

#     # file_name = file_names[0]
#     for file_name in file_names:
#         print(f"file_name: {file_name}")
#         # 读取原始数据
#         raw_data = read_file(file_dir + file_name)
#         # 分段合并, 保证每段不超过MAX_L
#         data= merge_data(raw_data, MAX_L=800, RED_L=100)
#         # 抽取
#         res_json = use_LLM_predict(data, model_name, model, tokenizer, prompt)
#         # 如果是json格式, 则写入json文件
#         if isinstance(res_json, dict):
#             write_json(save_dir + file_name + "_result.json", res_json)
#         else:
#             write_txt(save_dir + file_name + "_result.txt", res_json)

# %%
# def main():
#     # file_dir = '/nfs-data/spiderman/content/2024-05-18/'
#     file_dir = '/nfs-data/spiderman/content/2024-10-09/'
#     # file_dir = '/data/gaozhicheng/GenPrompts/journeys/'
#     # file_dir = '/nfs-data/spiderman/content/2024-05-28/'
#     # save_dir = '2024-05-18/'
#     save_dir = '2024-11-27/'
#     # save_dir = '2024-05-28/'

#     subdirs = [d for d in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, d))]
#     for subdir in subdirs:
#         subdir_path = os.path.join(file_dir, subdir)  # 原始子文件夹路径
#         save_subdir_path = os.path.join(save_dir, subdir)  # 保存结果的子文件夹路径

#         # 检查该路径是否存在
#         if not os.path.exists(save_dir):
#             # 如果不存在，则创建文件夹
#             os.makedirs(save_dir)
#         # 检查保存路径是否存在
#         if not os.path.exists(save_subdir_path):
#             os.makedirs(save_subdir_path)
#         file_names = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
#         print(f"Processing folder: {subdir}, files: {file_names}")

#         # 初始化模型
#         model_name, model, tokenizer, prompt = init_LLM()
#         # entries = os.listdir(file_dir)
#         # file_names = [f for f in entries if os.path.isfile(os.path.join(file_dir, f))]
#         # print(f"file_names: {file_names}")
#         # file_name = file_names[0]
#         for file_name in file_names:
#             # 读取原始数据
#             raw_data = read_file(file_dir + file_name)
#             # 分段合并, 保证每段不超过MAX_L
#             data= merge_data(raw_data, MAX_L=800, RED_L=100)
#             # 抽取
#             res_json = use_LLM_predict(data, model_name, model, tokenizer, prompt)
#             # 写入文件
#             result_file_path = os.path.join(save_subdir_path, file_name + "_result.json")
#             write_json(result_file_path, res_json)
            
# if __name__ == "__main__":
#     main()
        



# def main(file_dir, save_dir):
#     # 在这里使用传入的file_dir和save_dir，而不是硬编码它们
#     # file_dir = '/nfs-data/spiderman/content/2024-10-09/'
#     # save_dir = '2024-11-27/'
    
#     subdirs = [d for d in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, d))]
#     # 初始化模型
#     model_name, model, tokenizer, prompt = init_LLM()
#     for subdir in subdirs:
#         subdir_path = os.path.join(file_dir, subdir)  # 原始子文件夹路径
#         save_subdir_path = os.path.join(save_dir, subdir)  # 保存结果的子文件夹路径

#         # 检查该路径是否存在
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         # 检查保存路径是否存在
#         if not os.path.exists(save_subdir_path):
#             os.makedirs(save_subdir_path)
        
#         file_names = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
#         print(f"Processing folder: {subdir}, files: {file_names}")

#         for file_name in file_names:
#             # 读取原始数据
#             raw_data = read_file(subdir_path + file_name)
#             # 分段合并, 保证每段不超过MAX_L
#             data = merge_data(raw_data, MAX_L=800, RED_L=100)
#             # 抽取
#             res_json = use_LLM_predict(data, model_name, model, tokenizer, prompt)
#             # 写入文件
#             result_file_path = os.path.join(save_subdir_path, file_name + "_result.json")
#             write_json(result_file_path, res_json)
        
def main(file_dir, save_dir, model_name, model_dir):
    # 在这里使用传入的file_dir和save_dir，而不是硬编码它们
    # file_dir = '/nfs-data/spiderman/content/2024-10-09/'
    # save_dir = '2024-11-27/'
    
    # 获取file_dir下的所有文件夹
    top_level_dirs = [d for d in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, d))]
    
    # 初始化模型
    model_name, model, tokenizer, prompt = init_LLM(model_name, model_dir)
    
    for top_level_dir in top_level_dirs:
        # 获取每个一级目录下的两个子文件夹
        top_level_dir_path = os.path.join(file_dir, top_level_dir)
        subdirs = [d for d in os.listdir(top_level_dir_path) if os.path.isdir(os.path.join(top_level_dir_path, d))]
        
        # 假设每个一级目录下有两个子文件夹
        # if len(subdirs) != 2:
        #     print(f"Warning: {top_level_dir} does not contain exactly two subdirectories.")
        #     continue
        
        for subdir in subdirs:
            subdir_path = os.path.join(top_level_dir_path, subdir)  # 当前子文件夹路径
            save_subdir_path = os.path.join(save_dir, top_level_dir, subdir)  # 保存结果的子文件夹路径
            
            # 检查保存路径是否存在，不存在则创建
            if not os.path.exists(save_subdir_path):
                os.makedirs(save_subdir_path)
            
            # 获取子文件夹中的文件列表
            file_names = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            print(f"Processing folder: {top_level_dir}/{subdir}, files: {file_names}")

            # 遍历当前子文件夹中的文件
            for file_name in file_names:
                # 读取原始数据
                raw_data = read_file(os.path.join(subdir_path, file_name))
                # 分段合并，确保每段不超过MAX_L
                data = merge_data(raw_data, MAX_L=800, RED_L=100)
                # 信息抽取
                res_json = use_LLM_predict(data, model_name, model, tokenizer, prompt)
                # 写入文件
                result_file_path = os.path.join(save_subdir_path, file_name[:-4] + "_result.json")
                write_json(result_file_path, res_json)


'''
cd /data/gaozhicheng/miniconda3/bin/
conda activate /data/gaozhicheng/miniconda3/envs/llama3
cd -
'''
