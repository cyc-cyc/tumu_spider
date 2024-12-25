# 网络爬虫

## **Step 1. 代码配置** 

```
mkdir /path/to/your/folder

cd /path/to/your/folder

git clone https://github.com/cyc-cyc/tumu_spider.git
```

## **Step 2. 环境配置**

```
conda create -n spider python=3.8

conda activate spider

pip install -r requirements.txt
```

## **Step 3. 代理配置**

根据设备的条件自行配置代理

## **Step 4. 运行爬虫**
```
cd counselor
python main.py building_name building_id root_dir
```

爬取的文件以txt格式保存于本地目录`root_dir/content/building_id/building_name`中


# 抽取
## Step 1. Extract
for extracting info from website data using Qwen1.5-14B-Chat

## Step 2. Install
首先确保你的电脑安装了anaconda
创建虚拟环境：
```shell script
conda create -n extract_env python=3.9.19
```

激活虚拟环境：
```shell script
conda activate extract
```

安装pytorch：
```shell script
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

上述安装要求CUDA版本11.8或以上，若版本较低则进入pytorch官网挑选合适的pytorch版本。
pytorch官网：
```shell script
https://pytorch.org/get-started/previous-versions/
```

安装其他依赖：
```shell script
pip install -r requirements.txt
```

## Step 3. Run
修改run.sh脚本中的三个参数：
```shell script
--file_dir "/nfs-data/spiderman/content/2024-11-28/"                    # 爬取文件路径
--save_dir "/nfs-data/spiderman/result/2024-11-28/"                     # 保存文件路径
--model_name "Qwen1.5-14B-Chat"                                         # 模型名称（别动）
--model_dir "/nfs-data/zhengliwei/Projects/SHLP/LLMs/Qwen1.5-14B-Chat"  # 模型权重路径
--CUDA_VISIBLE_DEVICES "0, 1, 2, 3"                                     # 指定使用的gpu编号
```

运行脚本进行信息抽取：
```shell script
bash run.sh
```

# 大模型规则判别 + 溯源
## **step 1.环境配置**
首先确保你的电脑安装了anaconda，且相关配置文件 tumu_test.yml 已放入当前文件夹下
```
conda env create -f tumu_test.yml
```

## **step 2.数据准备**

模型路径：``

爬虫文件路径：``

信息抽取路径：``

结果保存路径：``
 
## **step 3.运行**
```
cd counselor
python merge_check_trace.py
```

# 自动化程序
```
python auto_process.py --root /path/to/your/save_folder
```
