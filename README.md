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
python main.py building_name building_id
```

爬取的文件以json格式保存于本地目录中


# 抽取
## extract
for extracting info from website data using Qwen1.5-14B-Chat

## install
首先确保你的电脑安装了anaconda
创建虚拟环境：
```shell script
conda create -n extract python=3.9.19
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

## run
修改run.sh脚本中的三个参数：
```shell script
--file_dir "/nfs-data/spiderman/content/2024-11-28/" # 爬取文件路径
--save_dir "/nfs-data/spiderman/result/2024-11-28/"  # 保存文件路径
--CUDA_VISIBLE_DEVICES "0, 1, 2, 3"                  # 指定使用的gpu编号
```

运行脚本进行信息抽取：
```shell script
bash run.sh
```
