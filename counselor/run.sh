source /data/gaozhicheng/miniconda3/bin/activate llama3
python run.py --file_dir "/nfs-data/spiderman/content/2024-11-28/" --save_dir "/nfs-data/spiderman/result/2024-12-25/" --model_name "Qwen1.5-14B-Chat"  --model_dir "/nfs-data/zhengliwei/Projects/SHLP/LLMs/Qwen1.5-14B-Chat" --CUDA_VISIBLE_DEVICES "4,5,3"
