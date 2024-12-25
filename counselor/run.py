import argparse
import os
import sys

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Run information extraction on a specified directory.")
    parser.add_argument('--file_dir', type=str, required=True, help="The directory containing the files to process.")
    parser.add_argument('--save_dir', type=str, required=True, help="The directory to save the results.")
    parser.add_argument('--model_name', type=str, required=True, help="The directory containing the files to process.")
    parser.add_argument('--model_dir', type=str, required=True, help="The directory to save the results.")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, required=True, help="CUDA_VISIBLE_DEVICES to set.")

    args = parser.parse_args()

    # 检查输入目录是否存在
    if not os.path.exists(args.file_dir):
        print(f"Error: The directory {args.file_dir} does not exist.")
        sys.exit(1)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # 导入原始提取代码
    from extract_web_new import main as extract_main

    # 运行信息抽取
    extract_main(file_dir=args.file_dir, save_dir=args.save_dir, model_name=args.model_name, model_dir=args.model_dir)

if __name__ == "__main__":
    main()
