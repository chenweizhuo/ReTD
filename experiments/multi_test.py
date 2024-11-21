import argparse
from pathlib import Path
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from dataloaders import multi_dataload
from multi_class import inference_transformer


def main(args):
    repo_id = "./CompVis/stable-diffusion-2-base"
    print(f"使用 {repo_id} 重构特征数据集测试")

    # 定义输出路径
    output_dir = Path("output/model") / args.experiment_id
    preprocessed_data_base_dir = Path("preprocessed_data/only_diff")
    model_output_dir = output_dir / repo_id.replace('/', '_').replace('.', '') / 'multi/only_diff'
    model_output_dir.mkdir(parents=True, exist_ok=True)
    i = 1000

    # 加载推理数据
    infer_dataloader = multi_dataload(repo_id, preprocessed_data_base_dir, args.batch_size, i)

    # 模型相关路径
    model_name = repo_id.replace('/', '_').replace('.', '')
    model_path = model_output_dir / f"vit_model_{model_name}_only_diff_11.pth"
    pyramid_pooling_path = model_output_dir / f"cnn_pooling_{model_name}_only_diff_11.pth"

    # 执行推理
    inference_outputs = inference_transformer(infer_dataloader, model_path, pyramid_pooling_path, model_name)

    # 获取真实标签和预测结果
    true_labels = []
    for _, labels in infer_dataloader:
        true_labels.extend(labels.numpy())
    true_labels = torch.tensor(true_labels)
    predicted_labels = torch.argmax(inference_outputs, dim=1)

    # 计算混淆矩阵
    true_labels_np = true_labels.numpy()
    predicted_labels_np = predicted_labels.numpy()
    conf_matrix = confusion_matrix(true_labels_np, predicted_labels_np, labels=list(range(11)))

    # 输出混淆矩阵
    print("混淆矩阵（11类）：")
    print(conf_matrix)

    # 计算总体准确率
    accuracy = (predicted_labels == true_labels).float().mean().item()

    # 计算微观指标
    precision_micro = precision_score(true_labels_np, predicted_labels_np, average='micro')
    recall_micro = recall_score(true_labels_np, predicted_labels_np, average='micro')
    f1_micro = f1_score(true_labels_np, predicted_labels_np, average='micro')

    # 计算宏观指标
    precision_macro = precision_score(true_labels_np, predicted_labels_np, average='macro')
    recall_macro = recall_score(true_labels_np, predicted_labels_np, average='macro')
    f1_macro = f1_score(true_labels_np, predicted_labels_np, average='macro')

    # 逐类准确率
    class_accuracies = []
    for cls in range(11):
        cls_idx = true_labels_np == cls
        cls_total = cls_idx.sum()
        cls_correct = (predicted_labels_np[cls_idx] == true_labels_np[cls_idx]).sum()
        class_accuracy = cls_correct / cls_total if cls_total > 0 else 0
        class_accuracies.append(class_accuracy)

    # 输出总体指标
    print(f"总体准确率: {accuracy * 100:.2f}%")
    print(f"微观精度: {precision_micro * 100:.2f}%")
    print(f"微观召回率: {recall_micro * 100:.2f}%")
    print(f"微观 F1 分数: {f1_micro * 100:.2f}%")
    print(f"宏观精度: {precision_macro * 100:.2f}%")
    print(f"宏观召回率: {recall_macro * 100:.2f}%")
    print(f"宏观 F1 分数: {f1_macro * 100:.2f}%")

    # 输出逐类准确率
    for cls_idx, cls_acc in enumerate(class_accuracies):
        print(f"类别 {cls_idx} 的准确率: {cls_acc * 100:.2f}%")

    # 保存推理结果
    torch.save(inference_outputs, model_output_dir / f"inference_outputs_{model_name}_only_diff.txt")
    print(f"推理输出已保存到 {model_output_dir / f'inference_outputs_{model_name}_only_diff.txt'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="default", help="实验ID")
    parser.add_argument("--batch-size", type=int, default=20, help="批处理大小")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())