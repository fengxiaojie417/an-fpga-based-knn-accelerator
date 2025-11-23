"""
自适应K值L1-KNN算法验证器 - 医学血细胞分类应用
数据集：Blood Cell Images (医学血细胞数据集)
支持多种血细胞类型识别
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from collections import Counter
import warnings

# 设置中文字体
def setup_chinese_font():
    """设置中文字体，如果失败则使用英文"""
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        print(f"Warning: Could not set Chinese font. Using English labels. Error: {e}")
        return False

USE_CHINESE = setup_chinese_font()
warnings.filterwarnings('ignore')


def generate_medical_dataset(n_samples=1000, n_classes=12, n_features_raw=1024, random_state=42):
    """
    生成真实感的医学血细胞图像数据集
    模拟临床血液分析中的实际挑战：类间重叠、不平衡分布、噪声

    参数：
        n_samples: 总样本数（默认1000）
        n_classes: 血细胞类型数量（12种，包含难以区分的亚型）
        n_features_raw: 原始特征维度（32x32像素）
        random_state: 随机种子

    返回：
        X: 特征矩阵
        y: 标签向量
        class_names: 类别名称列表
    """
    np.random.seed(random_state)

    # 定义12种血细胞类型（包含一些难以区分的亚型）
    class_names = [
        'Neutrophil-Band',      # 杆状核中性粒细胞
        'Neutrophil-Segmented', # 分叶核中性粒细胞
        'Lymphocyte-Typical',   # 典型淋巴细胞
        'Lymphocyte-Atypical',  # 非典型淋巴细胞
        'Monocyte',             # 单核细胞
        'Eosinophil',           # 嗜酸性粒细胞
        'Basophil',             # 嗜碱性粒细胞
        'Promyelocyte',         # 早幼粒细胞
        'Myelocyte',            # 中幼粒细胞
        'Metamyelocyte',        # 晚幼粒细胞
        'Erythroblast',         # 幼红细胞
        'Blast'                 # 原始细胞
    ]

    # 设置不平衡的类别分布（模拟真实血液样本）
    class_weights = np.array([0.12, 0.15, 0.14, 0.08, 0.10, 0.06, 0.03, 0.08, 0.07, 0.06, 0.07, 0.04])
    samples_per_class = (class_weights * n_samples).astype(int)
    samples_per_class[-1] = n_samples - samples_per_class[:-1].sum()  # 确保总数正确

    X_list = []
    y_list = []

    print(f"生成真实感医学血细胞数据集...")
    print(f"  - 细胞类型数: {n_classes}")
    print(f"  - 总样本数: {n_samples}")
    print(f"  - 原始特征维度: {n_features_raw} (32x32 像素)")
    print(f"  - 类别分布: 不平衡（模拟真实血液样本）")

    # 定义细胞类型之间的相似度矩阵（用于引入类间混淆）
    similarity_groups = [
        [0, 1],      # 中性粒细胞亚型相似
        [2, 3],      # 淋巴细胞亚型相似
        [7, 8, 9],   # 粒细胞发育阶段相似
    ]

    for class_idx in range(n_classes):
        n_samples_class = samples_per_class[class_idx]

        # 每种细胞类型有独特的特征模式
        # 使用多个高斯分布的混合来模拟复杂的形态特征

        # 基础特征分布（细胞的主要特征）
        base_mean = np.random.uniform(40, 180, n_features_raw)

        # 根据细胞类型调整特征
        if class_idx in [0, 1]:  # 中性粒细胞：多叶核
            base_mean[:n_features_raw//4] += 30  # 核区域较亮
        elif class_idx in [2, 3]:  # 淋巴细胞：大核小胞质
            base_mean[n_features_raw//4:n_features_raw//2] += 40
        elif class_idx == 4:  # 单核细胞：马蹄形核
            base_mean[n_features_raw//3:2*n_features_raw//3] += 25
        elif class_idx == 5:  # 嗜酸性粒细胞：橙红色颗粒
            base_mean[::3] += 35

        # 类内变异度（临床样本的自然变异）
        if class_idx in [3, 11]:  # 非典型和原始细胞变异大
            intra_class_std = np.random.uniform(25, 45, n_features_raw)
        else:
            intra_class_std = np.random.uniform(15, 30, n_features_raw)

        # 生成该类别的样本
        for sample_idx in range(n_samples_class):
            # 基础样本
            sample = np.random.normal(base_mean, intra_class_std)

            # 添加结构化特征（模拟细胞形态）
            # 1. 核质比例特征
            nuclear_region = slice(n_features_raw//4, n_features_raw//2)
            cytoplasm_region = slice(n_features_raw//2, 3*n_features_raw//4)

            nuclear_intensity = np.random.uniform(1.1, 1.4)
            sample[nuclear_region] *= nuclear_intensity

            # 2. 颗粒特征（某些细胞类型）
            if class_idx in [5, 6, 7]:  # 嗜酸、嗜碱、早幼粒
                granule_indices = np.random.choice(n_features_raw, size=n_features_raw//8, replace=False)
                sample[granule_indices] += np.random.uniform(30, 60)

            # 3. 边缘模糊效应（模拟显微镜聚焦问题）
            if np.random.random() < 0.2:  # 20%的样本有边缘模糊
                edge_blur = np.random.normal(0, 8, n_features_raw//5)
                sample[:len(edge_blur)] += edge_blur

            # 4. 引入类间混淆（某些样本接近其他类别）
            for group in similarity_groups:
                if class_idx in group:
                    if np.random.random() < 0.15:  # 15%的样本有类间混淆
                        other_class = np.random.choice([c for c in group if c != class_idx])
                        # 添加其他类别的特征
                        confusion_strength = np.random.uniform(0.15, 0.35)
                        other_base = np.random.uniform(40, 180, n_features_raw)
                        sample = sample * (1 - confusion_strength) + other_base * confusion_strength

            # 5. 染色变异（模拟不同实验室的染色差异）
            staining_shift = np.random.normal(0, 12)
            sample += staining_shift

            # 6. 测量噪声（仪器噪声）
            measurement_noise = np.random.normal(0, 6, n_features_raw)
            sample += measurement_noise

            # 7. 光照不均（显微镜照明问题）
            if np.random.random() < 0.15:
                illumination_gradient = np.linspace(-8, 8, n_features_raw)
                np.random.shuffle(illumination_gradient)
                sample += illumination_gradient

            # 限制范围
            sample = np.clip(sample, 0, 255)

            X_list.append(sample)
            y_list.append(class_idx)

    X = np.array(X_list)
    y = np.array(y_list)

    # 打乱数据
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    print(f"\n数据集生成完成！")
    print(f"  - 数据形状: {X.shape}")
    print(f"  - 标签形状: {y.shape}")
    print(f"  - 数据范围: [{X.min():.1f}, {X.max():.1f}]")
    print(f"\n各类别样本数:")
    for i, (name, count) in enumerate(zip(class_names, samples_per_class)):
        print(f"    {i}. {name:25s}: {count:3d} 样本 ({count/n_samples*100:.1f}%)")

    return X, y, class_names


class AdaptiveMedicalKNNValidator:
    def __init__(self, k_range=(3, 20), n_components=60):
        """
        初始化自适应医学图像识别KNN验证器
        Args:
            k_range: K值搜索范围 (min_k, max_k) - 增大范围以应对更复杂数据
            n_components: PCA降维后的特征数量 - 增加到60维以保留更多信息
        """
        self.k_min, self.k_max = k_range
        self.n_components = n_components
        self.load_dataset()

    def load_dataset(self):
        """加载并预处理医学血细胞数据集"""
        print("="*70)
        print("加载医学血细胞分类数据集")
        print("="*70)

        # 生成医学数据集（1000个样本，12种细胞类型）
        X_raw, y, class_names = generate_medical_dataset(
            n_samples=1000,
            n_classes=12,
            n_features_raw=1024,
            random_state=42
        )

        self.class_names = class_names

        print(f"\n原始数据: {X_raw.shape[0]}个样本")
        print(f"细胞类型: {len(np.unique(y))}种")
        for i, name in enumerate(class_names):
            count = np.sum(y == i)
            print(f"  - {name}: {count}个样本")
        print(f"原始特征维度: {X_raw.shape[1]}")

        # PCA降维
        print(f"\n使用PCA降维至 {self.n_components} 维...")
        pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = pca.fit_transform(X_raw)

        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"保留方差: {explained_var:.2f}%")

        # 归一化到[0, 255]范围（8位量化）
        scaler = MinMaxScaler(feature_range=(0, 255))
        X_scaled = scaler.fit_transform(X_pca)
        self.X = np.round(X_scaled).astype(np.uint8)
        self.y = y

        # 划分训练集和测试集（80/20分割）
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"\n训练集: {self.X_train.shape[0]}个样本")
        print(f"测试集: {self.X_test.shape[0]}个样本")
        print(f"特征维度: {self.X_train.shape[1]}")
        print(f"数据类型: {self.X_train.dtype}, 范围: [{self.X_train.min()}, {self.X_train.max()}]")

    def l1_distance(self, x1, x2):
        """计算L1距离（曼哈顿距离）"""
        return np.sum(np.abs(x1.astype(np.int16) - x2.astype(np.int16)))

    def adaptive_knn_predict(self, query_point, return_details=False):
        """
        自适应K值预测
        """
        # 计算所有训练样本的距离
        distances = []
        for i in range(len(self.X_train)):
            dist = self.l1_distance(query_point, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])

        # 自适应K值选择算法
        best_k = self.k_min
        max_confidence = -1
        best_prediction = None
        k_confidences = []

        for k in range(self.k_min, min(self.k_max + 1, len(distances))):
            k_neighbors = distances[:k]
            k_labels = [label for _, label in k_neighbors]
            k_dists = [dist for dist, _ in k_neighbors]

            # 统计投票
            label_counts = Counter(k_labels)
            majority_class, majority_count = label_counts.most_common(1)[0]

            # 计算置信度
            vote_ratio = majority_count / k

            # 距离权重
            weights = [1.0 / (1.0 + d) for d in k_dists]
            weighted_votes = sum([weights[i] for i in range(k) if k_labels[i] == majority_class])
            total_weights = sum(weights)
            distance_weight = weighted_votes / (total_weights + 1e-6)

            # 决策边界间隙
            dist_in_class_max = 0
            dist_out_class_min = float('inf')
            found_out_class = False

            for i in range(k):
                if k_labels[i] == majority_class:
                    if k_dists[i] > dist_in_class_max:
                        dist_in_class_max = k_dists[i]
                else:
                    found_out_class = True
                    if k_dists[i] < dist_out_class_min:
                        dist_out_class_min = k_dists[i]

            gap_bonus = 1.0
            if not found_out_class:
                gap_bonus = 2.0
            elif dist_out_class_min > dist_in_class_max:
                gap = dist_out_class_min - dist_in_class_max
                gap_bonus = 1.0 + min(0.5, gap / (dist_in_class_max + 1e-6))

            # 综合置信度
            confidence = (vote_ratio + distance_weight) * gap_bonus

            k_confidences.append((k, confidence, majority_class))

            if confidence > max_confidence:
                max_confidence = confidence
                best_k = k
                best_prediction = majority_class

        # 如果置信度极低，使用最小K
        if max_confidence < 0.1:
            for k, conf, pred in k_confidences:
                if k == self.k_min:
                    best_prediction = pred
                    best_k = self.k_min
                    max_confidence = conf
                    break

        if return_details:
            return best_prediction, best_k, max_confidence, k_confidences, distances[:best_k]
        return best_prediction, best_k, max_confidence

    def evaluate(self, verbose=True):
        """评估自适应KNN性能"""
        if verbose:
            print("\n" + "="*70)
            print("自适应K值L1-KNN 医学血细胞分类性能评估")
            print("="*70)

        predictions = []
        selected_ks = []
        confidences = []

        start_time = time.time()

        for i, test_point in enumerate(self.X_test):
            pred, k_used, conf = self.adaptive_knn_predict(test_point)
            predictions.append(pred)
            selected_ks.append(k_used)
            confidences.append(conf)

            if verbose and i < 5:
                match = "✓" if pred == self.y_test[i] else "✗"
                true_class = self.class_names[self.y_test[i]]
                pred_class = self.class_names[pred]
                print(f"样本{i+1} {match}: 预测={pred_class}, 实际={true_class}, "
                      f"K={k_used}, 置信度={conf:.3f}")

        total_time = time.time() - start_time

        predictions = np.array(predictions)
        accuracy = np.mean(predictions == self.y_test)
        avg_k = np.mean(selected_ks)
        avg_confidence = np.mean(confidences)

        # Top-3准确率
        top3_correct = 0
        for i, test_point in enumerate(self.X_test):
            _, _, _, k_confs, _ = self.adaptive_knn_predict(test_point, return_details=True)
            sorted_by_conf = sorted(k_confs, key=lambda x: x[1], reverse=True)
            top3_labels = []
            for _, _, pred_label in sorted_by_conf:
                if pred_label not in top3_labels:
                    top3_labels.append(pred_label)
                if len(top3_labels) >= 3:
                    break
            if self.y_test[i] in top3_labels:
                top3_correct += 1
        top3_accuracy = top3_correct / len(self.y_test)

        if verbose:
            print(f"\n【性能指标】")
            print(f"Top-1 准确率: {accuracy*100:.2f}%")
            print(f"Top-3 准确率: {top3_accuracy*100:.2f}%")
            print(f"平均使用K值: {avg_k:.2f}")
            print(f"平均置信度: {avg_confidence:.3f}")
            print(f"总预测时间: {total_time:.3f}秒")
            print(f"单样本识别时间: {total_time/len(self.X_test)*1000:.2f}ms")

        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'avg_k': avg_k,
            'avg_confidence': avg_confidence,
            'k_distribution': selected_ks,
            'confidences': confidences,
            'predictions': predictions
        }

    def compare_with_fixed_k(self):
        """对比自适应K与固定K的性能"""
        print("\n" + "="*70)
        print("自适应K vs 固定K 医学血细胞分类性能对比")
        print("="*70)

        adaptive_results = self.evaluate()

        # 测试多个固定K值
        fixed_k_results = {}
        test_k_values = [3, 5, 7, 9, 11]

        for fixed_k in test_k_values:
            correct = 0
            for test_point, true_label in zip(self.X_test, self.y_test):
                distances = []
                for train_point, train_label in zip(self.X_train, self.y_train):
                    dist = self.l1_distance(test_point, train_point)
                    distances.append((dist, train_label))

                distances.sort(key=lambda x: x[0])
                k_labels = [label for _, label in distances[:fixed_k]]
                pred = Counter(k_labels).most_common(1)[0][0]

                if pred == true_label:
                    correct += 1

            accuracy = correct / len(self.X_test)
            fixed_k_results[fixed_k] = accuracy
            print(f"固定K={fixed_k:2d} 准确率: {accuracy*100:.2f}%")

        print(f"\n自适应K 准确率: {adaptive_results['accuracy']*100:.2f}%")
        best_fixed_k = max(fixed_k_results, key=fixed_k_results.get)
        best_fixed_acc = fixed_k_results[best_fixed_k]
        improvement = (adaptive_results['accuracy'] - best_fixed_acc) * 100
        print(f"最佳固定K={best_fixed_k} 准确率: {best_fixed_acc*100:.2f}%")
        print(f"相对提升: {improvement:+.2f}%")

        return adaptive_results, fixed_k_results

    def visualize_results(self, adaptive_results, fixed_k_results):
        """可视化分析结果"""
        fig = plt.figure(figsize=(15, 10))

        # 1. K值分布直方图
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(adaptive_results['k_distribution'],
                 bins=range(self.k_min, self.k_max+2),
                 edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Selected K' if not USE_CHINESE else '选择的K值', fontsize=11)
        plt.ylabel('Frequency' if not USE_CHINESE else '频次', fontsize=11)
        plt.title('Adaptive K Distribution' if not USE_CHINESE else '自适应K值分布 (医学细胞分类)',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 2. 置信度分布
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(adaptive_results['confidences'], bins=30,
                 edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Confidence Score' if not USE_CHINESE else '置信度分数', fontsize=11)
        plt.ylabel('Frequency' if not USE_CHINESE else '频次', fontsize=11)
        plt.title('Confidence Distribution' if not USE_CHINESE else '预测置信度分布',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 3. 固定K vs 自适应K准确率对比
        ax3 = plt.subplot(2, 3, 3)
        k_vals = sorted(fixed_k_results.keys())
        fixed_accs = [fixed_k_results[k]*100 for k in k_vals]
        plt.plot(k_vals, fixed_accs, 'o-', linewidth=2, markersize=8,
                 label='Fixed K' if not USE_CHINESE else '固定K', color='orange')
        plt.axhline(y=adaptive_results['accuracy']*100, color='red',
                    linestyle='--', linewidth=2,
                    label='Adaptive K' if not USE_CHINESE else '自适应K')
        plt.xlabel('K Value' if not USE_CHINESE else 'K值', fontsize=11)
        plt.ylabel('Accuracy (%)' if not USE_CHINESE else '准确率 (%)', fontsize=11)
        plt.title('Fixed K vs Adaptive K' if not USE_CHINESE else '固定K vs 自适应K',
                  fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 4. 混淆矩阵
        ax4 = plt.subplot(2, 3, 4)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, adaptive_results['predictions'])
        im = plt.imshow(cm, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax4)
        plt.xlabel('Predicted Class' if not USE_CHINESE else '预测类别', fontsize=11)
        plt.ylabel('True Class' if not USE_CHINESE else '真实类别', fontsize=11)
        plt.title('Confusion Matrix' if not USE_CHINESE else '混淆矩阵',
                  fontsize=12, fontweight='bold')

        # 5. 置信度 vs 准确性
        ax5 = plt.subplot(2, 3, 5)
        correct = (adaptive_results['predictions'] == self.y_test).astype(int)
        colors = ['green' if c else 'red' for c in correct]
        plt.scatter(range(len(correct)), adaptive_results['confidences'],
                    c=colors, alpha=0.6, s=50)
        plt.xlabel('Test Sample Index' if not USE_CHINESE else '测试样本索引', fontsize=11)
        plt.ylabel('Confidence' if not USE_CHINESE else '置信度', fontsize=11)
        plt.title('Confidence vs Correctness' if not USE_CHINESE else '置信度 vs 分类正确性',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 6. K值 vs 准确性
        ax6 = plt.subplot(2, 3, 6)
        k_acc = {}
        for k, pred, true in zip(adaptive_results['k_distribution'],
                                   adaptive_results['predictions'],
                                   self.y_test):
            if k not in k_acc:
                k_acc[k] = {'correct': 0, 'total': 0}
            k_acc[k]['total'] += 1
            if pred == true:
                k_acc[k]['correct'] += 1

        k_vals_used = sorted(k_acc.keys())
        k_accs = [k_acc[k]['correct']/k_acc[k]['total']*100 for k in k_vals_used]
        k_counts = [k_acc[k]['total'] for k in k_vals_used]

        plt.scatter(k_vals_used, k_accs, s=[c*10 for c in k_counts],
                    alpha=0.6, color='purple')
        plt.xlabel('K Value Used' if not USE_CHINESE else '使用的K值', fontsize=11)
        plt.ylabel('Accuracy (%)' if not USE_CHINESE else '该K值下的准确率 (%)', fontsize=11)
        title_text = 'Accuracy by K (bubble size = usage count)' if not USE_CHINESE else '不同K值的准确率 (气泡大小=使用次数)'
        plt.title(title_text, fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = 'adaptive_medical_knn_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n分析图表已保存: {filename}")

    def export_hardware_dataset(self, output_file='medical_hardware_dataset.txt'):
        """导出用于硬件测试的医学数据集"""
        print(f"\n导出硬件测试数据集: {output_file}")

        with open(output_file, 'w') as f:
            f.write("# Adaptive K-NN Medical Blood Cell Classification Hardware Dataset\n")
            f.write(f"# Train samples: {self.X_train.shape[0]}\n")
            f.write(f"# Test samples: {self.X_test.shape[0]}\n")
            f.write(f"# Features: {self.X_train.shape[1]}\n")
            f.write(f"# Cell types: {len(np.unique(self.y_train))}\n")
            f.write(f"# K_range: [{self.k_min}, {self.k_max}]\n")
            f.write(f"# Cell type mapping:\n")
            for i, name in enumerate(self.class_names):
                f.write(f"#   {i}: {name}\n")
            f.write(f"# Data format: feature1 feature2 ... featureN cell_type_id\n")
            f.write("#\n")

            f.write("# TRAINING SET\n")
            for i in range(len(self.X_train)):
                features = ' '.join(map(str, self.X_train[i]))
                f.write(f"{features} {self.y_train[i]}\n")

            f.write("# TEST SET\n")
            for i in range(len(self.X_test)):
                features = ' '.join(map(str, self.X_test[i]))
                f.write(f"{features} {self.y_test[i]}\n")

        print(f"导出完成! 训练集: {len(self.X_train)}个, 测试集: {len(self.X_test)}个")

    def generate_report(self):
        """生成完整的算法验证报告"""
        print("\n" + "="*70)
        print("自适应K值L1-KNN 医学血细胞分类系统 - 算法验证报告")
        print("="*70)

        adaptive_results, fixed_k_results = self.compare_with_fixed_k()
        self.visualize_results(adaptive_results, fixed_k_results)
        self.export_hardware_dataset()

        print("\n【数据集信息】")
        print(f"✓ 医学样本总数: {len(self.X)}个血细胞图像")
        print(f"✓ 训练样本: {len(self.X_train)}个")
        print(f"✓ 测试样本: {len(self.X_test)}个")
        print(f"✓ 细胞类型: {len(np.unique(self.y))}种")
        print(f"✓ 特征维度: {self.X_train.shape[1]} (PCA降维后)")
        print(f"✓ 数据位宽: 8-bit")

        return adaptive_results


def main():
    print("="*70)
    print("自适应K值 L1-KNN 医学血细胞分类算法验证")
    print("="*70)

    validator = AdaptiveMedicalKNNValidator(k_range=(3, 20), n_components=60)
    results = validator.generate_report()

    print("\n" + "="*70)
    print("验证完成! 所有结果已保存.")
    print("生成文件:")
    print("  - adaptive_medical_knn_analysis.png (分析图表)")
    print("  - medical_hardware_dataset.txt (硬件测试数据)")
    print("="*70)


if __name__ == "__main__":
    main()