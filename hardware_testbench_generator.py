"""
硬件测试文件生成器 - 医学血细胞分类
生成Verilog testbench、BRAM初始化文件、C头文件
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def generate_medical_dataset(n_samples=1000, n_classes=12, n_features_raw=1024, random_state=42):
    """生成真实感的医学血细胞图像数据集"""
    np.random.seed(random_state)

    class_names = [
        'Neutrophil-Band', 'Neutrophil-Segmented', 'Lymphocyte-Typical',
        'Lymphocyte-Atypical', 'Monocyte', 'Eosinophil', 'Basophil',
        'Promyelocyte', 'Myelocyte', 'Metamyelocyte', 'Erythroblast', 'Blast'
    ]

    class_weights = np.array([0.12, 0.15, 0.14, 0.08, 0.10, 0.06, 0.03, 0.08, 0.07, 0.06, 0.07, 0.04])
    samples_per_class = (class_weights * n_samples).astype(int)
    samples_per_class[-1] = n_samples - samples_per_class[:-1].sum()

    X_list = []
    y_list = []

    similarity_groups = [[0, 1], [2, 3], [7, 8, 9]]

    for class_idx in range(n_classes):
        n_samples_class = samples_per_class[class_idx]
        base_mean = np.random.uniform(40, 180, n_features_raw)

        if class_idx in [0, 1]:
            base_mean[:n_features_raw//4] += 30
        elif class_idx in [2, 3]:
            base_mean[n_features_raw//4:n_features_raw//2] += 40
        elif class_idx == 4:
            base_mean[n_features_raw//3:2*n_features_raw//3] += 25
        elif class_idx == 5:
            base_mean[::3] += 35

        if class_idx in [3, 11]:
            intra_class_std = np.random.uniform(25, 45, n_features_raw)
        else:
            intra_class_std = np.random.uniform(15, 30, n_features_raw)

        for sample_idx in range(n_samples_class):
            sample = np.random.normal(base_mean, intra_class_std)

            nuclear_region = slice(n_features_raw//4, n_features_raw//2)
            nuclear_intensity = np.random.uniform(1.1, 1.4)
            sample[nuclear_region] *= nuclear_intensity

            if class_idx in [5, 6, 7]:
                granule_indices = np.random.choice(n_features_raw, size=n_features_raw//8, replace=False)
                sample[granule_indices] += np.random.uniform(30, 60)

            if np.random.random() < 0.2:
                edge_blur = np.random.normal(0, 8, n_features_raw//5)
                sample[:len(edge_blur)] += edge_blur

            for group in similarity_groups:
                if class_idx in group:
                    if np.random.random() < 0.15:
                        other_class = np.random.choice([c for c in group if c != class_idx])
                        confusion_strength = np.random.uniform(0.15, 0.35)
                        other_base = np.random.uniform(40, 180, n_features_raw)
                        sample = sample * (1 - confusion_strength) + other_base * confusion_strength

            staining_shift = np.random.normal(0, 12)
            sample += staining_shift

            measurement_noise = np.random.normal(0, 6, n_features_raw)
            sample += measurement_noise

            if np.random.random() < 0.15:
                illumination_gradient = np.linspace(-8, 8, n_features_raw)
                np.random.shuffle(illumination_gradient)
                sample += illumination_gradient

            sample = np.clip(sample, 0, 255)

            X_list.append(sample)
            y_list.append(class_idx)

    X = np.array(X_list)
    y = np.array(y_list)

    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y, class_names


class MedicalHardwareGenerator:
    def __init__(self, n_components=60):
        """初始化硬件文件生成器"""
        self.n_components = n_components
        self.load_dataset()

    def load_dataset(self):
        """加载数据集"""
        print("=" * 70)
        print("硬件测试文件生成器 - 医学血细胞分类")
        print("=" * 70)

        X_raw, y, class_names = generate_medical_dataset(
            n_samples=1000, n_classes=12, n_features_raw=1024, random_state=42
        )

        self.class_names = class_names

        pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = pca.fit_transform(X_raw)

        scaler = MinMaxScaler(feature_range=(0, 255))
        X_scaled = scaler.fit_transform(X_pca)
        self.X = np.round(X_scaled).astype(np.uint8)
        self.y = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"训练集: {len(self.X_train)} 张细胞图像")
        print(f"测试集: {len(self.X_test)} 张细胞图像")
        print(f"特征维度: {self.X_train.shape[1]}")

    def generate_bram_mem(self, filename='medical_train_data_bram.mem'):
        """生成BRAM初始化文件（.mem格式）"""
        print(f"\n生成BRAM初始化文件: {filename}")

        with open(filename, 'w') as f:
            f.write(f"// Medical Blood Cell Classification Training Data - BRAM Format\n")
            f.write(f"// Generated by medical hardware generator\n")
            f.write(f"// Total Samples: {len(self.X_train)}\n")
            f.write(f"// Features per Sample: {self.X_train.shape[1]}\n")
            f.write(f"// Data Width: 8-bit (uint8)\n")
            f.write(f"// Memory Organization: [label][feature0][feature1]...[featureN]\n")
            f.write(f"// Cell type mapping:\n")
            for i, name in enumerate(self.class_names):
                f.write(f"//   {i}: {name}\n")
            f.write(f"// Each sample occupies {self.X_train.shape[1] + 1} addresses\n\n")

            for i, (features, label) in enumerate(zip(self.X_train, self.y_train)):
                base_addr = i * (self.X_train.shape[1] + 1)

                f.write(f"@{base_addr:08X} {label:02X}  // Sample {i}: {self.class_names[label]}\n")

                for j in range(0, len(features), 8):
                    chunk = features[j:min(j+8, len(features))]
                    for k, feature in enumerate(chunk):
                        addr = base_addr + j + k + 1
                        f.write(f"@{addr:08X} {feature:02X}\n")

        print(f"✓ 已生成 {filename}")
        print(f"  - 总地址数: {len(self.X_train) * (self.X_train.shape[1] + 1)}")

    def generate_coe_file(self, filename='medical_train_data.coe'):
        """生成Vivado COE文件"""
        print(f"\n生成Vivado COE文件: {filename}")

        with open(filename, 'w') as f:
            f.write("; Medical Blood Cell Classification Training Data - Vivado COE Format\n")
            f.write(f"; Total Samples: {len(self.X_train)}\n")
            f.write(f"; Features: {self.X_train.shape[1]}\n")
            f.write("; Data format: [label][features...]\n")
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")

            data_list = []
            for features, label in zip(self.X_train, self.y_train):
                data_list.append(f"{label:02X}")
                for feature in features:
                    data_list.append(f"{feature:02X}")

            for i, data in enumerate(data_list):
                if i == len(data_list) - 1:
                    f.write(f"{data};")
                else:
                    f.write(f"{data},")
                    if (i + 1) % 16 == 0:
                        f.write("\n")

        print(f"✓ 已生成 {filename}")

    def generate_test_vectors(self, filename='medical_test_vectors.mem', num_tests=30):
        """生成Verilog testbench测试向量"""
        print(f"\n生成测试向量文件: {filename}")

        test_samples = self.X_test[:num_tests]
        test_labels = self.y_test[:num_tests]

        with open(filename, 'w') as f:
            f.write("// Medical Blood Cell Classification Test Vectors\n")
            f.write(f"// Number of Tests: {num_tests}\n")
            f.write(f"// Features: {self.X_train.shape[1]}\n")
            f.write("// Format: test_id | features (hex) | expected_label\n")
            f.write("// Cell type mapping:\n")
            for i, name in enumerate(self.class_names):
                f.write(f"//   {i}: {name}\n")
            f.write("// Usage: Read this file in your Verilog testbench\n\n")

            for i, (features, label) in enumerate(zip(test_samples, test_labels)):
                f.write(f"// ============================================\n")
                f.write(f"// Test {i + 1}: Expected Cell Type = {self.class_names[label]}\n")
                f.write(f"// ============================================\n")

                for j in range(0, len(features), 8):
                    chunk = features[j:j + 8]
                    hex_str = ' '.join([f"{x:02X}" for x in chunk])
                    f.write(f"{hex_str}\n")

                f.write(f"EXPECTED: {label:02X}\n")
                f.write("\n")

        print(f"✓ 已生成 {filename}")
        print(f"  - 测试样本数: {num_tests}")

    def generate_c_header(self, filename='medical_knn_data.h'):
        """生成ARM端C程序头文件"""
        print(f"\n生成C头文件: {filename}")

        with open(filename, 'w') as f:
            f.write("/* Medical Blood Cell Classification KNN Data - ARM C Header */\n")
            f.write("/* Generated by medical hardware generator */\n")
            f.write("#ifndef MEDICAL_KNN_DATA_H\n")
            f.write("#define MEDICAL_KNN_DATA_H\n\n")
            f.write("#include <stdint.h>\n\n")

            f.write("// Dataset dimensions\n")
            f.write(f"#define N_TRAIN_SAMPLES {len(self.X_train)}\n")
            f.write(f"#define N_TEST_SAMPLES {len(self.X_test)}\n")
            f.write(f"#define N_FEATURES {self.X_train.shape[1]}\n")
            f.write(f"#define N_CELL_TYPES {len(np.unique(self.y_train))}\n\n")

            f.write("// Adaptive K-NN parameters\n")
            f.write(f"#define K_MIN 3\n")
            f.write(f"#define K_MAX 20\n\n")

            f.write("// Cell type names\n")
            f.write("const char* cell_type_names[N_CELL_TYPES] = {\n")
            for i, name in enumerate(self.class_names):
                comma = "," if i < len(self.class_names) - 1 else ""
                f.write(f"    \"{name}\"{comma}\n")
            f.write("};\n\n")

            f.write("// Training data sample (first 10 samples)\n")
            f.write("const uint8_t train_samples_sample[10][N_FEATURES] = {\n")
            for i in range(min(10, len(self.X_train))):
                f.write("    {")
                f.write(", ".join([f"{x}" for x in self.X_train[i]]))
                f.write("}," if i < 9 else "}\n")
                if i < 9:
                    f.write(f"  // {self.class_names[self.y_train[i]]}\n")
            f.write("};\n\n")

            f.write("const uint8_t train_labels_sample[10] = {")
            f.write(", ".join([f"{self.y_train[i]}" for i in range(min(10, len(self.X_train)))]))
            f.write("};\n\n")

            f.write("// Test data sample (first 5 samples)\n")
            f.write("const uint8_t test_samples_sample[5][N_FEATURES] = {\n")
            for i in range(min(5, len(self.X_test))):
                f.write("    {")
                f.write(", ".join([f"{x}" for x in self.X_test[i]]))
                f.write("}," if i < 4 else "}\n")
                if i < 4:
                    f.write(f"  // Expected: {self.class_names[self.y_test[i]]}\n")
            f.write("};\n\n")

            f.write("const uint8_t test_labels_sample[5] = {")
            f.write(", ".join([f"{self.y_test[i]}" for i in range(min(5, len(self.X_test)))]))
            f.write("};\n\n")

            f.write("// Helper functions (implement in your C code)\n")
            f.write("uint16_t l1_distance(const uint8_t* sample1, const uint8_t* sample2, uint16_t n_features);\n")
            f.write("uint8_t adaptive_knn_predict(const uint8_t* query_sample, uint8_t* confidence);\n")
            f.write("float calculate_confidence(uint8_t k, const uint16_t* distances, const uint8_t* labels);\n\n")

            f.write("#endif // MEDICAL_KNN_DATA_H\n")

        print(f"✓ 已生成 {filename}")

    def generate_verilog_params(self, filename='medical_knn_params.vh'):
        """生成Verilog参数文件"""
        print(f"\n生成Verilog参数文件: {filename}")

        with open(filename, 'w') as f:
            f.write("// Medical Blood Cell Classification KNN Accelerator - Parameters\n")
            f.write("// Auto-generated parameter file\n\n")

            f.write("// Dataset parameters\n")
            f.write(f"parameter N_TRAIN = {len(self.X_train)};\n")
            f.write(f"parameter N_TEST = {len(self.X_test)};\n")
            f.write(f"parameter D = {self.X_train.shape[1]};  // Feature dimension\n")
            f.write(f"parameter N_CLASSES = {len(np.unique(self.y_train))};\n\n")

            f.write("// Adaptive K-NN parameters\n")
            f.write(f"parameter K_MIN = 3;\n")
            f.write(f"parameter K_MAX = 20;\n\n")

            f.write("// Data widths\n")
            f.write("parameter FEATURE_WIDTH = 8;  // 8-bit quantized features\n")
            max_dist = self.X_train.shape[1] * 255
            dist_width = int(np.ceil(np.log2(max_dist + 1)))
            f.write(f"parameter DISTANCE_WIDTH = {dist_width};  // Max L1 distance = {max_dist}\n")
            f.write("parameter LABEL_WIDTH = 8;  // Cell type ID\n")
            f.write("parameter CONFIDENCE_WIDTH = 16;  // Fixed-point Q8.8 for confidence\n\n")

            f.write("// Memory addressing\n")
            train_addr_width = int(np.ceil(np.log2(len(self.X_train) * (self.X_train.shape[1] + 1))))
            f.write(f"parameter TRAIN_ADDR_WIDTH = {train_addr_width};  // BRAM address width\n")
            f.write(f"parameter FEATURE_ADDR_WIDTH = {int(np.ceil(np.log2(self.X_train.shape[1])))};\n\n")

            f.write("// Pipeline stages\n")
            f.write("parameter L1_PIPELINE_STAGES = 3;\n")
            f.write("parameter SORT_PIPELINE_STAGES = 2;\n")

        print(f"✓ 已生成 {filename}")

    def generate_all_files(self):
        """生成所有硬件文件"""
        print("\n" + "=" * 70)
        print("开始生成所有硬件测试文件")
        print("=" * 70)

        self.generate_bram_mem()
        self.generate_coe_file()
        self.generate_test_vectors(num_tests=30)
        self.generate_c_header()
        self.generate_verilog_params()

        print("\n" + "=" * 70)
        print("文件生成完成!")
        print("=" * 70)
        print("\n生成的文件列表:")
        print("  1. medical_train_data_bram.mem  - BRAM仿真初始化文件")
        print("  2. medical_train_data.coe       - Vivado综合初始化文件")
        print("  3. medical_test_vectors.mem     - Verilog testbench测试向量")
        print("  4. medical_knn_data.h           - ARM端C程序头文件")
        print("  5. medical_knn_params.vh        - Verilog参数定义文件")


def main():
    generator = MedicalHardwareGenerator(n_components=60)
    generator.generate_all_files()


if __name__ == "__main__":
    main()