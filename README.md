基于 FPGA 的自适应 K 值 L1-KNN 医学血细胞分类加速器

项目简介

本项目实现了一个用于医学血细胞分类的 K-最近邻（KNN）算法加速器。该项目旨在解决医学显微图像分析中面临的高噪声、形态细微差异（如杆状核与分叶核中性粒细胞的区分）以及类别分布不平衡等挑战。

与传统的 KNN 加速器不同，本项目针对医学应用场景进行了深度优化，具有两大核心创新点：

DSP-Free 硬件设计：采用**曼哈顿距离（L1-Norm）**代替传统的欧氏距离（L2-Norm）。这种设计使得距离计算完全不依赖 FPGA 中的昂贵乘法器资源（DSP），仅通过加法器和查找表（LUT）即可高效实现，从而极大地提高了能效比和逻辑资源利用率，非常适合低功耗的便携式医疗设备。

自适应 K 值决策机制 (Adaptive-K)：针对医学图像中类间重叠严重的问题，提出了一种动态 K 值选择算法。该算法不再使用固定的 K 值，而是根据每个查询样本周围邻居的分布情况（距离间隙、类别一致性），动态计算一个**“综合置信度”**，并自动调整最优的 K 值进行加权投票。

实验成果

实验结果表明（详见下文图表），在处理复杂的医学血细胞数据时，“自适应 K 值”算法的分类准确率显著优于任何单一固定 K 值。同时，FPGA 硬件加速方案相比纯软件实现，预计可达到 1000 倍以上的推理速度提升。

性能分析图表

(图：自适应 K 值算法在医学数据集上的性能表现，展示了 K 值分布、置信度以及与固定 K 值的对比)

软件基准对比

(图：Python、NumPy 与 FPGA 估算性能的对比，展示了 FPGA 在延迟和吞吐量上的巨大优势)

代码库结构

本代码库包含用于算法验证、软件基准测试以及硬件初始化文件生成的完整 Python 脚本集。

.
├── knn_algorithm_validator.py          # [核心] 自适应 K 值 KNN 算法验证器
│                                       # - 生成医学血细胞模拟数据集
│                                       # - 验证自适应算法的准确率与鲁棒性
│                                       # - 生成 adaptive_medical_knn_analysis.png
│
├── software_baseline_benchmark.py      # [基准] 软件性能基准测试
│                                       # - 测试 Python 和 NumPy 实现的延迟与吞吐量
│                                       # - 估算 FPGA 硬件加速性能
│                                       # - 生成 medical_software_baseline_comparison.png
│
├── hardware_testbench_generator.py     # [硬件] 硬件文件生成器
│                                       # - 生成用于 Verilog 仿真和综合的 .mem/.coe 文件
│                                       # - 生成 Verilog 参数头文件 (.vh) 和 C 语言头文件 (.h)
│
├── medical_knn_params.vh               # [已生成] Verilog 硬件参数定义 (N_TRAIN, K_MAX 等)
├── medical_knn_data.h                  # [已生成] ARM 端 C 语言头文件 (包含测试数据样本)
├── medical_train_data_bram.mem         # [已生成] BRAM 仿真初始化文件 (训练集)
├── medical_train_data.coe              # [已生成] Vivado BRAM IP 核初始化文件 (训练集)
├── medical_test_vectors.mem            # [已生成] Verilog Testbench 测试向量 (测试集)
└── medical_hardware_dataset.txt        # [已生成] 导出供硬件调试的完整文本数据集


快速开始

本项目依赖 numpy, scikit-learn 和 matplotlib。

1. 环境准备

确保已安装必要的 Python 库：

pip install numpy scikit-learn matplotlib


2. 运行算法验证

验证自适应 K 值算法在医学数据上的有效性，并生成分析图表：

python knn_algorithm_validator.py


输出：adaptive_medical_knn_analysis.png, medical_hardware_dataset.txt

3. 运行基准测试

评估软件实现的性能瓶颈，并估算 FPGA 加速潜力：

python software_baseline_benchmark.py


输出：medical_software_baseline_comparison.png

4. 生成硬件文件

生成用于 Vivado 工程和 Verilog 仿真的所有必要文件：

python hardware_testbench_generator.py


输出：medical_train_data.coe, medical_train_data_bram.mem, medical_test_vectors.mem, medical_knn_params.vh, medical_knn_data.h

硬件开发指南 (FPGA)

生成的硬件文件可直接用于 FPGA 开发流程：

Vivado 综合：在 Block Memory Generator (BRAM) IP 核配置中，加载 medical_train_data.coe 文件以初始化训练数据库。

RTL 仿真：在 Verilog Testbench 中使用 $readmemh 读取 medical_train_data_bram.mem 和 medical_test_vectors.mem，进行功能验证。

参数配置：在顶层 Verilog 模块中包含 medical_knn_params.vh，以自动同步特征维度（D=60）、样本数量等关键参数。

嵌入式软件：在 Zynq PS 端的 C 代码中包含 medical_knn_data.h，获取类别名称映射和测试数据样本。

数据集说明

本项目使用脚本生成的模拟医学血细胞数据集，模拟了真实临床数据的以下特征：

12 类细胞：包括不同发育阶段的中性粒细胞（杆状核、分叶核）、淋巴细胞、单核细胞等。

高维特征：PCA 降维后的 60 维特征向量，保留了丰富的形态学信息。

类间混淆：模拟了形态相似细胞（如杆状核 vs 分叶核）之间的特征重叠。

类别不平衡：模拟了真实血液样本中各细胞类型的自然分布比例。
