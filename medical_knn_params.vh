// Medical Blood Cell Classification KNN Accelerator - Parameters
// Auto-generated parameter file

// Dataset parameters
parameter N_TRAIN = 800;
parameter N_TEST = 200;
parameter D = 60;  // Feature dimension
parameter N_CLASSES = 12;

// Adaptive K-NN parameters
parameter K_MIN = 3;
parameter K_MAX = 20;

// Data widths
parameter FEATURE_WIDTH = 8;  // 8-bit quantized features
parameter DISTANCE_WIDTH = 14;  // Max L1 distance = 15300
parameter LABEL_WIDTH = 8;  // Cell type ID
parameter CONFIDENCE_WIDTH = 16;  // Fixed-point Q8.8 for confidence

// Memory addressing
parameter TRAIN_ADDR_WIDTH = 16;  // BRAM address width
parameter FEATURE_ADDR_WIDTH = 6;

// Pipeline stages
parameter L1_PIPELINE_STAGES = 3;
parameter SORT_PIPELINE_STAGES = 2;
