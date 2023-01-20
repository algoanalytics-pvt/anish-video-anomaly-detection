# Dataset Configuration Parameters
dataset = ""
train_dataset_folder = ""
validation_dataset_folder = ""
test_dataset_normal_folder = ""
test_dataset_anomalous_folder = ""
test_dataset_sequential_folder = ""
labels_path = ""

# Evaluation Configuration Parameters
evaluation_plot_extension = ".png"
evaluation_plot_location = (
    "../evaluation_plots/" + dataset + "_evaluation" + evaluation_plot_extension
)
histogram_location = (
    "../evaluation_plots/" + dataset + "_histogram" + evaluation_plot_extension
)
auprc_curve_location = (
    "../evaluation_plots/" + dataset + "_auprc" + evaluation_plot_extension
)
plot_vertical_limit = 5
normal_losses_save_path = "../losses/" + dataset + "_normal_losses.npy"
anomalous_losses_save_path = "../losses/" + dataset + "_anomalous_losses.npy"
losses_path = "../losses/" + dataset + "_losses.npy"

# image configuration
image_size = 224
input_channels = 3
output_channels = 3
image_dimensions = (image_size, image_size)
color_mode = "rgb"
class_mode = None
shuffle = True

# threshold configuration
threshold = 0.55

# model Saving configuration
models_base_folder = "../models/"
weights_location = models_base_folder + dataset + "_weights.h5"
model_location = models_base_folder + dataset + "_model"
tflite_model_location = models_base_folder + dataset + "_model.tflite"

# Model HyperParameters
patience = 3
bottleneck_size = 28
batch_size = 1
learning_rate = 0.000001
epochs = 100

# Keras summary configuration
summary_line_length = 125

# training
create_data_generators = False
augmentations = False

# live running configuration
save_inputs_and_reconstructions = True
localization_boxes = False
eager_execution = False
multithreaded_execution = False
tflite_threads = 4
window_name = "Anomaly Detection"
window_height = 700
window_width = 700
model_input_height = 224
model_input_width = 224
box_dims = 4
urls = [""]
thresholds = [0]
weights = ["", "", "", ""]
output_size = 700
