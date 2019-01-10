import os
from tensorflow.python import pywrap_tensorflow

# checkpoint_path = os.path.join("/home/cwd/project/PoseEstimationForMobile/trained_v5/mv2_stacked_hourglass_2x3_7W_1.2box_128x256/mv2_stacked_hourglass_2x3_batch-32_lr-0.001_gpus-1_128x256_experiments-mv2_stacked_hourglass_2x3/model-200")
checkpoint_path = os.path.join("/home/cwd/project/PoseEstimationForMobile/trained_v4/mv2_hourglass_multi_1.05/models/mv2_hourglass_batch-32_lr-0.002_gpus-1_128x256_experiments-mv2_hourglass_multi/model-5000")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)

