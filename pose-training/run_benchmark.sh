python3 ./src/benchmark.py --frozen_pb_path=/home/cwd/project/PoseEstimationForMobile/training/graph/SHG_lv2_conv2_3.pb \
--prefix=lv2_conv2_3 \
--output_node_name=l1_out/BiasAdd

python3 ./src/benchmark.py --frozen_pb_path=/home/cwd/project/PoseEstimationForMobile/training/graph/SHG_lv2_conv3_backbone3.pb \
--prefix=lv2_conv3_b \
--output_node_name=l1_out/BiasAdd

python3 ./src/benchmark.py --frozen_pb_path=/home/cwd/project/PoseEstimationForMobile/training/graph/SHG_lv3_conv1_2.pb \
--prefix=lv3_conv1_2 \
--output_node_name=l2_out/BiasAdd

python3 ./src/benchmark.py --frozen_pb_path=/home/cwd/project/PoseEstimationForMobile/training/graph/SHG_lv3_conv1.pb \
--prefix=lv3_conv1 \
--output_node_name=l2_out/BiasAdd

python3 ./src/benchmark.py --frozen_pb_path=/home/cwd/project/PoseEstimationForMobile/training/graph/SHG_lv4_conv1.pb \
--prefix=lv4_conv1 \
--output_node_name=l3_out/BiasAdd

