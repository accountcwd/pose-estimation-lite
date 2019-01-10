python3 src/gen_frozen_pb.py --model=network_SHG_lv2_conv2_3 \
--checkpoint=/home/cwd/project/PoseEstimationForMobile/trained_v6/network_SHG_lv2_conv2_3_batch-32_lr-0.001_gpus-1_128x256_experiments-network_SHG_lv2_conv2_3/model-112000 \
--output_node_names=l1_out/BiasAdd \
--output_graph=SHG_lv2_conv2_3.pb 

python3 src/gen_frozen_pb.py --model=network_SHG_lv2_conv3_backbone3 \
--checkpoint=/home/cwd/project/PoseEstimationForMobile/trained_v6/network_SHG_lv2_conv3_backbone3_batch-32_lr-0.001_gpus-1_128x256_experiments-network_SHG_lv2_conv2/model-112000 \
--output_node_names=l1_out/BiasAdd \
--output_graph=SHG_lv2_conv3_backbone3.pb \

python3 src/gen_frozen_pb.py --model=network_SHG_lv3_conv1_2 \
--checkpoint=/home/cwd/project/PoseEstimationForMobile/trained_v6/network_SHG_lv3_conv1_2_batch-32_lr-0.001_gpus-1_128x256_experiments-network_SHG_lv3_conv1_2/model-112000 \
--output_node_names=l2_out/BiasAdd \
--output_graph=SHG_lv3_conv1_2.pb 

python3 src/gen_frozen_pb.py --model=network_SHG_lv3_conv1 \
--checkpoint=/home/cwd/project/PoseEstimationForMobile/trained_v6/network_SHG_lv3_conv1_batch-32_lr-0.001_gpus-1_128x256_experiments-network_SHG_lv3_conv1/model-112000 \
--output_node_names=l2_out/BiasAdd \
--output_graph=SHG_lv3_conv1.pb 

python3 src/gen_frozen_pb.py --model=network_SHG_lv4_conv1 \
--checkpoint=/home/cwd/project/PoseEstimationForMobile/trained_v6/network_SHG_lv4_conv1_batch-32_lr-0.001_gpus-1_128x256_experiments-network_SHG_lv4_conv1/model-112000 \
--output_node_names=l3_out/BiasAdd \
--output_graph=SHG_lv4_conv1.pb 
