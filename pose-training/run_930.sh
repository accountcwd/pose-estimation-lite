python3 ./src/train.py experiments/network_SHG_lv2_conv2_3.cfg
sleep 3m
python3 ./src/train.py experiments/network_SHG_lv2_conv2.cfg
sleep 3m
python3 ./src/train.py experiments/network_SHG_lv2_conv3_backbone3.cfg
sleep 3m
python3 ./src/train.py experiments/network_SHG_lv3_conv1_2.cfg
sleep 3m
python3 ./src/train.py experiments/network_SHG_lv3_conv1.cfg
sleep 3m
python3 ./src/train.py experiments/network_SHG_lv4_conv1.cfg
sleep 5m
poweroff
halt
shutdown -h 5
