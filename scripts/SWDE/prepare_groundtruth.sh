clear

rm -rf ../../../pages_info.pkl

python3 ../../src/SWDE/prepare_groundtruth.py \
--input_swde_path=../../../pack_data.pkl \
--ground_truth_path=../../../SWDE/groundtruth \
--pages_info_path=../../../pages_info.pkl
