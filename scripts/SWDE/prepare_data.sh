clear

rm -rf ../../../nodes_info

python3 ../../src/SWDE/prepare_data.py \
--swde_path=../../../SWDE \
--pages_info_path=../../../pages_info.pkl \
--nodes_info_path=../../../nodes_info

# ALBERT: albert-base-v2
# BERT: bert-base-cased