clear

python3 ../../src/SWDE/in_vertical_train.py \
--swde_path=../../../SWDE \
--pack_path=../../../pack_data.pkl \
--nodes_info_path=../../../nodes_info \
--verticals="auto" \
--num_seeds=1 \
--shuffle=True \
--drop_last=True \
--batch_size=8 \
--num_epochs=10 \
--optimizer="Adam" \
--lr=0.001
