python run.py \
--mode train \
--pkg llm \
--task chat \
--use_hf_model true \
--data_dir jovis_model/_db/llm/fine-tuning/alpaca \
--train_file_name alpaca_train.csv \
--output_dir outputs \
--enable_fsdp true \
--use_fp16 true