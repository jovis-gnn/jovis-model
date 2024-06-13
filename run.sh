# fine tuning llm
python run.py \
--mode train --pkg llm --task chat --use_hf_model true \
--data_dir /home/omnious/workspace/jovis/jovis-model/jovis_model/_db/llm/fine-tuning/alpaca \
--train_file_name alpaca_train.csv \
--output_dir /home/omnious/workspace/jovis/jovis-model/jovis_model/outputs \
--hf_name meta-llama/Meta-Llama-3-8B-Instruct \
--use_peft true \
--quantization true \
--use_fp16 false \
--train_batch_size 1