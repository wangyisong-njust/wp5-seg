python eval.py \
  --ckpt runs/wp5_baseline_20260119_211557/best.ckpt \
  --data_dir /mnt/sdb/Data/3DDL-WP5-Data/3ddl-dataset/data \
  --output_dir runs/wp5_baseline_20260119_211557/eval \
  --save_preds \
  --heavy \
  --hd_percentile 95 \
  --log_to_file
