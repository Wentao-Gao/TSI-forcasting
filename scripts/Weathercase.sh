for seed in $(seq 0 4); do
  # multivar
  python -u train.py weather_case forecast_multivar --epochs 10 --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
  # univar
  python -u train.py weather_case forecast_univar --epochs 10 --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed ${seed} --eval
done