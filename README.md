# RL Airfoil Optimization (TD3-first, XAI-ready)

Bu proje TD3 tabanlı airfoil optimizasyonunu modüler evaluator yapısıyla çalıştırır (`surrogate`, `xfoil`) ve XAI odaklı log üretir.

## 1) Sanal ortam kurulumu

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Surrogate dosyalarını yerleştirme

`checkpoints/` klasörüne koyun:

- `surrogate_s1d.pt` (veya s2d/s3d)
- `scalers.json`

> `scalers.json` içinde `x_mean`, `x_scale`, `y_mean`, `y_std`, `use_log_re` alanları kullanılmaktadır.

## 3) TD3 eğitim

```bash
python -m src.main train \
  --algorithm td3 \
  --evaluator surrogate \
  --surrogate-model-name S-1D \
  --surrogate-checkpoint-path checkpoints/surrogate_s1d.pt \
  --scaler-json-path checkpoints/scalers.json \
  --seed 42 \
  --total-timesteps 200000
```

## 4) TD3 deterministic evaluate + AoA sweep

```bash
python -m src.main evaluate \
  --algorithm td3 \
  --evaluator surrogate \
  --surrogate-model-name S-1D \
  --surrogate-checkpoint-path checkpoints/surrogate_s1d.pt \
  --scaler-json-path checkpoints/scalers.json \
  --rl-checkpoint-path checkpoints/td3_surrogate_s-1d.zip \
  --run-dir logs/td3/run_<id> \
  --episodes 10 \
  --aoa-sweep "-2,0,2,4,6,8"
```

## 5) XFOIL evaluator ile çalışma

Aynı komutlarda `--evaluator xfoil` seçerek çözücüyü değiştirin.

## 6) Üretilen XAI logları

Her run altında:

- `experiment_metadata.json`
- `rollout_step_logs.csv`
- `policy_outputs.csv`
- `episode_summary.csv`
- `training_update_logs.csv`
- `xfoil_validation_logs.csv`

Run dizinleri otomatik `logs/td3/run_<timestamp>` olarak oluşturulur.
