# Backend — Data Pipeline, Training & Telemetry Services

This directory contains the backend services and ML pipeline for the **TinyML-Based Predictive Maintenance for BLDC Motors** project.

## Planned Contents

- **Data Ingestion & Logging:** MQTT broker / WebSocket bridge and session-logging utilities that align ESP32 sensor streams with timestamped fault-injection ground truth.
  - **Dataset CSV Schema:** `[timestamp, ax, ay, az, current_a, current_b, current_c, temp, label]`
- **Feature Extraction:** Sliding-window segmentation ($512\text{–}1024$ samples, $50\%$ overlap) and extraction of time-domain (RMS, kurtosis, crest factor, peak-to-peak) and frequency-domain (FFT, spectral kurtosis, BPFO/BPFI) features.
- **Model Training & Comparison:** Training scripts for the **1D-CNN**, **LSTM**, and **Autoencoder** models with episode-level train/validation/test splitting ($70/15/15$).
- **TinyML Compression:** Post-training 8-bit quantization, weight pruning, and `.tflite` / C-array export targeting **< 50 KB** flash and **< 100 ms** latency on ESP32.
