# Firmware — ESP32 Acquisition & On-Device TinyML Inference

This directory contains the embedded firmware for the **ESP32-WROOM-32** acquisition and inference node.

## Planned Contents

- **DMA Sensor Acquisition:** High-rate ($\ge 1000\text{ Hz}$) non-blocking SPI/ADC sampling for the **ADXL345** 3-axis accelerometer and **3× ACS712** Hall-effect phase current sensors, plus separate $\text{I}^2\text{C}$/1-Wire polling for **DS18B20** temperature and **INA219** power benchmarking.
- **Timestamp Synchronization:** NTP Unix-timestamp sync at session start combined with a monotonic hardware counter for sub-millisecond sample alignment.
- **Fault-Injection Relay Control:** Firmware trigger for controlled single-phase open-circuit fault injection with exact timestamp logging.
- **On-Device Inference:** TensorFlow Lite for Microcontrollers (TFLM) runtime executing the quantized `.tflite` model and streaming verdicts over **BLE (Nordic UART)**, **USB Serial**, and **Wi-Fi/MQTT**.
