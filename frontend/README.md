# Frontend — Multilingual Machine Health Dashboard

This directory contains the front-end web dashboards for visualizing BLDC motor health in real time.

## Planned Contents

Both dashboard versions are designed as **single-file, zero-dependency HTML/CSS/JS** applications using inline **SVG** graphics so they load fast, work offline, and run smoothly on low-end smartphones:

- **Version 1 (`simple-alert`):** Minimalist single-status-card alert view for lowest-end devices (supports BLE, USB Serial, Wi-Fi, and Demo mode).
  - **Protocol:** `H` (healthy), `W,<fault_id>,<confidence>` (warning), `F,<fault_id>,<confidence>` (fault).
- **Version 2 (`full-dashboard`):** Complete diagnostic view featuring a half-circle **Machine Health Gauge**, **Live Sensor Cards** (vibration, current, sound), **Real-Time SVG Trend Graphs**, **AI Degradation Indicator**, **Failure Timeline** (fault start $\rightarrow$ AI warning $\rightarrow$ failure), and **Alert History**.
  - **Protocol:** `D,<vibration>,<current>,<sound>,<degradation_score>`

## Key Design Features

- **5 Languages:** English, Hindi (`हिन्दी`), Marathi (`मराठी`), Gujarati (`ગુજરાતી`), and Tamil (`தமிழ்`) with automatic browser locale detection and persistent selection.
- **Accessibility:** Color + symbol encoding (`✔`, `⚠`, `✖`) for color-blind users and bright-sunlight readability, large touch targets ($\ge 44\text{ px}$ / $58\text{ px}$), and automatic light/dark mode support.
- **Connectivity:** Web Bluetooth (Nordic UART BLE), Web Serial ($9600\text{ baud}$ USB OTG), and built-in simulated failure **Demo** mode.
