<div align="center">

# 🌩️ StormSense

### AI-Powered Severe Weather Early Warning & Nowcasting System

<p>
  <strong>Predict • Detect • Visualize • Warn</strong>
</p>

<p>
  <img src="https://img.shields.io/badge/AI-Severe%20Weather-blueviolet?style=for-the-badge" alt="AI">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Deep%20Learning-ConvGRU-orange?style=for-the-badge" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Weather-Nowcasting-00AEEF?style=for-the-badge" alt="Nowcasting">
  <img src="https://img.shields.io/badge/Region-West%20Bengal%2C%20India-138808?style=for-the-badge" alt="West Bengal">
</p>

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-technology-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a>
</p>

</div>

---

<div align="center">

## 🌪️ Turning Weather Data Into Early Warnings

</div>

**StormSense** is an AI-powered severe-weather early warning and nowcasting platform designed for **West Bengal, India**.

It combines meteorological data, deep learning, real-time weather observations, interactive geospatial visualization, and historical event analysis to provide short-term severe-weather intelligence.

Instead of simply displaying weather information, StormSense is designed around one core question:

> ### **"What severe weather could develop in the next few hours?"**

---

# 🚨 Why StormSense?

Severe weather can change rapidly.

Heavy rainfall, convective storms, strong winds, and extreme precipitation can create significant risks for:

- 🚗 Transportation
- 🏙️ Urban infrastructure
- 🌧️ Flood-prone regions
- ⚡ Power infrastructure
- 🏠 Communities
- 🚑 Emergency response
- 🌾 Agriculture

StormSense provides a visual decision-support interface that combines **current weather conditions** with **short-term AI-based risk predictions**.

---

# ⚡ Features

<div align="center">

<table>
<tr>

<td align="center" width="33%">

### 🧠
### AI Nowcasting

Short-term severe-weather prediction using a trained deep-learning model.

</td>

<td align="center" width="33%">

### 🌧️
### Rainfall Intelligence

Visualizes precipitation-related risk across the target region.

</td>

<td align="center" width="33%">

### 🗺️
### Interactive Maps

Geospatial risk visualization across West Bengal.

</td>

</tr>

<tr>

<td align="center">

### 📡
### Live Weather

Uses current external weather observations for live conditions.

</td>

<td align="center">

### ⏱️
### +2h / +4h / +6h

Short-term forecast horizons for severe-weather risk.

</td>

<td align="center">

### 🛰️
### Historical Mode

Analyze historical severe-weather events such as Cyclone Remal.

</td>

</tr>
</table>

</div>

---

# 🎯 Prediction Horizons

StormSense provides multiple short-term prediction horizons:

| Horizon | Purpose |
|---|---|
| 🟢 **+2 Hours** | Immediate severe-weather risk |
| 🟡 **+4 Hours** | Short-term developing risk |
| 🔴 **+6 Hours** | Extended short-term risk |

The interface allows users to switch between horizons and inspect the corresponding spatial risk information.

---

# 🛰️ LIVE MODE

<div align="center">

### 📡 Real-Time Weather Intelligence

</div>

LIVE mode is designed to represent the **current state of the atmosphere** separately from future predictions.

The system combines:


Current Weather Observations
            +
NOAA GFS Atmospheric Data
            +
StormSense AI Model
            ↓
     Short-Term Risk
            ↓
       Map + Alerts
LIVE Mode provides
📍 Current conditions
🌧️ Current precipitation information
💨 Wind information
🌡️ Temperature
💧 Humidity
🗺️ Current spatial risk
⏱️ +2h / +4h / +6h predictions
🕰️ HISTORICAL MODE
<div align="center">
🌀 Reconstructing Severe Weather Events
</div>

Historical Mode allows StormSense to analyze previously recorded weather events without mixing them with current live observations.

A supported historical scenario is:

🌀 Cyclone Remal — May 2024

Historical mode uses archived atmospheric and satellite-related datasets to reproduce the event timeline and visualize predicted severe-weather conditions.

Historical Atmospheric Data
            ↓
      Model Inference
            ↓
     +2h / +4h / +6h
            ↓
    Historical Risk Map
            ↓
      Event Analysis
Historical Mode is intentionally isolated from LIVE mode.

This prevents:

❌ Current timestamps leaking into historical analysis
❌ Current radar being presented as historical radar
❌ Live observations being mixed with historical conditions
❌ Forecast information being mislabeled as observations
🧠 How It Works

StormSense uses a multi-stage inference pipeline.

                   ┌──────────────────────┐
                   │  NOAA GFS Analysis   │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Data Harmonization   │
                   │ & Feature Preparation│
                   └──────────┬───────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      StormSense Model        │
              │                               │
              │  Tri-Stream ConvGRU Network  │
              │            +                  │
              │   Autoregressive Decoder      │
              └───────────────┬───────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Severe Weather Risk  │
                   │      Prediction      │
                   └──────────┬───────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       Risk Processing        │
              └───────────────┬───────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Interactive Web Map  │
                   │      & Dashboard     │
                   └──────────────────────┘
🔬 AI Model

StormSense uses a trained:

SevereWeatherNetV2

The model uses a tri-stream ConvGRU architecture with an autoregressive lead decoder.

Input

The model processes a sequence of atmospheric states:

6 consecutive hourly analysis states
                ↓
       ┌────────┴────────┐
       │                 │
 Surface Variables   Pressure Levels
       │                 │
       └────────┬────────┘
                │
               DEM
                ↓
          Neural Network
                ↓
       Severe Weather Risk
Model Inputs
Surface Variables
u10
v10
d2m
t2m
sp
cape
cin
tcwv
tp
Pressure-Level Variables
u
v
z
q
t

across multiple atmospheric pressure levels.

Terrain

Digital Elevation Model (DEM) information is also incorporated into the model input.

📊 Model Output

StormSense produces spatial predictions for severe-weather risk at:

+2 hours
+4 hours
+6 hours

The predictions are then transformed into spatial risk information for visualization on the map.

🗺️ Interactive Weather Intelligence

The StormSense interface is built around an interactive geospatial dashboard.

Users can inspect:

🌧️ Rainfall risk
⛈️ Severe-weather risk
📍 Location-specific conditions
🗺️ District-level information
📊 Forecast statistics
⏱️ Different prediction horizons
🧠 Model context
🛰️ Weather data feeds

The goal is to make complex atmospheric information easier to understand at a glance.

🛰️ Weather Data Sources

StormSense separates observations, forecast inputs, and historical datasets.

Data	Role
🌐 NOAA GFS	Atmospheric forecast/analysis input
🌦️ OpenWeather	Current weather observations
📡 Radar Feed	Current radar visualization
🛰️ INSAT Archives	Historical satellite visualization
🗺️ Boundary Data	West Bengal spatial visualization
⛰️ DEM	Terrain information

Important: External operational weather data is kept distinct from StormSense AI predictions. The system does not present third-party forecasts as StormSense model output.

🏗️ Project Architecture
StormSense/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── static/
│
├── backend/
│   ├── main.py
│   ├── hazard_engine.py
│   ├── openweather.py
│   ├── config.py
│   └── __init__.py
│
├── src/
│   ├── data/
│   ├── inference/
│   ├── models/
│   └── ...
│
├── Data/
│   ├── BOUNDARIES/
│   └── outputs/
│       └── checkpoints/
│
├── processed/
│   └── cache/
│       └── era5_memmap/
│
├── configs/
│   └── default.yaml
│
├── scripts/
│   └── download_runtime_data.py
│
├── tests/
│
├── reports/
│
├── run_server.py
├── requirements.txt
├── .env.example
└── README.md
🛠️ Technology Stack
<div align="center"> <table> <tr> <td align="center"><b>🐍 Python</b></td> <td align="center"><b>🧠 PyTorch</b></td> <td align="center"><b>🌐 HTML / CSS / JS</b></td> </tr> <tr> <td align="center"><b>🗺️ Leaflet</b></td> <td align="center"><b>🌦️ Weather APIs</b></td> <td align="center"><b>📡 NOAA GFS</b></td> </tr> </table> </div>
Core Technologies
Python 3.13
PyTorch
ConvGRU
NumPy
FastAPI / Python backend
HTML5
CSS3
JavaScript
Leaflet
NOAA GFS
OpenWeather
INSAT / MOSDAC historical data
📁 Runtime Data

StormSense intentionally separates runtime data from large training datasets.

LIVE Runtime

The LIVE system requires:

StormSense source code
        +
AI model checkpoint
        +
Normalization statistics
        +
West Bengal boundaries
        +
DEM
        +
Internet connection
        +
OpenWeather API key
HISTORICAL Runtime

Historical analysis additionally requires the processed historical atmospheric dataset.

Large raw training datasets are not required to run the application.

🚀 Installation
1️⃣ Clone the Repository
git clone https://github.com/Souvik686/StormSense.git
cd StormSense
2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
🔐 4️⃣ Configure Environment

Create your environment file:

copy .env.example .env

or on Linux/macOS:

cp .env.example .env

Add your OpenWeather API key:

OPENWEATHER_API_KEY=your_api_key_here
📦 5️⃣ Download Runtime Data

For a public repository, download the required runtime assets:

python scripts/download_runtime_data.py

To download only LIVE requirements:

python scripts/download_runtime_data.py --live-only

To verify downloaded files:

python scripts/download_runtime_data.py --verify

Runtime downloads are verified using SHA-256 checksums.

▶️ 6️⃣ Start StormSense

Run:

python run_server.py

Then open:

http://127.0.0.1:8000
🎮 Using StormSense
LIVE
Select → LIVE
        ↓
Choose +2h / +4h / +6h
        ↓
Inspect risk map
        ↓
Inspect weather conditions
        ↓
Analyze affected regions
HISTORICAL
Select → HISTORICAL
        ↓
Choose event
        ↓
Choose prediction horizon
        ↓
Explore historical risk
        ↓
Analyze event conditions
📈 Prediction Dashboard

StormSense is designed to present multiple layers of information rather than a single prediction number.

Risk Layer

Displays the spatial distribution of predicted severe-weather risk.

Weather Layer

Provides environmental context such as:

Temperature
Humidity
Wind
Rainfall
Atmospheric conditions
Geographic Layer

Provides:

West Bengal boundary
District boundaries
Location information
Spatial risk distribution
🧪 Model Evaluation

StormSense has been evaluated across historical cases using multiple forecast horizons.

Horizon	CSI	POD	PR-AUC
+2h	0.1045	0.1900	0.0996
+4h	0.0832	0.1455	0.1165
+6h	0.0867	0.1816	0.0981

These metrics represent the current model evaluation and are provided transparently rather than presenting the system as perfectly accurate.

⚠️ Important Design Principles

StormSense follows several important data-integrity principles.

1. Observation ≠ Forecast

Current observations are not presented as AI predictions.

2. External Forecast ≠ StormSense AI

Third-party operational forecasts are clearly separated from StormSense model output.

3. Historical ≠ LIVE

Historical event analysis does not use current weather observations.

4. No Fabricated Weather Data

Missing observations are not silently replaced with invented values.

5. Timestamps Matter

Atmospheric analysis time, observation time, forecast target time, and wall-clock time are kept conceptually separate.

🔒 Data & Repository Modes

StormSense supports two repository deployment approaches.

🔐 Private Repository

The private deployment can contain the runtime model and historical runtime assets directly.

Private Repository
       │
       ├── Source Code
       ├── Model
       ├── Historical Dataset
       ├── Runtime Assets
       └── Environment Configuration
🌍 Public Repository

The public repository can remain lightweight while runtime assets are downloaded separately.

Public GitHub
     │
     ├── Source Code
     ├── Frontend
     ├── Backend
     ├── Model Downloader
     └── Runtime Manifest
              │
              ▼
       Runtime Data Host

This avoids committing multi-gigabyte datasets directly into the Git repository.

💾 Git LFS

Large runtime assets such as model/data files can use Git Large File Storage (Git LFS) where appropriate.

Example:

*.npy filter=lfs diff=lfs merge=lfs -text
Data/outputs/checkpoints/v2_calibrated_best.pt filter=lfs diff=lfs merge=lfs -text
🧪 Testing

The project includes tests covering important application behavior.

pytest

Historical behavior can also be tested independently to ensure that changes to the application do not unintentionally alter the historical pipeline.

🔮 Future Improvements

StormSense can be extended with:

🌧️ Higher-resolution precipitation nowcasting
🛰️ More satellite products
⚡ Lightning integration
🌊 Flood-risk estimation
🏙️ Hyperlocal urban risk
📱 Mobile alerts
🔔 Automated emergency notifications
🤖 Improved model calibration
📊 Expanded historical validation
🧠 Explainable AI improvements
🌍 Expansion beyond West Bengal
🏆 Hackathon Vision
<div align="center">
From Weather Data → To Actionable Intelligence
DATA
 ↓
AI
 ↓
PREDICTION
 ↓
VISUALIZATION
 ↓
EARLY WARNING
 ↓
ACTION
🌩️ StormSense

Building a smarter way to understand severe weather.

</div>
👨‍💻 Project
<div align="center">
StormSense

AI-Powered Severe Weather Early Warning & Nowcasting

Developed for a weather/AI hackathon with a focus on:

Artificial Intelligence • Meteorology • Geospatial Intelligence • Early Warning Systems

<br>

⭐ If you find this project interesting, consider giving the repository a star!

</div>
<div align="center">
🌩️ StormSense

<sub>Predict the risk. Understand the storm. Act earlier.</sub>

<br><br>

© 2026 StormSense

</div> 
