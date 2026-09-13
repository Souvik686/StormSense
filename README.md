<div align="center">

# 🌩️ StormSense

### AI-Powered Severe Weather Early Warning & Nowcasting System

<br>

<img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/NOAA-GFS-00AEEF?style=for-the-badge">
<img src="https://img.shields.io/badge/Weather-Nowcasting-00AEEF?style=for-the-badge">
<img src="https://img.shields.io/badge/West%20Bengal-India-138808?style=for-the-badge">

<br><br>

<b>Predict • Detect • Visualize • Warn</b>

</div>

---

<div align="center">

# 🚀 Get StormSense Running

### Download the required files, configure the environment, and launch the complete system.

</div>

<br>

<div align="center">

<table>
<tr>
<td align="center" width="150">

### 01
📥

<b>Clone</b>

</td>

<td align="center" width="150">

### 02
🐍

<b>Python</b>

</td>

<td align="center" width="150">

### 03
📦

<b>Install</b>

</td>

<td align="center" width="150">

### 04
💾

<b>Download</b>

</td>

<td align="center" width="150">

### 05
🔐

<b>Configure</b>

</td>

<td align="center" width="150">

### 06
⚡

<b>Launch</b>

</td>
</tr>
</table>

</div>

---

## 📥 01 — Clone StormSense

Open a terminal and clone the repository:

git clone https://github.com/Souvik686/StormSense.git
cd StormSense
<div align="center">

Your StormSense workspace is ready.

</div>
🐍 02 — Set Up Python

StormSense requires Python 3.13.

<details> <summary><b>🪟 Windows</b></summary> <br>

Create a virtual environment:

python -m venv venv

Activate it:

venv\Scripts\activate

Verify Python:

python --version
</details> <details> <summary><b>🐧 Linux / 🍎 macOS</b></summary> <br>

Create a virtual environment:

python3 -m venv venv

Activate it:

source venv/bin/activate

Verify Python:

python3 --version
</details>
📦 03 — Install Dependencies

Make sure your virtual environment is active.

Then install all required Python packages:

pip install -r requirements.txt

Wait until the installation completes successfully.

💾 04 — Download Runtime Data

StormSense requires several runtime assets for the complete LIVE + HISTORICAL system.

For the public repository, download the required runtime files using:

python scripts/download_runtime_data.py

The downloader prepares the required:

<div align="center"> <table> <tr> <td align="center">

🧠<br>
<b>AI Model</b>

</td> <td align="center">

📊<br>
<b>Normalization Data</b>

</td> <td align="center">

🗺️<br>
<b>Map Boundaries</b>

</td> <td align="center">

⛰️<br>
<b>Terrain Data</b>

</td> <td align="center">

🕐<br>
<b>Historical Data</b>

</td> </tr> </table> </div>
⚡ LIVE Only

If you only need the LIVE system:

python scripts/download_runtime_data.py --live-only
🔍 Verify Downloaded Files

After downloading, verify the runtime files:

python scripts/download_runtime_data.py --verify

The downloader verifies the files using their expected SHA-256 checksums.

⚠️ Important: Do not manually rename, move, or rearrange downloaded runtime files. StormSense expects them at specific paths.

🔐 05 — Configure Environment

StormSense uses an environment file for external API configuration.

🪟 Windows
copy .env.example .env
🐧 Linux / 🍎 macOS
cp .env.example .env

Open the newly created .env file and add your OpenWeather API key:

OPENWEATHER_API_KEY=your_api_key_here
<div align="center">
🔑 OpenWeather API Key

Required for the LIVE weather observation layer.

</div>
⚡ 06 — Launch StormSense

Make sure you are inside the StormSense project root and your virtual environment is active.

Start the application:

python run_server.py

When the server starts successfully, open:

<div align="center">
🌐 http://127.0.0.1:8000
</div>
🌩️ LIVE Mode
<div align="center">
📡 Real-Time Weather Intelligence
</div>

Once StormSense opens:

Select LIVE mode.
Choose a forecast horizon.
Explore the current weather conditions.
Inspect the AI-generated severe-weather risk.
Explore the interactive map and regional information.
Available Horizons
<div align="center">

+2h     +4h     +6h

</div>
🌀 HISTORICAL Mode
<div align="center">
Reconstruct & Analyze Historical Severe Weather
</div>

To use Historical Mode:

Open StormSense.
Switch from LIVE to HISTORICAL.
Select the available historical event.
Select the desired forecast horizon.
Explore the historical risk visualization.

🕰️ Historical Mode uses the downloaded historical runtime dataset and is kept separate from LIVE weather observations.

🧪 Verify the Complete Installation

If you want to verify that the runtime assets are present:

python scripts/download_runtime_data.py --verify

Then launch the application:

python run_server.py

Open:

http://127.0.0.1:8000

If the StormSense dashboard loads successfully, the installation is ready.

🛠️ Quick Troubleshooting
<details> <summary><b>❌ Python is not recognized</b></summary>

Check your Python installation:

python --version

StormSense requires Python 3.13.

If the command is unavailable, install Python and make sure it is added to your system PATH.

</details> <details> <summary><b>❌ ModuleNotFoundError</b></summary>

Make sure your virtual environment is activated:

venv\Scripts\activate

Then reinstall the dependencies:

pip install -r requirements.txt
</details> <details> <summary><b>❌ Runtime files are missing</b></summary>

Run the runtime downloader:

python scripts/download_runtime_data.py

Then verify:

python scripts/download_runtime_data.py --verify
</details> <details> <summary><b>❌ Live weather is unavailable</b></summary>

Check that your .env file contains:

OPENWEATHER_API_KEY=your_api_key_here

Make sure the API key is valid and the machine has an internet connection.

</details> <details> <summary><b>❌ The application does not start</b></summary>

Make sure you are running the command from the StormSense root directory:

python run_server.py

Also confirm that your virtual environment is active and dependencies are installed.

</details>
<div align="center">
✅ You're Ready!
<br> <table> <tr> <td align="center" width="250">

🌩️

LIVE

Current weather
+
AI risk prediction

</td> <td align="center" width="250">

🌀

HISTORICAL

Historical event
+
AI risk analysis

</td> </tr> </table> <br>
🚀

<b>Clone → Install → Download → Configure → Launch</b>

<br><br>

<i>Predict the risk. Understand the storm. Act earlier.</i>

</div> 