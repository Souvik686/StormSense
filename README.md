
<div align="center">

| Step | What to do |
|:---:|---|
| 1️⃣ | Clone the repository |
| 2️⃣ | Create virtual environment |
| 3️⃣ | Install dependencies |
| 4️⃣ | Download runtime data |
| 5️⃣ | Configure API key |
| 6️⃣ | Start StormSense |

</div>

---

## 📥 1. Clone the Repository

```bash
git clone https://github.com/Souvik686/StormSense.git
cd StormSense
🐍 2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
📦 3. Install Dependencies
pip install -r requirements.txt
💾 4. Download Required Runtime Data

For the PUBLIC repository, run:

python scripts/download_runtime_data.py

This downloads the required files for:

🟢 LIVE Mode
🕐 HISTORICAL Mode
🧠 StormSense AI Model
🗺️ Geographic boundaries
📊 Historical inference data
LIVE Only

If you only want to run LIVE mode:

python scripts/download_runtime_data.py --live-only
Verify Downloads
python scripts/download_runtime_data.py --verify

⚠️ Do not manually rename or move downloaded files.
StormSense expects them at their original paths.

🔐 5. Configure Environment

Create your .env file from the provided template.

Windows
copy .env.example .env
Linux / macOS
cp .env.example .env

Open .env and add:

OPENWEATHER_API_KEY=your_api_key_here
▶️ 6. Start StormSense

From the project root:

python run_server.py

When the server starts, open:

http://127.0.0.1:8000
🌩️ 7. Run LIVE Mode
Open StormSense in your browser.
Select LIVE mode.
Select the desired forecast horizon:
+2h
+4h
+6h
Explore the live weather and AI risk information.
🌀 8. Run HISTORICAL Mode
Open StormSense.
Switch to HISTORICAL mode.
Select the available historical event.
Select the forecast horizon.
Explore the reconstructed historical severe-weather conditions.

🕰️ Historical Mode uses the downloaded historical runtime dataset and is kept separate from LIVE weather data.

🧪 9. Verify the Installation

From the project root:

python scripts/download_runtime_data.py --verify

Then start the application:

python run_server.py

If the application opens successfully at:

http://127.0.0.1:8000

your StormSense installation is ready.

<div align="center">
✅ You're Ready!
🌩️ StormSense

LIVE Weather Intelligence + Historical Severe Weather Analysis

<br>

Clone → Install → Download → Configure → Run

</div> ```