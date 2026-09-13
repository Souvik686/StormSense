# AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting

## Smart India Hackathon 2024 | West Bengal Region

---

## Quick Start

### 1. Create GitHub account
Open Chrome and go to:

[GitHub Sign Up](https://github.com/signup?utm_source=chatgpt.com)

Create the account and verify your email address.

### 2. Accept the StormSense invitation

Check your email for the GitHub repository invitation from Souvik.

Or open:

[GitHub Invitations](https://github.com/notifications?utm_source=chatgpt.com)

Accept the invitation to the StormSense repository.

### 3. Install Python 3.13.7

Open:

[Python 3.13.7 for Windows](https://www.python.org/downloads/release/python-3137/?utm_source=chatgpt.com)

Download Windows installer (64-bit).

Run the installer.

Important: On the first installation screen, tick:

```
☑ Add python.exe to PATH
```

Then click Install Now.

### 4. Check Python

Open Command Prompt and run:

```bash
py -3.13 --version
```

It must show:
```
Python 3.13.7
```

### 5. Install Git

Open:

[Git for Windows](https://git-scm.com/download/win?utm_source=chatgpt.com)

Download and install Git.

After installation, open a new Command Prompt and run:

```bash
git --version
```

### 6. Install Git LFS

Open:

[Git LFS](https://git-lfs.com/?utm_source=chatgpt.com)

Download and install Git LFS.

Then open Command Prompt and run:

```bash
git lfs install
```

You should see:

```
Git LFS initialized.
```

### 7. Clone StormSense

In Command Prompt, go to the location where you want the project.

For example:

```bash
cd Desktop
```

Then:

```bash
git clone https://github.com/Souvik686/StormSense.git
```

Enter the StormSense folder:

```bash
cd StormSense
```

### 8. Download the large LFS files

Run:

```bash
git lfs pull
```

Wait until it finishes completely.

### 9. Create the Python 3.13.7 virtual environment

Run:

```bash
py -3.13 -m venv venv
```

### 10. Activate the environment

```bash
venv\Scripts\activate
```

You should see (venv) at the beginning of the Command Prompt.

### 11. Verify Python version

```bash
python --version
```

It must show:

```
Python 3.13.7
```

### 12. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 13. Install StormSense dependencies

```bash
python -m pip install -r requirements.txt
```

Wait until installation finishes.

### 14. Start StormSense

Make sure (venv) is still visible in Command Prompt, then run:

```bash
python run_server.py
```

### 15. Open StormSense

The terminal will show the local server address.

Open that address in Chrome.

### Every time they want to run StormSense again

Open Command Prompt:

```bash
cd Desktop\StormSense
venv\Scripts\activate
python run_server.py
```