# Setup Guide — UAP-FedECG

## Prerequisites

- Python 3.10+
- Git

## 1. Clone the repo

```
git clone https://github.com/WimukthiPWijerathne/Uncertainty-Aware-and-Differentially-Private-Federated-Learning-for-Non-IID-ECG-Classification.git
cd filename //change the folder name after cloning cuz the topic name is too long would cause issues in requirement installing
```

## 2. Create and activate a virtual environment

**Windows (PowerShell):**

```
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**

```
python3 -m venv venv
source venv/bin/activate
```

You'll know it worked when your prompt shows `(venv)` at the start.

## 3. Install dependencies

```
pip install -r requirements.txt
```

## 4. Confirm everything works

```
python models/cnn1d.py
```

Expected output:

```
Forward pass output shape: torch.Size([8, 5])
MC dropout mean_probs shape: torch.Size([8, 5])
MC dropout uncertainty shape: torch.Size([8])
```

If you see these three lines, your environment is set up correctly.

## Notes

- `venv/` and the dataset folder are gitignored — never commit them.
- **Windows users:** keep your local repo path short (e.g. `D:\uap-fedecg`, not buried inside long folder names like "Final Year\8th Semester\..."). A long path + venv can hit Windows' path length limit during `pip install` and fail with a confusing OSError.

## What's next

- `data/download_ptbxl.py` downloads and unzips the PTB-XL dataset locally (not tracked in git — everyone runs this themselves).
