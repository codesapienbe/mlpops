# 🚀 Belgian Job Analysis MLOps Portfolio

Welcome to your interactive **MLOps playground**! Upload job data and instantly explore insights into the Belgian job market.

🎯 **Key Features**

- 📥 **Upload & Analyze** your `jobs.csv`
- 🔄 **Monitor** pipeline stages: Ingest → Clean → Train → Predict
- 📖 **Read** quick documentation right here

---

## Prerequisites

- Python 3.11+
- uv package manager
- Terraform (for infrastructure deployment)
- AWS CLI (for cloud deployment)
- AWS Account (for cloud deployment)

---

## 🏁 Quick Start (End-Users)

```bash
# Setup the environment
cd mlops-exam
# Install dependencies using uv
uv sync

# Run the full pipeline
uv run train

# Start the FastAPI server
uv run api

# Launch the Streamlit dashboard
uv run web
```

Open your browser at <http://localhost:8501> to start exploring!

---

## 🔧 For Developers

```yaml
project: mlpops
version: 0.1.0
requirements:
  - fastapi
  - uvicorn
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - dvc
  - mlflow
```  

```console
📂  mlops-exam/
├─ src/
│  ├─ pipeline/ (ingest, clean, train, predict)
│  ├─ api/      (FastAPI endpoints)
│  └─ web/      (Streamlit dashboard)
├─ config.yml   (settings)
├─ Dockerfile   (container setup)
└─ README.md    (this guide)
```

---

Made with ❤️ by CodeSapiens 🚀
