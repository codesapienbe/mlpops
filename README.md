# 🚀 Belgian Job Analysis MLOps Portfolio

Welcome to your interactive **MLOps playground**! Upload job data and instantly explore insights into the Belgian job market.

🎯 **Key Features**

- 📥 **Upload & Analyze** your `jobs.csv`
- 🤖 **Scrape** Belgian job sites automatically
- 🔄 **Monitor** pipeline stages: Ingest → Clean → Train → Predict
- 📖 **Read** quick documentation right here

---

## Prerequisites

- Python 3.11+
- Terraform (for infrastructure deployment)
- AWS CLI (for cloud deployment)
- AWS Account (for cloud deployment)

---

## 🏁 Quick Start (End-Users)

```bash
# Setup the environment
cd mlops-exam
# On Linux/Mac
./setup.sh
# On Windows
setup.bat

# Activate the virtual environment
# On Linux/Mac
source .venv/bin/activate
# On Windows
.venv\Scripts\activate.bat

# Scrape job listings from Belgian job sites
python -m src.scrape

# Run the full pipeline
python src/main.py

# Start the FastAPI server
python -m src.api

# Launch the Streamlit dashboard
python -m src.web
```

Open your browser at <http://localhost:8501> to start exploring!

---

## 🤖 Job Scraping Feature

The application includes an automated scraper for collecting job listings from popular Belgian job sites:

- VDAB
- Jobat
- ICTJobs

To run the scraper:

```bash
python -m src.scrape
```

This will:
1. Connect to each job site
2. Extract job listings using AI
3. Save the data to the configured location in `config.yml`
4. Process the text according to the configured language settings

The scraper supports multiple languages and can be configured in the `config.yml` file.

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
  - crawl4ai
```  

```console
📂  mlops-exam/
├─ src/
│  ├─ pipeline/ (ingest, clean, train, predict)
│  ├─ api/      (FastAPI endpoints)
│  ├─ web/      (Streamlit dashboard)
│  └─ scrape/   (job scraping functionality)
├─ config.yml   (settings)
├─ Dockerfile   (container setup)
└─ README.md    (this guide)
```

🐛 **Troubleshooting**: If you run into errors with the scraper, ensure you've installed the browser dependencies:

```bash
python -m playwright install --with-deps chromium
```

---

Made with ❤️ by CodeSapiens 🚀
