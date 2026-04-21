# 🧾 BILLSENTINEL AI: Medical Billing Error Detection System

BILLSENTINEL AI is an end-to-end medical auditing tool designed to detect billing errors, fraudulent charges, and hidden fees using a hybrid approach of **Rule-based Logic** and **Machine Learning**.

## 🎯 Features

- **Synthetic Dataset Generation**: Creates realistic medical billing data with balanced noise and error patterns (ICU, Surgery, X-ray, Consultation, Medicine).
- **Hybrid Detection**:
  - **Rule Engine**: Deterministic checks for duplicate charges, post-discharge billing, and overpricing based on category averages.
  - **ML Models**: statistical anomaly detection using `IsolationForest` and supervised classification using `RandomForestClassifier`.
- **Intelligent OCR Parser**: Extracts structured data from PDF, JPG, PNG, and Text files using `pytesseract`.
- **Risk Scoring**: Calculates a "Bill Health Score" and identifies "High Risk" bills.
- **Cost Savings Estimation**: Quantifies potential savings based on detected errors.
- **Streamlit UI**: Premium, dark-mode dashboard for real-time analysis.

## 🏗️ Technology Stack

- **Core**: Python
- **Machine Learning**: `scikit-learn`, `numpy`
- **Frontend**: `Streamlit`
- **OCR/Document Processing**: `pytesseract`, `PIL`, `pdf2image`
- **Aesthetics**: Glassmorphism and dark-mode CSS

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (must be installed on your system)
- Poppler (for PDF support via `pdf2image`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TejasTechluxverse7/billsentinel-ai.git
   cd billsentinel-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have pytesseract and pdf2image requirements met on your OS)*

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Project Structure

- `app.py`: Streamlit frontend and integration logic.
- `billsentinel_ai.py`: Core ML pipeline and data generation engine.
- `.gitignore`: Configured to exclude large datasets and system artifacts.

## 📊 Error Types Detected

- **duplicate_charge**: Identifying services billed multiple times on the same date.
- **post_discharge_charge**: Catching charges injected after the official discharge date.
- **mismatch_treatment**: Detecting code-description inconsistencies.
- **hidden_fee**: Catching split charges that inflate the total.
- **overpricing**: Statistical identification of costs > 2x the category average.

## 🛡️ License

MIT License.

---
*Built with ❤️ for AI-driven Medical Auditing.*
