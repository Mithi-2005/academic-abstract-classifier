# Academic Abstract Classifier

This project is a machine learning application designed to classify academic paper abstracts into one of 11 subject categories. It utilizes a fine-tuned **DeBERTa-v3-small** model.

## features

- **Text Classification**: Classifies abstracts into categories like "Computer Vision", "Artificial Intelligence", "Commutative Algebra", etc.
- **Microservice Architecture**:
  - **Backend**: Built with [FastAPI](https://fastapi.tiangolo.com/), handling model inference.
  - **Frontend**: Built with [Streamlit](https://streamlit.io/), providing an interactive user interface.
- **State-of-the-Art Model**: Uses a LoRA fine-tuned DeBERTa model for efficient and accurate predictions.

## Project Structure

```
├── backend/
│   └── app.py            # FastAPI backend server
├── frontend/
│   └── app.py            # Streamlit frontend application
├── final_deberta_model/  # Directory containing the fine-tuned model adapter
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Transformer-Based-Academic-Classifier-main
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the full application, you need to start both the backend and the frontend servers.

### 1. Start the Backend Server

The backend runs on port `8000`.

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
*Alternatively, you can run the script directly:*
```bash
python backend/app.py
```

### 2. Start the Frontend Application

Open a new terminal window and run the Streamlit app.

```bash
streamlit run frontend/app.py
```

The application should open automatically in your browser (usually at `http://localhost:8501`).

## Model Details

The model is based on `microsoft/deberta-v3-small` and has been fine-tuned using PEFT (Parameter-Efficient Fine-Tuning) with LoRA. It maps input text to one of the following 11 labels:

1. Commutative Algebra
2. Computer Vision and Pattern Recognition
3. Artificial Intelligence
4. Systems and Control
5. Group Theory
6. Computational Engineering, Finance, and Science
7. Programming Languages
8. Information Theory
9. Data Structures and Algorithms
10. Neural and Evolutionary Computing
11. Statistics Theory
