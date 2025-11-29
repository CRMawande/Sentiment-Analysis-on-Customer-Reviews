# Sentiment Analysis on Customer Reviews  
**Task 1.2 – Development Phase**  
**IU DLBDSEAI502**  

This repository contains the **complete source code and documentation** for Task 1.2: Sentiment analysis of customer reviews.

The task required building a production-ready 3-class sentiment classifier (negative / neutral / positive) on real-world Amazon beauty product reviews by Hou et al. (2024), using modern NLP techniques.

### Final Model Performance (Validation Set – 69,205 reviews)

| Metric              | Value     |
|---------------------|-----------|
| Accuracy            | **85.48 %** |
| Macro F1            | **0.740** |
| Weighted F1         | **0.874** |
| Eval Loss           | 0.492     |

#### Detailed Classification Report

| Class     | Precision | Recall | F1-score | Support  |
|-----------|-----------|--------|----------|----------|
| negative  | 0.87      | 0.78   | 0.82     | 14,323   |
| neutral   | 0.34      | 0.71   | 0.46     | 5,565    |
| positive  | 0.98      | 0.89   | 0.94     | 49,317   |
| **accuracy** |           |        | **0.85** | 69,205   |
| **macro avg** | 0.73     | 0.79   | **0.74** | 69,205   |
| **weighted avg** | 0.91  | 0.85   | **0.87** | 69,205   |

![Confusion Matrix](reports/figures/confusion_matrix_final.png)

### Project Highlights
- DistilBERT (Sanh et al., 2019) fine-tuned with SST-2 weight transfer
- Custom WeightedTrainer + class-weighted loss
- Training: 3 epochs, effective batch size 512, ~6 hours on Kaggle T4
- Streamlit demo app 
 ---

### User Guide
### **1\. Prerequisites**

To run the Streamlit application, you must have **Python 3.9+** installed and access to a command-line environment (Terminal or Command Prompt).

### **2\. Installation and Setup**

##### **Step 2.1: Clone the Repository**

```bash
git clone https://github.com/CRMawande/Sentiment-Analysis-on-Customer-Reviews.git
cd Sentiment-Analysis-on-Customer-Reviews
```

##### **Step 2.2: Setup Virtual Environment & Install Dependencies**
1. Create and activate the environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Install Dependencies:

```bash
pip install -r requirements.txt
```

##### **Step 2.3: Ensure Model Availability**
The application requires the fine-tuned DistilBERT model to be present in a specific directory. 

**Action Required:** Ensure the following directory exists and contains the unzipped model files.

1. Create Directory: If missing, create the necessary subdirectory structure:
```bash
./models/beauty_sentiment_model_final/
```
2. Unzip Model: Locate the provided compressed file **(beauty_sentiment_model_final.zip)**, which should be available in the main project folder.

- Unzip the contents of this ZIP file (which contains the beauty_sentiment_model_final folder) directly into the ./models/ directory.

- This action will create the complete set of model files (e.g., model.safetensors, config.json) inside the required path: ./models/beauty_sentiment_model_final/.

### **3. Running the Application**

The application is run using **Streamlit**. Navigate to the main project directory and execute the following command:

```bash
streamlit run app/streamlit_app.py
```

1. Automatic Launch: This command will automatically open the web application in your default browser.

2. Access: The application provides a single input text box and a button to analyze the sentiment.

### **4. How to Use the Tool**

1. Input Review: Paste or type a customer review into the text input area on the web page.

2. Analyze: Click the "Analyze Sentiment" button.

3. View Results:

- The Predicted Sentiment (POSITIVE, NEUTRAL, or NEGATIVE) will be displayed prominently.

- A confidence score (bar chart) will show the probability distribution across all three classes, allowing the marketing team to gauge the model's certainty.

 ---

## References
Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and Recommendation* (No. arXiv:2403.03952). arXiv. https://doi.org/10.48550/arXiv.2403.03952

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv Preprint arXiv:1910.01108.


