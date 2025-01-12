# Market Anomaly Detection Using LSTM

## Overview
Financial markets often exhibit anomalies—unusual patterns that deviate from expected behavior. These anomalies can provide valuable insights into market volatility and risk. This project leverages a **Long Short-Term Memory (LSTM)** neural network to detect anomalies in the **VIX Index**, also known as the "fear index." By combining time-series analysis, machine learning, and anomaly detection techniques, the project aims to uncover unusual patterns in market behavior.

---

## **Features**
- **Data Preprocessing**: Scales and sequences financial data for effective time-series analysis.
- **LSTM Model**: Employs a deep learning model with two LSTM layers and dropout for regularization, tailored for sequential data.
- **Anomaly Detection**: Identifies anomalies by setting a threshold based on the 95th percentile of training errors.
- **Visualization**: Provides plots for training/validation loss, prediction errors, and detected anomalies.
- **Feature Importance Analysis**: Highlights the most influential features in predicting the VIX Index.
- **Future Predictions**: Predicts future VIX values using the trained LSTM model.

---

## **Technologies Used**
- **Python**: The primary programming language for data analysis and machine learning tasks.
- **TensorFlow/Keras**: Utilized for constructing and training the LSTM model.
- **Pandas**: Essential for data manipulation and analysis.
- **NumPy**: Employed for numerical computations.
- **Matplotlib**: Used for data visualization.
- **Scikit-learn**: Implemented for data preprocessing and feature scaling.

---

## **Dataset**
The dataset encompasses the following financial indicators:
- **VIX Index**: Measures market volatility.
- **Nasdaq Futures**: Represents futures contracts for the Nasdaq stock index.
- **30-Year Treasury Yield**: Indicates the yield on 30-year U.S. Treasury bonds.
- **Gold Futures**: Encompasses futures contracts for gold prices.

The data is preprocessed to form sequences of 10 time steps, which are subsequently used to predict the next value of the VIX Index.

---

## **Project Workflow**

### **1. Data Preparation**
- Load and preprocess the dataset.
- Scale the data using `MinMaxScaler`.
- Create sequences of 10 time steps for time-series analysis.

### **2. Model Building**
Construct an LSTM model featuring:
- Two LSTM layers with 64 and 32 units.
- Dropout layers to mitigate overfitting.
- A Dense layer for output predictions across all features.

### **3. Model Training**
- Train the model over 50 epochs with a batch size of 32.
- Allocate 10% of the training data for validation purposes.

### **4. Anomaly Detection**
- Compute the **Mean Absolute Error (MAE)** for model predictions.
- Establish an anomaly threshold at the 95th percentile of training errors.
- Flag predictions with errors exceeding the threshold as anomalies.

### **5. Visualization**
- Plot training and validation loss over epochs.
- Utilize **SHAP** or **LIME** for model explainability to better understand the model’s predictions.

---

## **How to Run the Project**
Follow these steps to set up and run the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sashank-Singh/Detecting-Market-Anomalies-with-LSTM.git
   cd Detecting-Market-Anomalies-with-LSTM
   jupyter notebook 
