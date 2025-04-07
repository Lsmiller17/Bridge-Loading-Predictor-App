# Bridge Load Prediction using Artificial Neural Networks

This project uses an artificial neural network (ANN) to predict the maximum load (in tons) a bridge can handle based on various structural features. The model was developed using TensorFlow and deployed with a Streamlit app for interactive use.

Deliverables

Data Exploration and Preprocessing
- Loaded and explored bridge dataset (`lab_11_bridge_data.csv`)
- Handled encoding of the categorical variable (`Material`)
- Standardized numerical features using `StandardScaler`
- Split data into training and test sets (80/20)

Model Development
- Built a simple feedforward ANN using TensorFlow:
  - Input layer with 8 features
  - Hidden layers: [16, 8] with ReLU activation
  - Output layer for regression
- Model predicts `Max_Load_Tons` (target)

raining and Evaluation
- Trained the model over 100 epochs with MSE loss
- Plotted training vs validation loss over epochs  
  → See `training_loss_plot.png`
- Saved model to `tf_bridge_model.h5`
- Saved preprocessing pipeline to `preprocessor.pkl`

Deployment
- Created a Streamlit app (`app.py`)
- Users can input new bridge data and get real-time load predictions

---

How to Run

1. Install Dependencies

```bash
pip install -r requirements.txt
