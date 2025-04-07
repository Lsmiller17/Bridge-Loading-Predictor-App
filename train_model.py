
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("lab_11_bridge_data.csv")

# Drop Bridge_ID
df = df.drop(columns=["Bridge_ID"])

# Split features and target
X = df.drop(columns=["Max_Load_Tons"])
y = df["Max_Load_Tons"]

# Define preprocessing
categorical_features = ["Material"]
numerical_features = X.select_dtypes(include=["int64"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(), categorical_features)
])

# Fit and transform data
X_processed = preprocessor.fit_transform(X)
joblib.dump(preprocessor, "preprocessor.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([

    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100)

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig("training_loss_plot.png")

# Save model
model.save("tf_bridge_model.h5")
