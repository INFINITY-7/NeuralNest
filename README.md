NeuralNest 🏠
House Price Estimation using an Artificial Neural Network (ANN)
NeuralNest is a beginner-friendly Python machine-learning project that estimates a house's price from features such as floor area, bedrooms, bathrooms, age, location score, and parking. It uses TensorFlow/Keras to train a small feed-forward neural network and lets the user enter a new property's details in the terminal for an estimate.
> **Disclaimer:** This project generates synthetic data using an assumed formula. It is a learning/demo model—not a real property valuation tool. Predictions do not reflect actual market prices.
Features
Generates 1,000 synthetic housing samples
Uses six house features: area, bedrooms, bathrooms, age, location score, and parking
Splits data into training and test sets
Standardises input features with `StandardScaler`
Trains a feed-forward neural network with TensorFlow/Keras
Reports test loss and Mean Absolute Error (MAE)
Accepts manual house details through terminal prompts
Prints the predicted price in lakhs and rupees
Model architecture
Layer	Units	Activation
Input / Hidden layer 1	16	ReLU
Hidden layer 2	8	ReLU
Output	1	Linear
Training configuration: Adam optimizer · Mean Squared Error loss · 100 epochs · batch size 16 · 20% validation split of the training data.
Input features
Feature	Expected input
Area	Square feet (sq.ft)
Bedrooms	Number of bedrooms
Bathrooms	Number of bathrooms
Age	House age in years
Location score	Integer from 1 to 10 (synthetic rating)
Parking	Number of parking spaces (0–2 in generated data)
The sample price formula creates target values in lakhs. It is intentionally assumed for demonstration and does not represent a fitted real-estate market model.
Requirements
Python 3.9+ (a Python version compatible with the installed TensorFlow release is recommended)
pip

Installation and running
Clone this repository:
```bash
   git clone https://github.com/YOUR-USERNAME/NeuralNest.git
   cd NeuralNest
   ```
(Recommended) Create and activate a virtual environment:
Windows:
```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
macOS/Linux:
```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
Install dependencies:
```bash
   pip install -r requirements.txt
   ```
Run the program:
```bash
   python src/main.py
   ```
Follow the terminal prompts to enter the house details. The model trains at startup, then displays an estimated price.
Example terminal session
```text
--- House Price Estimator ---
Enter house area (sq.ft): 1500
Enter number of bedrooms: 3
Enter number of bathrooms: 2
Enter house age (years): 4
Enter location score (1-10): 8
Enter number of parking spaces: 1

HOUSE PRICE ESTIMATION
Estimated Price: Rs. [model output] Lakhs
Estimated Value: Rs. [model output]
```
Your numeric output will vary based on training and the generated data. The example above is illustrative, not a promised result.
Project structure
```text
NeuralNest/
├── src/
│   └── main.py
├── assets/
│   └── README.md
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
Future improvements
Train using a real housing dataset rather than synthetic formula-generated values
Add locality-level features and property type
Save/load the trained model and scaler instead of retraining each run
Add a graphical interface using Streamlit
Visualise training history and actual-vs-predicted prices
Compare ANN with Linear Regression and Random Forest baselines
Add input validation and meaningful feature ranges
Built with
Python
NumPy
scikit-learn
TensorFlow / Keras

