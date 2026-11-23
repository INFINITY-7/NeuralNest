import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def generate_sample_data(n=1000):
    """Generate synthetic housing data and assumed prices (in lakhs)."""
    np.random.seed(42)
    area = np.random.randint(500, 3000, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 4, n)
    age = np.random.randint(0, 30, n)
    location_score = np.random.randint(1, 11, n)
    parking = np.random.randint(0, 3, n)

    price = (
        area * 0.025
        + bedrooms * 4
        + bathrooms * 3
        + location_score * 5
        - age * 0.5
        + parking * 2
        + np.random.normal(0, 5, n)
    )

    X = np.column_stack([area, bedrooms, bathrooms, age, location_score, parking])
    y = price
    return X, y



def build_model():
    """Build a small feed-forward neural network for regression."""
    model = Sequential([
        Dense(16, activation="relu", input_shape=(6,)),
        Dense(8, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model



def main():
    print("Generating sample housing data...")
    X, y = generate_sample_data()



    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )



    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    model = build_model()
    print("Training neural network...")
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
    )



    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.2f} lakhs")
    print(f"Test MSE: {loss:.2f}")



    print("\n--- House Price Estimator ---")
    try:
        area = float(input("Enter house area (sq.ft): "))
        bedrooms = int(input("Enter number of bedrooms: "))
        bathrooms = int(input("Enter number of bathrooms: "))
        age = int(input("Enter house age (years): "))
        location_score = int(input("Enter location score (1-10): "))
        parking = int(input("Enter number of parking spaces: "))

        new_house = np.array([[area, bedrooms, bathrooms, age, location_score, parking]])
        new_house_scaled = scaler.transform(new_house)
        prediction = model.predict(new_house_scaled, verbose=0)
        price_in_lakhs = float(prediction[0][0])

        print("\n-----------------------------")
        print("   HOUSE PRICE ESTIMATION")
        print("-----------------------------")
        print(f"Estimated Price: Rs. {price_in_lakhs:.2f} Lakhs")
        print(f"Estimated Value: Rs. {price_in_lakhs * 100000:,.0f}")
        print("-----------------------------")
    except ValueError:
        print("Invalid input. Please enter numeric values for all house details.")



if __name__ == "__main__":
    main()
