"""
LSTM (Long Short-Term Memory) forecasting model using TensorFlow/Keras.
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def create_sequences(data: np.ndarray, lookback: int) -> tuple:
    """Create input sequences and targets for LSTM training."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def fit_lstm(
    series: pd.Series,
    forecast_steps: int = 30,
    lookback: int = 60,
    epochs: int = 50,
    batch_size: int = 32,
    units: int = 64,
    layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> dict:
    """
    Fit LSTM neural network for time series forecasting.
    
    Args:
        series: Time series with DatetimeIndex
        forecast_steps: Number of steps to forecast
        lookback: Number of past time steps to use as input
        epochs: Training epochs
        batch_size: Batch size for training
        units: LSTM units per layer
        layers: Number of LSTM layers
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
    
    Returns:
        dict with forecast, training history, scaler info
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        raise ImportError("TensorFlow or scikit-learn not installed.")
    
    # Suppress TF warnings
    tf.get_logger().setLevel("ERROR")
    
    series = series.dropna().astype(float)
    values = series.values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)
    
    # Adjust lookback if needed
    if lookback >= len(scaled_data) - 10:
        lookback = max(10, len(scaled_data) // 4)
    
    # Create sequences
    X, y = create_sequences(scaled_data, lookback)
    
    if len(X) < 10:
        raise Exception("Not enough data for LSTM training. Need more historical data points.")
    
    # Split into train/validation
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Build model
    model = Sequential()
    
    for i in range(layers):
        return_seq = i < layers - 1
        if i == 0:
            model.add(tf.keras.layers.Bidirectional(LSTM(units, return_sequences=return_seq), input_shape=(lookback, 1)))
        else:
            model.add(tf.keras.layers.Bidirectional(LSTM(units, return_sequences=return_seq)))
        model.add(Dropout(dropout))
    
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    
    # Early stopping and Learning Rate reduction for better optimization
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    
    # Generate forecast
    last_sequence = scaled_data[-lookback:]
    predictions = []
    
    current_input = last_sequence.reshape(1, lookback, 1)
    for _ in range(forecast_steps):
        pred = model.predict(current_input, verbose=0)
        predictions.append(pred[0, 0])
        # Slide window
        current_input = np.append(current_input[0, 1:, :], pred.reshape(1, 1), axis=0)
        current_input = current_input.reshape(1, lookback, 1)
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    forecast_values = scaler.inverse_transform(predictions).flatten()
    
    # Generate future dates
    last_date = series.index[-1]
    freq = pd.infer_freq(series.index) or "D"
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
    
    forecast_series = pd.Series(forecast_values, index=future_dates, name="Forecast")
    
    # Estimate confidence interval (simple approach using training residuals)
    train_pred = model.predict(X, verbose=0)
    train_pred_inv = scaler.inverse_transform(train_pred).flatten()
    y_inv = scaler.inverse_transform(y).flatten()
    residual_std = np.std(y_inv - train_pred_inv)
    
    lower = forecast_series - 1.96 * residual_std
    upper = forecast_series + 1.96 * residual_std
    lower.name = "Lower"
    upper.name = "Upper"
    
    # Fitted values
    fitted_values = scaler.inverse_transform(train_pred).flatten()
    fitted_index = series.index[lookback:lookback + len(fitted_values)]
    fitted = pd.Series(fitted_values, index=fitted_index, name="Fitted")
    
    return {
        "forecast": forecast_series,
        "lower_bound": lower,
        "upper_bound": upper,
        "fitted": fitted,
        "training_loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", []),
        "epochs_trained": len(history.history["loss"]),
        "lookback": lookback,
        "units": units,
        "layers": layers,
    }
