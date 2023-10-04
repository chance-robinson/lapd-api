import pandas as pd
from config import engine
from tensorflow import keras
# from scikit-learn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# import plotly.express as px
from datetime import datetime, timedelta

class CrimePredictionModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.length = None
        self.totalLength = None

    # def print_metrics(self, X_test, y_test, y_pred):
    #     # Compute the confusion matrix
    #     cm = confusion_matrix(y_test, y_pred)
    #     print("Confusion Matrix:")
    #     print(cm)

    #     # Compute precision, recall, and F1 score
    #     report = classification_report(y_test, y_pred)
    #     print("Classification Report:")
    #     print(report)

    #     # Print test loss and accuracy
    #     test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
    #     print("Test Loss:", test_loss)
    #     print("Test Accuracy:", test_accuracy)

    #     # Print validation accuracy
    #     val_accuracy = self.history.history['val_accuracy']
    #     print("Validation Accuracy:", val_accuracy)

    # def parallel_coordinate_plot(self, X_test, y_pred, y_col_name):
    #     df_pred = X_test.copy()
    #     df_pred[y_col_name] = y_pred

    #     fig = px.parallel_coordinates(df_pred, color=f'{y_col_name}')
    #     fig.show()

    def process_data(self, query, cols):
        df = pd.DataFrame(query, columns=cols)
        df['crm_occ'] = 1
        df['date_occ'] = pd.to_datetime(df['date_occ'])
        self.length = df.shape[0]
        df = df.drop(columns=["crm_cd_desc", "area_name", "location", "dr_no"])
        coords = (df['lat'], df['long'])
        df = df.drop(columns=["lat", "long"])
        # Find the newest and oldest dates
        newest_date = df['date_occ'].max()
        oldest_date = df['date_occ'].min()
        # Create a new DataFrame with minute-level granularity
        minute_range = pd.date_range(start=oldest_date, end=newest_date, freq='min')
        df_new = pd.DataFrame({'date_occ': minute_range})
        # Merge the new DataFrame with the original DataFrame based on 'date_occ'
        df_merged = pd.merge(df_new, df, on='date_occ', how='left')
        # Fill missing values in 'crm_occ' with 0
        df_merged['crm_occ'].fillna(0, inplace=True)
        # Extract the year, month, day, hour, and minute into separate columns
        df_merged['year'] = df_merged['date_occ'].dt.year
        df_merged['month'] = df_merged['date_occ'].dt.month
        df_merged['day'] = df_merged['date_occ'].dt.day
        df_merged['hour'] = df_merged['date_occ'].dt.hour
        df_merged['minute'] = df_merged['date_occ'].dt.minute
        newest_date = df_merged['date_occ'].max()
        oldest_date = df_merged['date_occ'].min()
        df_merged = df_merged.drop('date_occ', axis=1)
        self.totalLength = df_merged.shape[0]
        # Split the data into features (X) and target variable (y)
        X = df_merged.drop('crm_occ', axis=1)
        y = df_merged['crm_occ']
        return X, y

    def run_model(self, X_train, y_train):
        # Define the model architecture
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        self.history = self.model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1, validation_split=0.2)

    def y_pred(self, X_test):
        # Evaluate the model on the test set
        y_pred_probs = self.model.predict(X_test)
        sorted_predictions = sorted(y_pred_probs)
        top_percent_index = int((1-(self.length/self.totalLength)) * len(sorted_predictions))  # 94% of the values are below this index
        value_at_percent = sorted_predictions[top_percent_index]
        y_pred = (y_pred_probs >= value_at_percent).astype(int)
        return y_pred

    def localTest(self):
        X_train, X_test, y_train, y_test = self.process_data(None, None)
        self.run_model(X_train, y_train)
        y_pred = self.y_pred(X_test)
        self.print_metrics(X_test, y_test, y_pred)
        self.parallel_coordinate_plot(X_test, y_pred, 'crm_occ')

    def runAll(self, query, cols, days):
        start_date = datetime.now().replace(second=0, microsecond=0)
        end_date = (start_date + timedelta(days=days)).replace(second=0, microsecond=0)
        date_range = pd.date_range(start=start_date, end=end_date, freq='min')
        df = pd.DataFrame({
            'year': date_range.year,
            'month': date_range.month,
            'day': date_range.day,
            'hour': date_range.hour,
            'minute': date_range.minute
        })
        X_train, y_train = self.process_data(query, cols)
        self.run_model(X_train, y_train)
        df['crm_occ'] = self.y_pred(df)
        # Combine year, month, day, hour, and minute into a datetime column
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
        # Round the datetime values to the nearest minute
        df['date'] = df['date'].dt.round('1min')
        filtered_df = df[df['crm_occ'] == 1][['date']]
        return filtered_df['date'].astype(str).to_json(orient='values')

    
# cpm = CrimePredictionModel()
# cpm.localTest()