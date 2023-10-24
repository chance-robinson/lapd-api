from config import engine
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

class CrimePredictionModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.length = None
        self.totalLength = None
        self.model2 = None

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
        if not query: 
            query = '''
            SELECT date_occ
            FROM public.crime
            WHERE crm_cd IN (330, 410)
            ORDER BY dr_no ASC
            '''
            df = pd.read_sql(query, engine)
            # Close the engine connection if needed
            engine.dispose()
                    # Data processing
            df['crm_occ'] = 1
            df['date_occ'] = pd.to_datetime(df['date_occ'])
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

            # Split the data into features (X) and target variable (y)
            X = df_merged.drop('crm_occ', axis=1)
            y = df_merged['crm_occ']
        else:
            df = pd.DataFrame(query, columns=cols)
            df['crm_occ'] = 1
            df['date_occ'] = pd.to_datetime(df['date_occ'])
            self.length = df.shape[0]
            df = df.drop(columns=["crm_cd_desc", "area_name", "location", "dr_no"])
            coords = (df['lat'], df['long'])
            time_df = pd.DataFrame({'lat': coords[0], 'long': coords[1], 'date_occ': df['date_occ']})
            time_df['year'] = time_df['date_occ'].dt.year
            time_df['month'] = time_df['date_occ'].dt.month
            time_df['day'] = time_df['date_occ'].dt.day
            time_df['hour'] = time_df['date_occ'].dt.hour
            time_df['minute'] = time_df['date_occ'].dt.minute
            time_df = time_df.drop(columns=["date_occ"])
            # Select the columns for the target variable (lat and long)
            target_df = time_df[['lat', 'long']]
            # Select the columns for the feature variables (year, month, day, hour, minute)
            feature_df = time_df[['year', 'month', 'day', 'hour', 'minute']]
            print(time_df)
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
            return X, y, target_df, feature_df
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def run_model(self, X_train, y_train, target_df, feature_df):
        # Define the model architecture
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        self.history = self.model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=1, validation_split=0.2)

        print(feature_df)
        print(target_df)
                # Define the model architecture
        # self.model2 = keras.Sequential([
        #     keras.layers.Dense(128, activation='relu', input_shape=(feature_df.shape[1],)),
        #     keras.layers.Dense(64, activation='relu'),
        #     keras.layers.Dense(32, activation='relu'),
        #     keras.layers.Dense(2)  # Two output neurons for latitude and longitude, no activation function
        # ])

        # # Compile the model with mean squared error (MSE) as the loss function
        # self.model2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
        #         # Train the model
        # self.history2 = self.model2.fit(feature_df, target_df, epochs=10, batch_size=16, verbose=1, validation_split=0.2)
        print(feature_df)
        print(target_df)

        svr = SVR(kernel='poly')  # You can choose different kernels like 'linear', 'poly', or 'sigmoid'
        self.model2 = MultiOutputRegressor(svr)
        self.model2.fit(feature_df, target_df)



    def y_pred(self, X_test):
        # Evaluate the model on the test set
        y_pred_probs = self.model.predict(X_test)
        sorted_predictions = sorted(y_pred_probs)
        top_percent_index = int((1-(self.length/self.totalLength)) * len(sorted_predictions))  # 94% of the values are below this index
        value_at_percent = sorted_predictions[top_percent_index]
        y_pred = (y_pred_probs >= value_at_percent).astype(int)
        return y_pred
    
    def y_pred2(self, filtered_df):
        occ_copy = filtered_df.copy()
        occ_copy['year'] = occ_copy['date'].dt.year
        occ_copy['month'] = occ_copy['date'].dt.month
        occ_copy['day'] = occ_copy['date'].dt.day
        occ_copy['hour'] = occ_copy['date'].dt.hour
        occ_copy['minute'] = occ_copy['date'].dt.minute
        occ_copy = occ_copy.drop(columns=['date'])
        y_pred2_vals = self.model2.predict(occ_copy)
        return y_pred2_vals
        

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
        X_train, y_train, target_df, feature_df = self.process_data(query, cols)
        self.run_model(X_train, y_train, target_df, feature_df)
        df['crm_occ'] = self.y_pred(df)
        # Combine year, month, day, hour, and minute into a datetime column
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
        # Round the datetime values to the nearest minute
        df['date'] = df['date'].dt.round('1min')
        filtered_df = df[df['crm_occ'] == 1][['date']]
        filtered_df.reset_index(drop=True, inplace=True)
        y_pred_formatted = [[round(value, 6) for value in pair] for pair in self.y_pred2(filtered_df)]
        latlong_preds = pd.DataFrame(y_pred_formatted, columns=['lat', 'long'])
        combined_df = pd.concat([filtered_df, latlong_preds], axis=1)
        combined_df['date'] = combined_df['date'].astype(str)
        print(combined_df)
        return combined_df.to_json(orient='values')

    
# cpm = CrimePredictionModel()
# cpm.localTest()