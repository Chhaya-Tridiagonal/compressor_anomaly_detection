import asyncio
import logging
from datetime import timedelta
from kelvin.application import KelvinApp, filters
from kelvin.message import Number, ControlChange, Recommendation
from kelvin.krn import KRNAssetDataStream
from rolling_window import RollingWindow
import dill
import pandas as pd
import joblib
import shap
import numpy as np



async def process_data(app: KelvinApp, asset: str,df: pd.DataFrame) -> None:
    with open(r'anomaly_wrapper_class_v11.pkl', 'rb') as f:
        # AnomalyDetectionWrapper = pickle.load(f)
        AnomalyDetectionWrapper = dill.loads(f.read())

        # Initialize the class instance
        wrapper = AnomalyDetectionWrapper
        print('wrapper_loaded')

        # Load the model
        wrapper.load_model(r'model_new.joblib')
        print('model_loaded')

        if df.shape[1] < 28:
            return None
        df.fillna(0, inplace=True)
        df = df[df['compressor_speed']!=0]
        if df.shape[0]< 3:
            return None
#        comp_list = []
 #       comp_value = df.reset_index(drop=True)[['compressor_speed']].values.tolist()
  #      comp_list.append(comp_value)
        # Define the path to the new CSV file
        # df = r"new_data.csv"

        # Detect anomalies and get the result DataFrame
        # data = wrapper.preprocess_data(df)

        event_return = wrapper.detect_anomalies(df)
        if event_return[-1] == -1:
              print("Anomaly Detected")
        print(df.columns)
        print(event_return)

        current_rpm = df['compressor_speed'].iloc[-1]
        # current_rpm = 3000
        print(current_rpm)
        # print(current_rpm)
        # data['compressor_speed']
        previous_rpm = df['compressor_speed'].iloc[-2]
        print(previous_rpm)
        # compressor speed at (t-10)

       # new_set_point = wrapper.monitor_rpm(curr_rpm, prev_rpm)
        #print(new_set_point)
        rpm_change_percentage = ((current_rpm - previous_rpm) / previous_rpm) * 100
        print(rpm_change_percentage)
        if rpm_change_percentage < -0.03:
            new_set_point = previous_rpm+(current_rpm - previous_rpm) * 0.05
        elif rpm_change_percentage > 0.03:
            new_set_point = previous_rpm-(current_rpm - previous_rpm) * 0.05
        else:
            new_set_point = None
        print(new_set_point)
        if new_set_point is not None:
            await app.publish(
                 ControlChange(
                    resource=KRNAssetDataStream(asset, "compressor_speed"),
                    payload=new_set_point,
                    expiration_date=timedelta(minutes=5)
                )
            )
           # print(f"Control change to  RPM by: {control_change}%")

        if new_set_point is not None:
            await app.publish(
                Recommendation(
                    resource=KRNAsset(asset=asset),
                    type="compressor-rpm-change",
                    description='check-new-set-point',
                    # description=event_return['recommended_actions'].values[0].split(",")[:1],
                    expiration_date=timedelta(hours=1),
 #                   control_changes=[control_change]
                )
            )


async def main() -> None:
    # Creating instance of Kelvin App Client
    app = KelvinApp()

    # Connect the App Client
    await app.connect()

    # Subscribe to the asset data streams
    msg_queue: asyncio.Queue[Number] = app.filter(filters.is_asset_data_message)
    print(msg_queue)

    # Create a rolling window
    rolling_window = RollingWindow(max_data_points=100, timestamp_rounding_interval=timedelta(seconds=1))

    while True:
        # Await a new message from the queue
        message = await msg_queue.get()
        print(message)

        # Add the message to the rolling window
        rolling_window.add_message(message)

        # Get asset
        asset = message.resource.asset

        # Retrieve dataframe from the rolling window for the specified asset
        df = rolling_window.get_asset_dataframe(asset)

        # Process the data
        await process_data(app, asset, df)


if __name__ == "__main__":
    asyncio.run(main())
