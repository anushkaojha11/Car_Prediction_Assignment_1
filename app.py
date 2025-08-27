#Importing packages

import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import pandas as pd
import numpy as np
import pickle
import dash_bootstrap_components as dbc

#Initializing the app 

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#Loading data 
vehicle_df = pd.read_csv('Cars.csv')

#Loading models
model = pickle.load(open("model/car_prediction.model",'rb'))
scalar = pickle.load(open("model/prediction_scalar.model",'rb'))
label_car = pickle.load(open("model/brand-label.model",'rb'))

#Assigning the categories
brand_cat = list(label_car.classes_)
num_cols = ['max_power', 'mileage', 'year']

default_values = {
    'max_power': 40,
    'mileage': 40,
    'year': 2012,
    'brand': 'BMW',
}

app.layout = dbc.Container(
    dbc.Row(
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H1("Car Price Prediction", className="text-center mb-4"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Brand"),
                            dcc.Dropdown(id="brand", options=brand_cat, value=brand_cat[0])
                        ], width=4),

                        dbc.Col([
                            dbc.Label("Year"),
                            dcc.Dropdown(
                                id="year",
                                options=[{"label": y, "value": y} for y in sorted(vehicle_df['year'].unique())],
                                value=vehicle_df['year'].min()
                            )
                        ], width=4),

                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Mileage (km/l)"),
                            dcc.Input(id="mileage", type="number", value=0, style={"width": "100%"})
                        ], width=6),

                        dbc.Col([
                            dbc.Label("Max Power (bhp)"),
                            dcc.Input(id="max_power", type="number", value=0, style={"width": "100%"})
                        ], width=6),
                    ], className="mb-3"),

                    dbc.Button("Predict Price", id="submit", color="primary", className="w-100 mb-3"),
                    html.Div(id="prediction_result", className="text-center fs-4 fw-bold")
                ]),
                className="shadow p-4"
            ),
            width=8,  # make card 8/12 of the page width
            className="mx-auto my-5"  # center horizontally and add vertical margin
        )
    ),
    fluid=True
)


@callback(
    Output("prediction_result", "children"),
    Input("submit", "n_clicks"),
    State("max_power", "value"),
    State("mileage", "value"),
    State("year", "value"),
    State("brand", "value"),
    prevent_initial_call=True
)

def predict_price(n, max_power, mileage, year, brand):
    # Handle missing/invalid inputs
    features = {
        "max_power": max_power,
        "mileage": mileage,
        "year": year,
        "brand": brand,
    }

    for f in features:
        if not features[f]:
            features[f] = default_values[f]
        elif f in num_cols and features[f] < 0:
            features[f] = default_values[f]

    # Convert to dataframe
    X = pd.DataFrame(features, index=[0])


    # Scale numeric
    X[num_cols] = scalar.transform(X[num_cols])

    # Encode categorical
    X['brand'] = label_car.transform(X['brand'])

    # Prediction
    price = np.round(np.exp(model.predict(X)), 2)[0]
   

    return f"The predicted price of the model is: ${price}"

#Running app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)


