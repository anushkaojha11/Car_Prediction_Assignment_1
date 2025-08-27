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
        dbc.Col([

            # Title card
            dbc.Card([
                dbc.CardHeader("Car Price Prediction", 
                               className="bg-primary text-white fs-4 text-center"),
                dbc.CardBody([
                    html.P("Fill out the details below to estimate the price", 
                           className="text-center text-muted mb-0")
                ])
            ], className="shadow mb-4"),

            # Vehicle Info
            dbc.Card([
                dbc.CardHeader("Step 1: Vehicle Info", 
                               className="bg-info text-white fs-5"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Brand", className="fw-bold"),
                            dcc.Dropdown(id="brand", options=brand_cat, value=brand_cat[0])
                        ], md=6),

                        dbc.Col([
                            dbc.Label("Year", className="fw-bold"),
                            dcc.Dropdown(
                                id="year",
                                options=[{"label": y, "value": y} 
                                         for y in sorted(vehicle_df['year'].unique())],
                                value=vehicle_df['year'].min()
                            )
                        ], md=6),
                    ])
                ])
            ], className="shadow mb-4"),

            # Performance
            dbc.Card([
                dbc.CardHeader("Step 2: Performance Details", 
                               className="bg-warning text-dark fs-5"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Mileage (km/l)", className="fw-bold"),
                            dcc.Input(id="mileage", type="number", value=0, 
                                      style={"width": "100%"})
                        ], md=6),

                        dbc.Col([
                            dbc.Label("Max Power (bhp)", className="fw-bold"),
                            dcc.Input(id="max_power", type="number", value=0, 
                                      style={"width": "100%"})
                        ], md=6),
                    ])
                ])
            ], className="shadow mb-4"),

            # Prediction
            dbc.Card([
                dbc.CardHeader("Step 3: Get Prediction", 
                               className="bg-success text-white fs-5"),
                dbc.CardBody([
                    dbc.Button("Predict Price", id="submit", 
                               color="success", className="w-100 mb-3 fs-5"),
                    html.Div(id="prediction_result", 
                             className="text-center fs-3 fw-bold text-success")
                ])
            ], className="shadow")
        ],
        md=8, className="mx-auto"),  
    ),
    fluid=True,
    className="my-5"
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
   

    return f"The predicted price of the model is: ฿{price}"

#Running app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)


