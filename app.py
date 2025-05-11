import pandas as pd
import dash
import plotly.graph_objects as go
import joblib
import os
from components.model import model_pipeline
from components import tab
from dash import Output, Input
from dash import dash_table


#################### Load pickle file ###################


print("🔍 Checking for results.pkl...")
if os.path.exists("results.pkl"):
    print("⏳ Loading saved results.")
    results = joblib.load("results.pkl")
else:
    print("🚀 File NOT found. Running pipeline...")
    kpis_dict, roc_auc_results, models = model_pipeline()
    results = {
        "kpis": kpis_dict,
        "roc_auc": roc_auc_results,
        "models": models
    }
    joblib.dump(results, "results.pkl")
    print("✅ Results saved.")


    


# Load original data for preview table (first 5 rows)
full_data_preview = pd.read_csv("components/data/cleaned_dataset_no_ghosts.csv")
preview_columns = ["gender", "time_in_hospital", "num_lab_procedures", "max_glu_serum",
                   "A1Cresult", "metformin", "insulin", "change", "visit", "readmitted"]

preview_table_df = full_data_preview[preview_columns].head(5)



fig_kpi = go.Figure()  


roc_data = results["roc_auc"]
kpi_data = results["kpis"]

# Get list of model names
model_names = list(roc_data.keys())

five_patients_df = pd.read_csv("components/data/five_patients_df.csv")



# ################### Initialize the Dash app ##################

app = dash.Dash(__name__)
app.layout = tab.create_layout(results, preview_table_df)


################### Callback ##################


@app.callback(
    Output("roc-graph", "figure"),
    Output("kpi-graph", "figure"),
    Output("prediction-table", "children"),
    Input("model-dropdown", "value")
)
def update_charts(selected_models):
    if not selected_models:
        return go.Figure(), go.Figure()

    # === ROC Plot with variable line width ===
    roc_fig = go.Figure()

    for i, model in enumerate(selected_models):
        roc_data = results["roc_auc"][model]
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]

        roc_fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=model,
            line=dict(width=8 - i)  
        ))

    # Add reference line
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Guess",
        line=dict(dash="dash", color="black", width=2)
    ))

    roc_fig.update_layout(
        title=dict(
        text="ROC Curves",
        font=dict(size=26, family="Arial, sans-serif")
        ),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True, linecolor='black', linewidth=2),
        yaxis=dict(showgrid=False, showline=True, linecolor='black', linewidth=2)
    )

    # === KPI Bar Chart ===
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]
    kpi_fig = go.Figure()

    for model in selected_models:
        kpi_data = results["kpis"][model]
        auc = results["roc_auc"][model]["auc"]

        values = [
            kpi_data["Accuracy"],
            kpi_data["Macro Precision"],
            kpi_data["Macro Recall"],
            kpi_data["Macro F1"],
            auc
        ]

        kpi_fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            name=model
        ))

    kpi_fig.update_layout(
        title=dict(
        text="Model's KPIs",
        font=dict(size=26, family="Arial, sans-serif")
        ),
        yaxis=dict(range=[0, 1], showgrid=False, showline=True, linecolor='black', linewidth=2),
        xaxis=dict(showline=True, linecolor='black', linewidth=2),
        barmode="group",  # Show bars side-by-side
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )


    # === Prediction Table ===
    selected_columns = [
        "gender", "time_in_hospital", "num_lab_procedures", "max_glu_serum",
        "A1Cresult", "metformin", "insulin", "change", "diabetesMed"  
    ]
    table_df = five_patients_df[selected_columns].copy()


    for model_name in selected_models:
        column_name = f"Prediction for {model_name}"
        table_df[column_name] = results["models"][model_name].predict(five_patients_df)

    # Define which columns to show in what order (first original features, then prediction columns)
    columns = [{"name": col, "id": col} for col in table_df.columns]

    prediction_table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=columns,
        style_table={'overflowX': 'auto', 'border': '2px solid black',},
        style_cell={'textAlign': 'center', 'border': '1.5px solid #cccccc', 'fontFamily': 'Arial, sans-serif', 'fontSize': '18px'},
        style_data={'border': '1.5px solid #cccccc'},
        style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold', 'border': '1.5px solid #cccccc'}
        )

    return roc_fig, kpi_fig, prediction_table




################### Run the app ##################
if __name__ == "__main__":
    app.run_server(debug=True)