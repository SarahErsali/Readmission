from dash import html, dcc
from dash import dash_table

def create_layout(results, preview_table_df):
    styles = {
        'header': {
            'backgroundColor': '#1f77b4',
            'padding': '30px',
            'color': 'white',
            'textAlign': 'center',
            'fontSize': '36px',
            'marginBottom': '40px',
            'borderRadius': '10px',
            'border': '2px solid #ffffff'
        },
        'frame': {
            'border': '2px solid #1f77b4',
            'borderRadius': '10px',
            'padding': '20px',
            'margin': '20px',
            'backgroundColor': '#f9f9f9',
            'boxShadow': '2px 2px 10px rgba(0, 0, 0, 0.1)'
        },
        'details': {
            'fontSize': '22px',
            'color': '#333',
            'marginBottom': '15px',
            'lineHeight': '1.6',
            'fontFamily': 'Arial, sans-serif'
        },
        'container': {
            'display': 'flex',
            'flexDirection': 'row',
            'justifyContent': 'space-between'
        },
        'leftPanel': {
            'width': '25%',
            'padding': '20px',
            'border': '2px solid #cccccc',
            'borderRadius': '10px',
            'backgroundColor': '#ffffff',
            'boxShadow': '1px 1px 5px rgba(0, 0, 0, 0.05)'
        },
        'rightPanel': {
            'width': '70%',
            'padding': '20px',
            'border': '2px solid #cccccc',
            'borderRadius': '10px',
            'backgroundColor': '#ffffff',
            'boxShadow': '1px 1px 5px rgba(0, 0, 0, 0.05)'
        },
        'explanationFrame': {
            'border': '2px solid #1f77b4',
            'borderRadius': '10px',
            'padding': '20px',
            'margin': '20px',
            'backgroundColor': '#ffffff',
            'boxShadow': '1px 1px 5px rgba(0, 0, 0, 0.05)'
        }
    }

    return html.Div([
        # Header
        html.Div([
            html.H1("Banex Consulting Data Insight Service", style={'fontSize': '40px', 'margin': '10', 'fontFamily': 'Arial, sans-serif'}),
            html.H2("Hospital Readmission Predictive Models for Diabetes", style={'fontSize': '24px', 'margin': '0', 'fontFamily': 'Arial, sans-serif'})
        ], style=styles['header']),

        # Explanation Frame
        html.Div([
            html.H3("Introduction", style={'fontSize': '26px', 'fontFamily': 'Arial, sans-serif'}),
            html.P(
                "The dataset spans a decade (1999–2008) of clinical care across 130 U.S. hospitals and integrated delivery networks. "
                "Each record reflects a patient diagnosed with diabetes, including details about their hospital stay (up to 14 days), lab tests, and medications.",
                style=styles['details']
            ),
            html.P(
                "Inadequate diabetes management not only escalates hospital operational costs due to frequent readmissions but also worsens patient outcomes "
                "by increasing the risk of diabetes-related complications, morbidity, and mortality.",
                style=styles['details']
            ),

            html.H3("Business Objective", style={'fontSize': '26px', 'fontFamily': 'Arial, sans-serif'}),
            html.P(
                "The goal of this project is to identify and understand the key factors contributing to early hospital readmissions among diabetic patients. "
                "Although various treatment strategies have proven effective in improving clinical outcomes for diabetes, a significant portion of patients either do not receive these treatments or fail to adhere to them after discharge.",
                style=styles['details']
            ),
            html.P(
                "This gap in diabetes care leads to costly hospital readmissions and puts patients at increased risk of serious health complications. "
                "By analyzing historical patient data, we aim to build predictive models that can flag individuals at high risk of readmission — supporting proactive care planning and ultimately improving both health outcomes and hospital resource efficiency.",
                style=styles['details']
            ),

            html.H3("Key Features", style={'fontSize': '26px', 'fontFamily': 'Arial, sans-serif'}),
            html.P(
                "The dataset consists of over 50 features capturing patient information and hospital outcomes. "
                "Records were selected based on several criteria: the encounter had to be an inpatient admission for diabetes, "
                "with a hospital stay between 1 and 14 days. Each record also required evidence of laboratory tests and administered medications during the visit.",
                style=styles['details']
            ),
            html.P(
                "The attributes include demographic details (age, gender), clinical indicators (A1C results, glucose levels), "
                "hospital-related variables (admission type, time in hospital), and treatment information (medications, diabetes drugs, number of visits). "
                "Below is a sample of selected features used for model development.",
                style=styles['details']
            ),

            html.Ul([
                html.Li([html.Span("Age ", style={"fontWeight": "bold"}), "The patient's age group (e.g., 60-70, 70-80)."], style=styles['details']),
                html.Li([html.Span("Number of inpatient ", style={"fontWeight": "bold"}), "Past inpatient admissions as an indicator of disease severity."], style=styles['details']),
                html.Li([html.Span("Number of emergency and Number of outpatient ", style={"fontWeight": "bold"}), "Frequency of emergency and outpatient visits."], style=styles['details']),
                html.Li([html.Span("Time spent in hospital ", style={"fontWeight": "bold"}), "Duration of the patient's latest hospital stay."], style=styles['details']),
                html.Li([html.Span("Num of lab procedures ", style={"fontWeight": "bold"}), "Count of lab tests during hospitalization."], style=styles['details']),
                html.Li([html.Span("Num of medications ", style={"fontWeight": "bold"}), "Total medications prescribed during the stay."], style=styles['details']),
                html.Li([html.Span("Maximum glucose serum test ", style={"fontWeight": "bold"}), "Max glucose serum levels observed."], style=styles['details']),
                html.Li([html.Span("A1C test result ", style={"fontWeight": "bold"}), "Long-term blood sugar control indication."], style=styles['details']),
                html.Li([html.Span("Insulin, Metformin, and other diabetes medications ", style={"fontWeight": "bold"}), "Prescribed diabetic drugs during stay."], style=styles['details']),
                html.Li([html.Span("Change and Diabetes Medication ", style={"fontWeight": "bold"}), "Whether medications were altered and if diabetes drugs were prescribed."], style=styles['details']),
            ]),

            html.H3("Sample of Dataset Used for Training", style={'fontSize': '26px', 'marginTop': '30px', 'fontFamily': 'Arial, sans-serif'}),
            dash_table.DataTable(
                data=preview_table_df.to_dict("records"),
                columns=[{"name": col, "id": col} for col in preview_table_df.columns],
                style_table={
                    'overflowX': 'auto',
                    'border': '2px solid black'
                },
                style_cell={
                    'textAlign': 'center',
                    'border': '1.5px solid black',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '18px'
                },
                style_data={
                    'border': '1.5px solid black'
                },
                style_header={
                    'backgroundColor': '#f1f1f1',
                    'fontWeight': 'bold',
                    'border': '1.5px solid black'
                }
            ),

            html.H3("Model Development and Evaluation", style={'fontSize': '26px', 'marginTop': '30px', 'fontFamily': 'Arial, sans-serif'}),
            html.P(
                "In this project, several machine learning models were developed and evaluated — including XGBoost, LightGBM, Random Forest, and an ensemble model combining XGBoost and Random Forest. "
                "Each model was trained in both default and tuned configurations.",
                style=styles['details']
            ),
            html.P(
                "The performance of these models can be explored below using various metrics such as Accuracy, Precision, Recall, F1-score, and AUC, along with corresponding ROC curves. "
                "You can select and compare multiple models side-by-side.",
                style=styles['details']
            ),
            html.P(
                "Additionally, the table beneath the charts displays records of five patients used as test cases. "
                "These are run through the selected model(s) to show how accurately each model predicts hospital readmission outcomes.",
                style=styles['details']
            )
        ], style=styles['explanationFrame']),

        # Frame for Dropdowns + Plots + Prediction Table
        html.Div([
            html.Div([
                html.P("Select a model to view its performance:", style=styles['details']),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[{"label": name, "value": name} for name in results["roc_auc"].keys()],
                    value=[list(results["roc_auc"].keys())[0]],
                    multi=True,
                    style={'marginBottom': '20px'}
                )
            ], style=styles['leftPanel']),

            html.Div([
                dcc.Graph(id="roc-graph"),
                dcc.Graph(id="kpi-graph"),
                html.H3("In this table, data from five patients has been provided to each selected model to predict whether they will be readmitted to the hospital. A prediction of 0 indicates the model expects no readmission, while a prediction of 1 indicates the model predicts the patient will be readmitted.", style={**styles['details'], 'marginTop': '30px', 'fontWeight': 'normal'}),
                html.Div(id="prediction-table", style={'marginTop': '30px'})
            ], style=styles['rightPanel'])
        ], style={**styles['frame'], **styles['container']})

    ], style={"fontFamily": "Arial, sans-serif"})
