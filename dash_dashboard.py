from dash import dcc, html, Dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

merged_df = pd.read_csv('onebigtable.csv')

# Initialize the app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
famous_countries = ['United States', 'United Kingdom','Canada', 'Netherlands', 'Germany', 'India', 'China']

# Example scorecard data (replace with your own logic)
total_conversions = merged_df['composite_key'].nunique()
total_users = merged_df['session_id'].nunique()
conversion_rate = (total_conversions / total_users) * 100

# Layout
app.layout = dbc.Container([
    dbc.Row([
        # Header
        dbc.Col(
            html.H2("RevieFriends User Analysis Dashboard", className="text-center mb-4"),
            width=12
        ),
    ]),
    dbc.Row([
        # Left Column: Filters and Scorecards
        dbc.Col([
            # Filters Section
            dbc.Card([
                dbc.CardBody([
                    html.H5("Filters", className="card-title"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} for country in famous_countries],
                        value=['USA', 'UK', 'Germany'],
                        multi=True,
                        placeholder="Select Country"
                    ),
                    dcc.DatePickerRange(
                        id='date-picker',
                        start_date=merged_df['timestamp'].min(),
                        end_date=merged_df['timestamp'].max(),
                        display_format='YYYY-MM-DD',
                        className="mt-2"
                    ),
                    dcc.Dropdown(
                        id='device-dropdown',
                        options=[
                            {'label': 'All', 'value': 'All'},
                            {'label': 'Mobile', 'value': 'Mobile'},
                            {'label': 'Desktop', 'value': 'Desktop'}
                        ],
                        value='All',
                        placeholder="Select Device",
                        className="mt-2"
                    ),
                ])
            ], color="dark", inverse=True, className="mb-4"),

            # Scorecards Section
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Conversions", className="card-title"),
                    html.P(f"{total_conversions}", className="card-text"),
                ])
            ], color="primary", inverse=True, className="mb-4"),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Users", className="card-title"),
                    html.P(f"{total_users}", className="card-text"),
                ])
            ], color="info", inverse=True, className="mb-4"),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Conversion Rate", className="card-title"),
                    html.P(f"{conversion_rate:.2f}%", className="card-text"),
                ])
            ], color="success", inverse=True, className="mb-4"),
        ], width=2),  # Reduced width to make room for visualizations

        # Right Column: Visualizations
        dbc.Col([
            dbc.Row([
                # First row with Pie Charts and Bar Chart
                dbc.Col(dcc.Graph(id='country-pie-chart'), width=4),
                dbc.Col(dcc.Graph(id='device-pie-chart'), width=4),
                dbc.Col(dcc.Graph(id='ui-element-bar-chart'), width=4),
            ], className="mb-4"),
            dbc.Row([
                # Second row with Time Series and World Map
                dbc.Col(dcc.Graph(id='conversion-trend-chart'), width=6),
                dbc.Col(dcc.Graph(id='bubble-world-map'), width=6),
            ]),
        ], width=10),  # Increased width for visualizations
    ]),
], fluid=True, style={'backgroundColor': '#f8f9fa'})  # Lighter grey background

# Callback for updating pie chart based on selected countries
@app.callback(
    Output('country-pie-chart', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_country_pie_chart(selected_countries):
    if not selected_countries:
        filtered_df = merged_df
    else:
        filtered_df = merged_df[merged_df['country_name'].isin(selected_countries)]
    
    total_conversions = filtered_df['composite_key'].nunique()
    top_10_with_percentage = filtered_df['country_name'].value_counts().head(10).reset_index()
    top_10_with_percentage.columns = ['country_name', 'conversions']
    top_10_with_percentage['percentage'] = (top_10_with_percentage['conversions'] / total_conversions) * 100

    fig = px.pie(top_10_with_percentage, values='percentage', names='country_name', 
                 title="Top 10 Countries by Conversion Percentage")
    return fig

# Callback for device type pie chart
@app.callback(
    Output('device-pie-chart', 'figure'),
    [Input('device-dropdown', 'value')]
)
def update_device_pie_chart(selected_device):
    filtered_df = merged_df if selected_device == 'All' else merged_df[merged_df['is_mobile'] == selected_device]
    
    device_counts = filtered_df['is_mobile'].value_counts().reset_index()
    device_counts.columns = ['is_mobile', 'count']

    fig = px.pie(device_counts, values='count', names='is_mobile', title="Mobile vs Non-Mobile Users")
    return fig

# Callback for conversion trend chart
@app.callback(
    Output('conversion-trend-chart', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_conversion_trend(selected_countries, start_date, end_date):
    if not selected_countries:
        filtered_df = merged_df
    else:
        filtered_df = merged_df[merged_df['country_name'].isin(selected_countries)]
    
    filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & (filtered_df['timestamp'] <= end_date)]

    fig = px.line(filtered_df, x='timestamp', y='composite_key', color='country_name',
                  title="Conversion Trend Over Time (Top 10 Countries)")
    return fig

# Callback for inverted bar chart by UI element
@app.callback(
    Output('ui-element-bar-chart', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_ui_element_bar_chart(selected_countries):
    if not selected_countries:
        filtered_df = merged_df
    else:
        filtered_df = merged_df[merged_df['country_name'].isin(selected_countries)]
    
    ui_element_counts = filtered_df['ui_element'].value_counts().head(10).reset_index()
    ui_element_counts.columns = ['ui_element', 'count']

    fig = px.bar(ui_element_counts, x='count', y='ui_element', orientation='h',
                 title="Top UI Elements", labels={'count':'Conversions', 'ui_element':'UI Element'})
    return fig

# Callback for bubble chart on world map
@app.callback(
    Output('bubble-world-map', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_bubble_world_map(selected_countries):
    # If no country is selected, use the entire DataFrame
    if not selected_countries:
        filtered_df = merged_df
    else:
        # Filter DataFrame based on selected countries
        filtered_df = merged_df[merged_df['country_name'].isin(selected_countries)]

    # Calculate the count of each id
    filtered_df['id_count'] = filtered_df.groupby('id')['id'].transform('count')

    # Create the bubble chart
    fig = px.scatter_geo(
        filtered_df,
        locations="country_name",
        locationmode="country names",
        size="id_count",  # Use id_count for bubble size
        color="important_score",  # Use the importance score for color
        hover_name="country_name",
        title="Conversions by Country and Importance Score",
        projection="natural earth"
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
