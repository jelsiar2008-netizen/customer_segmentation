import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================
# LOAD CUSTOMER SEGMENTATION DATASET
# ==========================================

df = pd.read_csv("customer_segmentation_dataset.csv")


# ==========================================
# SELECT FEATURES
# ==========================================

features = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency"
]

X = df[features]


# ==========================================
# STANDARDIZE DATA
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================
# K-MEANS CLUSTERING
# ==========================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ==========================================
# CLUSTER SUMMARY
# ==========================================

summary = df.groupby("Cluster")[features].mean().round(2)


# ==========================================
# CREATE GRAPHS
# ==========================================

fig1 = px.scatter(
    df,
    x="Annual_Income",
    y="Spending_Score",
    color="Cluster",
    hover_data=[
        "CustomerID",
        "Age",
        "Purchase_Frequency"
    ],
    title="Customer Segmentation"
)

fig2 = px.bar(
    df["Cluster"].value_counts().sort_index(),
    title="Number of Customers in Each Segment",
    labels={
        "index": "Customer Segment",
        "value": "Number of Customers"
    }
)

fig3 = px.scatter(
    df,
    x="Age",
    y="Purchase_Frequency",
    color="Cluster",
    hover_data=["Annual_Income", "Spending_Score"],
    title="Age vs Purchase Frequency"
)


# ==========================================
# DASH APP
# ==========================================

app = Dash(__name__)

app.title = "Customer Segmentation"


# ==========================================
# DASHBOARD LAYOUT
# ==========================================

app.layout = html.Div(

    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": "#f5f7fa"
    },

    children=[

        html.H1(
            "Customer Segmentation Dashboard",
            style={
                "textAlign": "center",
                "color": "#1f2937"
            }
        ),

        html.P(
            "Customer analysis using K-Means Clustering",
            style={
                "textAlign": "center",
                "fontSize": "18px"
            }
        ),


        # ==================================
        # SUMMARY CARDS
        # ==================================

        html.Div(

            style={
                "display": "flex",
                "gap": "20px",
                "marginTop": "25px",
                "marginBottom": "30px"
            },

            children=[

                html.Div(
                    [
                        html.H3("Total Customers"),
                        html.H2(str(len(df)))
                    ],
                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px"
                    }
                ),

                html.Div(
                    [
                        html.H3("Customer Segments"),
                        html.H2("4")
                    ],
                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px"
                    }
                ),

                html.Div(
                    [
                        html.H3("Average Income"),
                        html.H2(
                            f"{df['Annual_Income'].mean():.1f}k"
                        )
                    ],
                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px"
                    }
                ),

                html.Div(
                    [
                        html.H3("Average Spending"),
                        html.H2(
                            f"{df['Spending_Score'].mean():.1f}"
                        )
                    ],
                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px"
                    }
                )
            ]
        ),


        # ==================================
        # GRAPH 1
        # ==================================

        dcc.Graph(
            figure=fig1
        ),


        # ==================================
        # GRAPH 2
        # ==================================

        dcc.Graph(
            figure=fig2
        ),


        # ==================================
        # GRAPH 3
        # ==================================

        dcc.Graph(
            figure=fig3
        ),


        # ==================================
        # CLUSTER SUMMARY
        # ==================================

        html.H2(
            "Customer Segment Characteristics"
        ),

        html.Div(

            [

                html.Table(

                    [

                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Segment"),
                                    html.Th("Age"),
                                    html.Th("Annual Income"),
                                    html.Th("Spending Score"),
                                    html.Th("Purchase Frequency")
                                ]
                            )
                        ),

                        html.Tbody(

                            [

                                html.Tr(
                                    [

                                        html.Td(str(cluster)),
                                        html.Td(str(row["Age"])),
                                        html.Td(str(row["Annual_Income"])),
                                        html.Td(str(row["Spending_Score"])),
                                        html.Td(str(row["Purchase_Frequency"]))

                                    ]
                                )

                                for cluster, row in summary.iterrows()

                            ]

                        )

                    ],

                    style={
                        "width": "100%",
                        "textAlign": "center",
                        "backgroundColor": "white",
                        "padding": "10px"
                    }

                )

            ]

        ),


        # ==================================
        # CUSTOMER INSIGHTS
        # ==================================

        html.H2(
            "Customer Insights",
            style={
                "marginTop": "35px"
            }
        ),

        html.Ul(

            [

                html.Li(
                    "Premium customers can be targeted with loyalty rewards and special offers."
                ),

                html.Li(
                    "High-income customers with low spending can be encouraged using personalized promotions."
                ),

                html.Li(
                    "Customers with high spending can be targeted with relevant product recommendations."
                ),

                html.Li(
                    "Low-spending customers can be encouraged through discounts and promotional campaigns."
                )

            ]

        )

    ]

)


# ==========================================
# RUN WEB APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)