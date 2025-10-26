import json
import logging
import time
from typing import List, Optional

import pandas as pd
import plotly.express as px
from google.cloud import storage

def generate_chart_from_data(
    chart_type: str,
    data_json_string: str,
    title: str,
    x_column: str,
    y_columns: List[str], # <-- Key change: accepts a LIST of y-columns
    labels_column: Optional[str] = None,
    values_column: Optional[str] = None,
    color_column: Optional[str] = None
) -> str:
    """
    Generates a chart from JSON data, uploads it to GCS, and returns a public URL.
    
    Args:
        chart_type (str): The type of chart ('bar', 'pie', 'line').
        data_json_string (str): The data in JSON format as a string.
        title (str): The title of the chart.
        x_column (str): The column name for the X-axis.
        y_columns (List[str]): A list of column names for the Y-axis.
        labels_column (Optional[str]): The column for pie chart labels.
        values_column (Optional[str]): The column for pie chart values.
        color_column (Optional[str]): The column to use for coloring lines in a line chart.
    """
    logging.info(f"--- [Chart Tool] Generating '{chart_type}' chart titled '{title}' ---")
    bucket_name = "finoptiagents-generated-graph"
    try:
        data = json.loads(data_json_string)
        if not data:
            return json.dumps({"error": "Input data is empty."})
        
        df = pd.DataFrame(data)
        for col in y_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        fig = None
        if chart_type.lower() == 'bar':
            df_melted = df.melt(id_vars=[x_column], value_vars=y_columns, var_name='Category', value_name='Value')
            fig = px.bar(df_melted, x=x_column, y='Value', color='Category', title=title, barmode='group', template="plotly_white")
        elif chart_type.lower() == 'pie':
            fig = px.pie(df, names=labels_column, values=values_column, title=title, template="plotly_white")
        elif chart_type.lower() == 'line':
            fig = px.line(df, x=x_column, y=y_columns[0], color=color_column, title=title, template="plotly_white")
        else:
            return json.dumps({"error": f"Unsupported chart type: '{chart_type}'."})

        # Save chart to a temporary local file
        chart_filename = f"{title.replace(' ', '_')}_{int(time.time())}.html"
        local_chart_path = f"/tmp/{chart_filename}"
        fig.write_html(local_chart_path)

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(chart_filename)
        blob.upload_from_filename(local_chart_path)

        logging.info(f"--- [Chart Tool] Successfully uploaded chart to GCS: {blob.public_url} ---")
        
        return f"Chart has been generated and is available at: {blob.public_url}"

    except Exception as e:
        logging.error(f"Chart generation or GCS upload failed: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {e}"})
