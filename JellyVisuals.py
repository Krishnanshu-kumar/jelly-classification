import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file
file_path = 'cleaned_JEDI.csv'
data = pd.read_csv(file_path)

# 1. 3D Scatter Plot: Latitude, Longitude, and Density
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Subset data for plotting to prevent overcrowding
subset_data = data.sample(n=5000, random_state=0)

# Plot Latitude, Longitude, and Density
ax.scatter(subset_data['lat'], subset_data['lon'], subset_data['density'],
           c=subset_data['density'], cmap='viridis', marker='o', alpha=0.5)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Density')
ax.set_title('3D Scatter Plot: Latitude, Longitude, and Density')
plt.show()

# 2. Correlation Heatmap for numerical columns
numerical_data = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Columns")
plt.show()

# 3. Box Plot of Density across different Collection Methods
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='collection_method', y='density', palette='Set2')
plt.yscale('log')  # Log scale to handle outliers
plt.title("Box Plot of Density by Collection Method")
plt.xlabel("Collection Method")
plt.ylabel("Density (log scale)")
plt.xticks(rotation=45)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import plotly.express as px
import plotly.graph_objects as go


class DataVisualizer:
    def __init__(self, file_path):
        """
        Initialize the DataVisualizer with the input file.
        """
        try:
            # Configure logging first
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s: %(message)s')
            self.logger = logging.getLogger(__name__)

            # Handle different file types
            if file_path.endswith('.csv'):
                self.cleaned_df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.cleaned_df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.cleaned_df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")

            self.logger.info(f"Successfully loaded data with {len(self.cleaned_df)} rows")

        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            raise

    def analyze_null_values(self, plot=True, figsize=(15, 8)):
        """
        Analyze and visualize null values.
        """
        self.logger.info("Analyzing Null Values...")

        # Null value analysis
        null_analysis = pd.DataFrame({
            'Total Rows': len(self.cleaned_df),
            'Null Count': self.cleaned_df.isnull().sum(),
            'Null Percentage': (self.cleaned_df.isnull().sum() / len(self.cleaned_df) * 100).round(2),
            'Data Type': self.cleaned_df.dtypes
        }).sort_values('Null Percentage', ascending=False)

        # Optional visualization
        if plot:
            plt.figure(figsize=figsize)
            ax = sns.barplot(x=null_analysis.index,
                             y='Null Percentage',
                             data=null_analysis,
                             palette="RdYlBu_r")
            plt.title('Percentage of Null Values by Column', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Null Percentage (%)')
            plt.xlabel('Columns')

            # Add value labels on top of bars
            for i, v in enumerate(null_analysis['Null Percentage']):
                ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

        return null_analysis

    def visualize_taxonomy_diversity(self, height=600):
        """
        Visualize the diversity across taxonomic ranks.
        """
        try:
            taxonomy_columns = [col for col in self.cleaned_df.columns
                                if col.startswith('rank_')]

            if not taxonomy_columns:
                self.logger.warning("No taxonomy columns found")
                return

            taxonomy_data = {
                'Taxonomic Rank': [col.replace('rank_', '').title() for col in taxonomy_columns],
                'Unique Count': [self.cleaned_df[col].nunique() for col in taxonomy_columns]
            }
            taxonomy_df = pd.DataFrame(taxonomy_data)

            fig = px.bar(taxonomy_df,
                         x='Taxonomic Rank',
                         y='Unique Count',
                         color='Unique Count',
                         title="Diversity Across Taxonomic Ranks",
                         color_continuous_scale='Viridis',
                         height=height)

            fig.update_layout(
                title_x=0.5,
                xaxis_title="Taxonomic Rank",
                yaxis_title="Number of Unique Entries",
                showlegend=True,
                barmode='group'
            )

            # Add value labels on top of bars
            fig.update_traces(texttemplate='%{y}', textposition='outside')

            fig.show()
        except Exception as e:
            self.logger.error(f"Error in taxonomy visualization: {e}")

    def visualize_study_type_distribution(self, height=600):
        """
        Visualize distribution of study types.
        """
        if 'study_type' not in self.cleaned_df.columns:
            self.logger.warning("study_type column not found")
            return

        study_type_counts = (self.cleaned_df['study_type']
                             .value_counts()
                             .reset_index()
                             .rename(columns={'index': 'Study Type',
                                              'study_type': 'Count'}))

        fig = px.bar(study_type_counts,
                     x='Study Type',
                     y='Count',
                     title="Distribution of Study Types",
                     color='Count',
                     color_continuous_scale='Viridis',
                     height=height)

        fig.update_layout(
            title_x=0.5,
            xaxis_tickangle=-45,
            xaxis_title="Study Type",
            yaxis_title="Count",
            showlegend=True
        )

        # Add value labels on top of bars
        fig.update_traces(texttemplate='%{y}', textposition='outside')

        fig.show()

    def visualize_geographic_distribution(self, height=800):
        """
        Create an enhanced geographic visualization using scatter_mapbox.
        """
        if not all(col in self.cleaned_df.columns for col in ['latitude', 'longitude']):
            self.logger.warning("Geographic coordinates not found")
            return

        # Remove any invalid coordinates
        valid_coords = self.cleaned_df[
            (self.cleaned_df['latitude'].between(-90, 90)) &
            (self.cleaned_df['longitude'].between(-180, 180))
            ].copy()

        if len(valid_coords) == 0:
            self.logger.warning("No valid coordinates found")
            return

        fig = px.scatter_mapbox(valid_coords,
                                lat='latitude',
                                lon='longitude',
                                color='study_type' if 'study_type' in valid_coords.columns else None,
                                hover_data=['rank_family'] if 'rank_family' in valid_coords.columns else None,
                                zoom=1,
                                height=height,
                                title="Geographic Distribution of Studies")

        fig.update_layout(
            mapbox_style="carto-positron",
            title_x=0.5,
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )

        fig.show()

