"""
Visualization Module for Credit Risk Assessment Application.

This module provides comprehensive visualization utilities for:
- Exploratory Data Analysis (EDA)
- Feature distributions and correlations
- Model performance comparison
- ROC curves and confusion matrices
- Feature importance charts

All visualizations are built with Plotly for interactive exploration.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports  
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PLOT_COLORS, PLOT_TEMPLATE


class CreditRiskVisualizer:
    """
    Visualization class for Credit Risk Analysis.
    
    Provides methods for creating interactive visualizations
    for data exploration and model evaluation.
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.colors = PLOT_COLORS
        self.template = PLOT_TEMPLATE
    
    # =========================================================================
    # DATA EXPLORATION VISUALIZATIONS
    # =========================================================================
    
    def plot_target_distribution(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'loan_status'
    ) -> go.Figure:
        """
        Plot the distribution of the target variable.
        
        Args:
            df: DataFrame containing the data
            target_col: Name of target column
            
        Returns:
            Plotly figure object
        """
        value_counts = df[target_col].value_counts()
        labels = ['No Default (0)', 'Default (1)']
        colors = [self.colors['positive'], self.colors['negative']]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=value_counts.values,
                hole=0.4,
                marker_colors=colors,
                textinfo='percent+value',
                textfont_size=14
            )
        ])
        
        fig.update_layout(
            title='Target Variable Distribution (Loan Default Status)',
            template=self.template,
            height=400,
            annotations=[dict(text='Loan<br>Status', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        
        return fig
    
    def plot_numerical_distributions(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        target_col: str = 'loan_status'
    ) -> go.Figure:
        """
        Plot distributions of numerical features by target class.
        
        Args:
            df: DataFrame containing the data
            columns: List of numerical column names
            target_col: Name of target column
            
        Returns:
            Plotly figure object
        """
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=columns,
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        for idx, col in enumerate(columns):
            row = idx // n_cols + 1
            col_idx = idx % n_cols + 1
            
            for status, color, name in [(0, self.colors['positive'], 'No Default'), 
                                        (1, self.colors['negative'], 'Default')]:
                data = df[df[target_col] == status][col].dropna()
                
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name=name,
                        opacity=0.7,
                        marker_color=color,
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col_idx
                )
        
        fig.update_layout(
            title='Numerical Feature Distributions by Loan Status',
            template=self.template,
            height=300 * n_rows,
            barmode='overlay',
            showlegend=True
        )
        
        return fig
    
    def plot_categorical_distributions(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        target_col: str = 'loan_status'
    ) -> go.Figure:
        """
        Plot distributions of categorical features by target class.
        
        Args:
            df: DataFrame containing the data
            columns: List of categorical column names
            target_col: Name of target column
            
        Returns:
            Plotly figure object
        """
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=columns,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for idx, col in enumerate(columns):
            row = idx // n_cols + 1
            col_idx = idx % n_cols + 1
            
            # Calculate default rate by category
            grouped = df.groupby(col)[target_col].agg(['sum', 'count'])
            grouped['default_rate'] = grouped['sum'] / grouped['count'] * 100
            grouped = grouped.sort_values('default_rate', ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=grouped.index.tolist(),
                    y=grouped['default_rate'].values,
                    marker_color=self.colors['primary'],
                    showlegend=False,
                    text=[f'{v:.1f}%' for v in grouped['default_rate'].values],
                    textposition='outside'
                ),
                row=row, col=col_idx
            )
            
            fig.update_yaxes(title_text='Default Rate (%)', row=row, col=col_idx)
        
        fig.update_layout(
            title='Default Rate by Categorical Features',
            template=self.template,
            height=350 * n_rows,
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_matrix(
        self, 
        df: pd.DataFrame, 
        columns: List[str]
    ) -> go.Figure:
        """
        Plot correlation matrix for numerical features.
        
        Args:
            df: DataFrame containing the data
            columns: List of numerical column names
            
        Returns:
            Plotly figure object
        """
        corr_matrix = df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            template=self.template,
            height=600,
            width=800
        )
        
        return fig
    
    def plot_missing_values(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot missing values analysis.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Plotly figure object
        """
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        
        if len(missing) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found in the dataset!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
        else:
            missing_pct = (missing / len(df) * 100).round(2)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=missing.index,
                    x=missing_pct.values,
                    orientation='h',
                    marker_color=self.colors['secondary'],
                    text=[f'{v:.1f}% ({int(c)})' for v, c in zip(missing_pct.values, missing.values)],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title='Missing Values by Feature',
                xaxis_title='Missing Percentage (%)',
                template=self.template,
                height=max(300, len(missing) * 40)
            )
        
        return fig
    
    def plot_outlier_analysis(
        self, 
        df: pd.DataFrame, 
        columns: List[str]
    ) -> go.Figure:
        """
        Plot box plots for outlier analysis.
        
        Args:
            df: DataFrame containing the data
            columns: List of numerical column names
            
        Returns:
            Plotly figure object
        """
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=columns,
            vertical_spacing=0.12
        )
        
        for idx, col in enumerate(columns):
            row = idx // n_cols + 1
            col_idx = idx % n_cols + 1
            
            fig.add_trace(
                go.Box(
                    y=df[col].dropna(),
                    name=col,
                    marker_color=self.colors['primary'],
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title='Outlier Analysis (Box Plots)',
            template=self.template,
            height=300 * n_rows
        )
        
        return fig
    
    # =========================================================================
    # MODEL EVALUATION VISUALIZATIONS
    # =========================================================================
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Plot model performance comparison as grouped bar chart.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            
        Returns:
            Plotly figure object
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, row in comparison_df.iterrows():
            fig.add_trace(go.Bar(
                name=row['Model'],
                x=metrics,
                y=[row[m] for m in metrics],
                marker_color=colors[i % len(colors)],
                text=[f"{row[m]:.3f}" for m in metrics],
                textposition='outside'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Metric',
            yaxis_title='Score',
            template=self.template,
            barmode='group',
            height=500,
            yaxis=dict(range=[0, 1.15]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_roc_curves(
        self, 
        roc_data: Dict[str, Dict],
        title: str = "ROC Curves Comparison"
    ) -> go.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            roc_data: Dictionary with ROC data for each model
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        # Add ROC curve for each model
        for i, (model_name, data) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(
                x=data['fpr'],
                y=data['tpr'],
                mode='lines',
                name=f"{model_name} (AUC={data['auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template=self.template,
            height=500,
            width=700,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        model_name: str = "Model"
    ) -> go.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            cm: Confusion matrix array
            model_name: Name of the model
            
        Returns:
            Plotly figure object
        """
        labels = ['No Default', 'Default']
        
        # Calculate percentages
        cm_pct = cm / cm.sum() * 100
        
        # Create text annotations
        text = [[f'{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)' for j in range(2)] for i in range(2)]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=text,
            texttemplate='%{text}',
            textfont={"size": 14},
            hoverongaps=False,
            showscale=True
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            template=self.template,
            height=400,
            width=500
        )
        
        return fig
    
    def plot_feature_importance(
        self, 
        importance_dict: Dict[str, float], 
        model_name: str = "Model",
        top_n: int = 15
    ) -> go.Figure:
        """
        Plot feature importance as horizontal bar chart.
        
        Args:
            importance_dict: Dictionary mapping feature names to importance scores
            model_name: Name of the model
            top_n: Number of top features to display
            
        Returns:
            Plotly figure object
        """
        # Sort by importance and take top N
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, values = zip(*sorted_importance)
        
        # Reverse for horizontal bar chart
        features = list(features)[::-1]
        values = list(values)[::-1]
        
        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=values,
                orientation='h',
                marker_color=self.colors['primary'],
                text=[f'{v:.4f}' for v in values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance - {model_name}',
            xaxis_title='Importance Score',
            template=self.template,
            height=max(400, top_n * 30)
        )
        
        return fig
    
    def plot_precision_recall_curves(
        self, 
        pr_data: Dict[str, Dict]
    ) -> go.Figure:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            pr_data: Dictionary with PR data for each model
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, (model_name, data) in enumerate(pr_data.items()):
            fig.add_trace(go.Scatter(
                x=data['recall'],
                y=data['precision'],
                mode='lines',
                name=f"{model_name} (AP={data['avg_precision']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            template=self.template,
            height=500,
            width=700
        )
        
        return fig
    
    def plot_training_time_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Plot training time comparison across models.
        
        Args:
            comparison_df: DataFrame with model comparison data
            
        Returns:
            Plotly figure object
        """
        df_sorted = comparison_df.sort_values('Train Time (s)', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                y=df_sorted['Model'],
                x=df_sorted['Train Time (s)'],
                orientation='h',
                marker_color=self.colors['secondary'],
                text=[f'{v:.2f}s' for v in df_sorted['Train Time (s)']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Model Training Time Comparison',
            xaxis_title='Training Time (seconds)',
            template=self.template,
            height=max(300, len(comparison_df) * 40)
        )
        
        return fig
    
    def plot_model_radar(self, comparison_df: pd.DataFrame, top_n: int = 5) -> go.Figure:
        """
        Plot radar chart comparing top models across metrics.
        
        Args:
            comparison_df: DataFrame with model comparison data
            top_n: Number of top models to include
            
        Returns:
            Plotly figure object
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        # Take top N models by ROC-AUC
        top_models = comparison_df.nsmallest(top_n, 'Rank')
        
        for i, (_, row) in enumerate(top_models.iterrows()):
            values = [row[m] for m in metrics]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                name=row['Model'],
                line_color=colors[i % len(colors)],
                fill='toself',
                opacity=0.5
            ))
        
        fig.update_layout(
            title=f'Top {top_n} Models - Radar Comparison',
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            template=self.template,
            height=500
        )
        
        return fig
    
    # =========================================================================
    # DATA PIPELINE VISUALIZATIONS
    # =========================================================================
    
    def plot_data_pipeline_diagram(self) -> go.Figure:
        """
        Create a visual representation of the data pipeline.
        
        Returns:
            Plotly figure object showing the data flow
        """
        fig = go.Figure()
        
        # Define pipeline stages
        stages = [
            ("Data Sources", 0, "Core Banking\nCredit Bureau\nApplication Forms"),
            ("Ingestion", 1, "Batch/Streaming\nData Validation"),
            ("Raw Storage", 2, "Data Lake\nRaw Zone"),
            ("Processing", 3, "Cleaning\nTransformation"),
            ("Feature Store", 4, "Feature Engineering\nFeature Serving"),
            ("ML Training", 5, "Model Training\nValidation"),
            ("Deployment", 6, "Real-time Scoring\nBatch Predictions")
        ]
        
        # Add boxes for each stage
        for name, x, description in stages:
            fig.add_trace(go.Scatter(
                x=[x],
                y=[0],
                mode='markers+text',
                marker=dict(size=60, color=self.colors['primary']),
                text=[name],
                textposition='top center',
                name=name,
                hovertext=description,
                showlegend=False
            ))
        
        # Add arrows between stages
        for i in range(len(stages) - 1):
            fig.add_annotation(
                x=stages[i+1][1] - 0.3,
                y=0,
                ax=stages[i][1] + 0.3,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.colors['neutral']
            )
        
        fig.update_layout(
            title='Credit Risk Data Pipeline Architecture',
            template=self.template,
            height=300,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1])
        )
        
        return fig


def create_visualizer() -> CreditRiskVisualizer:
    """Factory function to create a visualizer instance."""
    return CreditRiskVisualizer()
