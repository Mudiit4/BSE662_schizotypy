import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats

# Set the page configuration
st.set_page_config(
    page_title="Beads Task vs Schizotypy Analysis",
    page_icon="🧠",
    layout="wide"
)

# Load the data
@st.cache_data
def load_data():
    participant_avg_dtd = pd.read_csv("participant_avg_dtd.csv")
    schizotypy = pd.read_csv("schizotypy.csv")
    trial_wise_data_2 = pd.read_csv("trial_wise_data_2.csv")
    
    # Merge the data
    merged_data = pd.merge(participant_avg_dtd, schizotypy, left_on="participant_id", right_on="PID", how="inner")
    
    return participant_avg_dtd, schizotypy, trial_wise_data_2, merged_data

participant_avg_dtd, schizotypy, trial_wise_data_2, merged_data = load_data()

# Title and introduction
st.title("Beads Drawing Task vs Schizotypy Analysis")
st.markdown("""
This dashboard analyzes the relationship between performance on the beads drawing task and schizotypy scores.
The beads drawing task involves participants making decisions about which jar beads are drawn from,
with different jar ratios (15/85 and 40/60).

**Schizotypy Subscales:**
- SS1: Unusual Experiences
- SS2: Cognitive Disorganisation
- SS3: Introvertive Anhedonia
- SS4: Impulsive Nonconformity
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "DTD Analysis", "Confidence Analysis", "Statistical Analysis", "Clustering Analysis", "Cluster Insights", "Participant-level"])

with tab1:
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Participant Average DTD Summary")
        st.dataframe(participant_avg_dtd.describe())
    
    with col2:
        st.subheader("Schizotypy Scores Summary")
        st.dataframe(schizotypy[["olife_total", "ss1", "ss2", "ss3", "ss4"]].describe())
    
    # Histograms for key metrics
    st.subheader("Distribution of Key Metrics")
    
    # Define the metrics to display
    metrics = {
        "DTD (15/85)": merged_data["dtd_15_85"],
        "DTD (60/40)": merged_data["dtd_60_40"],
        "Decision Confidence (15/85)": merged_data["decision_confidence_15_85"],
        "Decision Confidence (60/40)": merged_data["decision_confidence_60_40"],
        "Total Schizotypy Score": merged_data["olife_total"]
    }
    
    # Create a 3x2 grid of histograms
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (title, data) in enumerate(metrics.items()):
        if i < len(axes):
            sns.histplot(data, kde=True, ax=axes[i])
            axes[i].set_title(title)
    
    # Remove empty subplot if any
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation matrix
    st.subheader("Correlation Matrix between DTD and Schizotypy Measures")
    
    # Select relevant columns for correlation
    correlation_columns = ["dtd_15_85", "dtd_60_40", "decision_confidence_15_85", 
                           "decision_confidence_60_40", "olife_total", "ss1", "ss2", "ss3", "ss4"]
    
    correlation_matrix = merged_data[correlation_columns].corr()
    
    # Create a heatmap using Plotly for better interactivity
    fig = px.imshow(correlation_matrix,
                   labels=dict(color="Correlation"),
                   x=correlation_matrix.columns,
                   y=correlation_matrix.columns,
                   color_continuous_scale="RdBu_r",
                   zmin=-1, zmax=1)
    
    fig.update_layout(width=800, height=700)
    st.plotly_chart(fig)

with tab2:
    st.header("Draws to Decision (DTD) Analysis")
    
    # Scatterplots of DTD vs Schizotypy scores
    st.subheader("DTD vs Schizotypy Scores")
    
    # For each ratio, create scatterplots against schizotypy measures
    dtd_measures = ["dtd_15_85", "dtd_60_40"]
    schizo_measures = ["olife_total", "ss1", "ss2", "ss3", "ss4"]
    schizo_labels = ["Total Schizotypy", "Unusual Experiences", "Cognitive Disorganisation", 
                     "Introvertive Anhedonia", "Impulsive Nonconformity"]
    
    for dtd_col in dtd_measures:
        ratio = "15/85" if dtd_col == "dtd_15_85" else "60/40"
        st.write(f"### DTD for {ratio} Jar Ratio")
        
        # Create multiple plots in tabs
        dtd_tabs = st.tabs(schizo_labels)
        
        for i, (tab, schizo_col, label) in enumerate(zip(dtd_tabs, schizo_measures, schizo_labels)):
            with tab:
                # Calculate correlation
                corr, p_value = pearsonr(merged_data[dtd_col], merged_data[schizo_col])
                
                fig = px.scatter(merged_data, x=schizo_col, y=dtd_col,
                               hover_data=["participant_id"],
                               labels={dtd_col: f"DTD ({ratio})", schizo_col: label},
                               trendline="ols")
                
                fig.update_layout(
                    title=f"DTD ({ratio}) vs {label} (r={corr:.2f}, p={p_value:.3f})",
                    height=500
                )
                
                st.plotly_chart(fig)
                
                # Display statistical significance
                alpha = 0.05
                if p_value < alpha:
                    st.success(f"Statistically significant correlation (p={p_value:.3f})")
                else:
                    st.info(f"No statistically significant correlation (p={p_value:.3f})")
    
    # Comparison of DTD between different jar ratios
    st.subheader("Comparison of DTD between Different Jar Ratios")
    
    fig = px.scatter(merged_data, x="dtd_15_85", y="dtd_60_40", 
                   hover_data=["participant_id", "olife_total"],
                   labels={"dtd_15_85": "DTD (15/85)", "dtd_60_40": "DTD (60/40)"},
                   trendline="ols")
    
    fig.update_layout(title="DTD Comparison: 15/85 vs 60/40", height=600)
    st.plotly_chart(fig)
    
    # Calculate if there's a significant difference between DTD for different ratios
    t_stat, p_val = stats.ttest_rel(merged_data["dtd_15_85"], merged_data["dtd_60_40"])
    
    st.write(f"T-test result: t={t_stat:.3f}, p={p_val:.3f}")
    if p_val < 0.05:
        st.success("There is a statistically significant difference between DTD for 15/85 and 60/40 jar ratios.")
    else:
        st.info("There is no statistically significant difference between DTD for 15/85 and 60/40 jar ratios.")

with tab3:
    st.header("Confidence Analysis")
    
    # Decision confidence vs Schizotypy
    st.subheader("Decision Confidence vs Schizotypy")
    
    conf_measures = ["decision_confidence_15_85", "decision_confidence_60_40"]
    
    for conf_col in conf_measures:
        ratio = "15/85" if conf_col == "decision_confidence_15_85" else "60/40"
        st.write(f"### Decision Confidence for {ratio} Jar Ratio")
        
        # Create multiple plots in tabs
        conf_tabs = st.tabs(schizo_labels)
        
        for i, (tab, schizo_col, label) in enumerate(zip(conf_tabs, schizo_measures, schizo_labels)):
            with tab:
                # Calculate correlation
                corr, p_value = pearsonr(merged_data[conf_col], merged_data[schizo_col])
                
                fig = px.scatter(merged_data, x=schizo_col, y=conf_col,
                               hover_data=["participant_id"],
                               labels={conf_col: f"Decision Confidence ({ratio})", schizo_col: label},
                               trendline="ols")
                
                fig.update_layout(
                    title=f"Decision Confidence ({ratio}) vs {label} (r={corr:.2f}, p={p_value:.3f})",
                    height=500
                )
                
                st.plotly_chart(fig)
                
                # Display statistical significance
                alpha = 0.05
                if p_value < alpha:
                    st.success(f"Statistically significant correlation (p={p_value:.3f})")
                else:
                    st.info(f"No statistically significant correlation (p={p_value:.3f})")
    
    # Confidence vs DTD analysis
    st.subheader("Decision Confidence vs Draws to Decision")
    
    # 15/85 ratio
    fig1 = px.scatter(merged_data, x="dtd_15_85", y="decision_confidence_15_85", 
                    hover_data=["participant_id", "olife_total"],
                    labels={"dtd_15_85": "DTD (15/85)", "decision_confidence_15_85": "Decision Confidence (15/85)"},
                    trendline="ols")
    
    corr1, p1 = pearsonr(merged_data["dtd_15_85"], merged_data["decision_confidence_15_85"])
    fig1.update_layout(title=f"DTD vs Decision Confidence (15/85) (r={corr1:.2f}, p={p1:.3f})", height=500)
    st.plotly_chart(fig1)
    
    # 60/40 ratio
    fig2 = px.scatter(merged_data, x="dtd_60_40", y="decision_confidence_60_40", 
                    hover_data=["participant_id", "olife_total"],
                    labels={"dtd_60_40": "DTD (60/40)", "decision_confidence_60_40": "Decision Confidence (60/40)"},
                    trendline="ols")
    
    corr2, p2 = pearsonr(merged_data["dtd_60_40"], merged_data["decision_confidence_60_40"])
    fig2.update_layout(title=f"DTD vs Decision Confidence (60/40) (r={corr2:.2f}, p={p2:.3f})", height=500)
    st.plotly_chart(fig2)

with tab4:
    st.header("Statistical Analysis")
    
    # Create a dataframe to store all correlation results
    correlations = []
    
    # DTD correlations with schizotypy measures
    for dtd_col in ["dtd_15_85", "dtd_60_40"]:
        for schizo_col in ["olife_total", "ss1", "ss2", "ss3", "ss4"]:
            corr, p_value = pearsonr(merged_data[dtd_col], merged_data[schizo_col])
            dtd_label = "DTD (15/85)" if dtd_col == "dtd_15_85" else "DTD (60/40)"
            schizo_label = {"olife_total": "Total Score", "ss1": "Unusual Experiences", 
                            "ss2": "Cognitive Disorganisation", "ss3": "Introvertive Anhedonia", 
                            "ss4": "Impulsive Nonconformity"}[schizo_col]
            
            correlations.append({
                "Metric 1": dtd_label,
                "Metric 2": schizo_label,
                "Correlation": corr,
                "p-value": p_value,
                "Significance": "Significant" if p_value < 0.05 else "Not Significant"
            })
    
    # Confidence correlations with schizotypy measures
    for conf_col in ["decision_confidence_15_85", "decision_confidence_60_40"]:
        for schizo_col in ["olife_total", "ss1", "ss2", "ss3", "ss4"]:
            corr, p_value = pearsonr(merged_data[conf_col], merged_data[schizo_col])
            conf_label = "Confidence (15/85)" if conf_col == "decision_confidence_15_85" else "Confidence (60/40)"
            schizo_label = {"olife_total": "Total Score", "ss1": "Unusual Experiences", 
                            "ss2": "Cognitive Disorganisation", "ss3": "Introvertive Anhedonia", 
                            "ss4": "Impulsive Nonconformity"}[schizo_col]
            
            correlations.append({
                "Metric 1": conf_label,
                "Metric 2": schizo_label,
                "Correlation": corr,
                "p-value": p_value,
                "Significance": "Significant" if p_value < 0.05 else "Not Significant"
            })
    
    # Convert to dataframe and sort by significance and correlation strength
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values(by=["Significance", "Correlation"], ascending=[True, False])
    
    # Display correlation table
    st.subheader("Correlation Analysis Summary")
    st.dataframe(corr_df.style.format({
        "Correlation": "{:.3f}",
        "p-value": "{:.3f}"
    }).background_gradient(subset=["Correlation"], cmap="coolwarm"))
    
    # Highlight most significant findings
    significant_corrs = corr_df[corr_df["Significance"] == "Significant"]
    
    if not significant_corrs.empty:
        st.subheader("Significant Correlations")
        
        for _, row in significant_corrs.iterrows():
            st.write(f"**{row['Metric 1']} vs {row['Metric 2']}**: r = {row['Correlation']:.3f}, p = {row['p-value']:.3f}")
        
        # Visualize the significant correlations
        fig = px.bar(
            significant_corrs,
            x="Metric 2",
            y="Correlation",
            color="Metric 1",
            barmode="group",
            labels={"Metric 2": "Schizotypy Measure", "Correlation": "Pearson's r"},
            title="Significant Correlations"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig)
    else:
        st.info("No statistically significant correlations were found.")
    
    # Group comparison based on schizotypy scores
    st.subheader("Group Comparison")
    
    # Create high/low schizotypy groups for comparison
    median_total = merged_data["olife_total"].median()
    merged_data["schizotypy_group"] = merged_data["olife_total"].apply(lambda x: "High" if x > median_total else "Low")
    
    st.write(f"Comparing groups based on median schizotypy score (median = {median_total})")
    
    # Compare DTD between groups
    for dtd_col in ["dtd_15_85", "dtd_60_40"]:
        ratio = "15/85" if dtd_col == "dtd_15_85" else "60/40"
        
        high_group = merged_data[merged_data["schizotypy_group"] == "High"][dtd_col]
        low_group = merged_data[merged_data["schizotypy_group"] == "Low"][dtd_col]
        
        t_stat, p_val = stats.ttest_ind(high_group, low_group)
        
        fig = px.box(merged_data, x="schizotypy_group", y=dtd_col, 
                   labels={"schizotypy_group": "Schizotypy Group", dtd_col: f"DTD ({ratio})"},
                   title=f"DTD ({ratio}) Comparison by Schizotypy Group (p={p_val:.3f})")
        
        fig.update_layout(height=500)
        st.plotly_chart(fig)
        
        if p_val < 0.05:
            st.success(f"Significant difference in DTD ({ratio}) between high and low schizotypy groups (p={p_val:.3f})")
        else:
            st.info(f"No significant difference in DTD ({ratio}) between groups (p={p_val:.3f})")
    
    # Compare confidence between groups
    for conf_col in ["decision_confidence_15_85", "decision_confidence_60_40"]:
        ratio = "15/85" if conf_col == "decision_confidence_15_85" else "60/40"
        
        high_group = merged_data[merged_data["schizotypy_group"] == "High"][conf_col]
        low_group = merged_data[merged_data["schizotypy_group"] == "Low"][conf_col]
        
        t_stat, p_val = stats.ttest_ind(high_group, low_group)
        
        fig = px.box(merged_data, x="schizotypy_group", y=conf_col, 
                   labels={"schizotypy_group": "Schizotypy Group", conf_col: f"Confidence ({ratio})"},
                   title=f"Decision Confidence ({ratio}) Comparison by Schizotypy Group (p={p_val:.3f})")
        
        fig.update_layout(height=500)
        st.plotly_chart(fig)
        
        if p_val < 0.05:
            st.success(f"Significant difference in Decision Confidence ({ratio}) between high and low schizotypy groups (p={p_val:.3f})")
        else:
            st.info(f"No significant difference in Decision Confidence ({ratio}) between groups (p={p_val:.3f})")

with tab5:
    st.header("Clustering Analysis")
    
    # Select features for clustering
    st.subheader("K-Means Clustering")
    
    feature_options = {
        'DTD Metrics': ['dtd_15_85', 'dtd_60_40'],
        'Confidence Metrics': ['decision_confidence_15_85', 'decision_confidence_60_40'],
        'Schizotypy Metrics': ['olife_total', 'ss1', 'ss2', 'ss3', 'ss4']
    }
    
    # Let user select feature categories
    selected_categories = st.multiselect(
        "Select feature categories for clustering:",
        options=list(feature_options.keys()),
        default=['DTD Metrics', 'Schizotypy Metrics']
    )
    
    # Get all selected features
    selected_features = []
    for category in selected_categories:
        selected_features.extend(feature_options[category])
    
    if not selected_features:
        st.warning("Please select at least one feature category for clustering.")
    else:
        # Create two columns for elbow chart and slider
        col1, col2 = st.columns([1, 1])
        
        # Prepare data for clustering
        X = merged_data[selected_features].copy()
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        with col1:
            # Add elbow chart
            st.subheader("Elbow Chart for Optimal Clusters")
            
            # Calculate inertia for different numbers of clusters (1-10)
            inertia = []
            k_range = range(1, 11)
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, k in enumerate(k_range):
                status_text.text(f"Calculating for {k} clusters...")
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
                progress_bar.progress((i + 1) / len(k_range))
            
            status_text.text("Elbow chart calculation complete!")
            
            # Calculate the rate of change and find the elbow point
            inertia_diff = np.diff(inertia)
            inertia_diff2 = np.diff(inertia_diff)
            suggested_clusters = np.argmax(inertia_diff2) + 2  # +2 because we started with k=1 and took differences twice
            
            # Create the elbow chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(k_range),
                y=inertia,
                mode='lines+markers',
                name='Inertia',
                marker=dict(size=10)
            ))
            
            # Add a vertical line at the suggested optimal number of clusters
            fig.add_vline(x=suggested_clusters, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title=f"Elbow Method (Suggested optimal clusters: {suggested_clusters})",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="Inertia (Within-Cluster Sum of Squares)",
                height=400
            )
            
            st.plotly_chart(fig)
            
            st.info(f"**Suggested optimal number of clusters: {suggested_clusters}**  \n"
                   f"This is based on finding the 'elbow' point in the inertia curve - the point where adding more clusters "
                   f"doesn't significantly reduce the within-cluster sum of squares. "
                   f"However, the final decision should also consider your research question and interpretability of results.")
        
        with col2:
            # Select number of clusters
            n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=suggested_clusters)
            
            # Apply K-means clustering with the selected number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to data
            merged_data_with_clusters = merged_data.copy()
            merged_data_with_clusters['Cluster'] = clusters
            
            # Store the clustered data in session state for other tabs to use
            st.session_state['merged_data_with_clusters'] = merged_data_with_clusters
            st.session_state['selected_features'] = selected_features
            st.session_state['n_clusters'] = n_clusters
            st.session_state['kmeans'] = kmeans
            st.session_state['X_scaled'] = X_scaled
            
            # Silhouette score to evaluate clustering quality
            from sklearn.metrics import silhouette_score
            
            silhouette_avg = silhouette_score(X_scaled, clusters)
            st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            
            if silhouette_avg < 0.2:
                st.warning("Low silhouette score indicates poor cluster separation.")
            elif silhouette_avg >= 0.5:
                st.success("High silhouette score indicates well-separated clusters.")
            else:
                st.info("Moderate silhouette score indicates reasonable cluster separation.")
        
        # Display cluster characteristics
        st.subheader("Cluster Characteristics")
        
        # Show cluster means
        cluster_means = merged_data_with_clusters.groupby('Cluster')[selected_features].mean()
        st.dataframe(cluster_means.style.background_gradient(cmap='viridis'))
        
        # Display cluster sizes
        cluster_counts = merged_data_with_clusters['Cluster'].value_counts().sort_index()
        
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Count'},
            title="Number of Participants in Each Cluster"
        )
        st.plotly_chart(fig)
        
        # Visualize clusters
        st.subheader("Cluster Visualization")
        
        # If we have 4+ features, use PCA for visualization
        if len(selected_features) > 3:
            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create a DataFrame for plotting
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clusters
            pca_df['participant_id'] = merged_data['participant_id']
            
            # Add key metrics for hover data
            for feature in ['olife_total', 'dtd_15_85', 'dtd_60_40']:
                if feature in merged_data.columns:
                    pca_df[feature] = merged_data[feature].values
            
            # Create PCA scatter plot
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                hover_data=['participant_id'] + [col for col in ['olife_total', 'dtd_15_85', 'dtd_60_40'] if col in pca_df.columns],
                title=f"PCA Visualization of Clusters (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})"
            )
            st.plotly_chart(fig)
            
            # Explained variance
            st.write(f"PC1 explained variance: {pca.explained_variance_ratio_[0]:.2%}")
            st.write(f"PC2 explained variance: {pca.explained_variance_ratio_[1]:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(2)],
                index=selected_features
            )
            st.subheader("PCA Feature Importance")
            st.dataframe(feature_importance.style.background_gradient(cmap='coolwarm'))
        
        # If we have exactly 2 features, create a direct scatter plot
        elif len(selected_features) == 2:
            fig = px.scatter(
                merged_data_with_clusters,
                x=selected_features[0],
                y=selected_features[1],
                color='Cluster',
                hover_data=['participant_id', 'olife_total'],
                title=f"Clusters based on {selected_features[0]} and {selected_features[1]}"
            )
            st.plotly_chart(fig)
        
        # For 3 features, create a 3D scatter plot
        elif len(selected_features) == 3:
            fig = px.scatter_3d(
                merged_data_with_clusters,
                x=selected_features[0],
                y=selected_features[1],
                z=selected_features[2],
                color='Cluster',
                hover_data=['participant_id', 'olife_total'],
                title=f"3D Clusters based on selected features"
            )
            st.plotly_chart(fig)
        
        # Cross-tabulation with schizotypy groups
        if 'olife_total' in merged_data.columns:
            st.subheader("Clusters and Schizotypy Groups")
            
            # Define high/low schizotypy groups based on median
            median_schizo = merged_data['olife_total'].median()
            merged_data_with_clusters['Schizotypy_Group'] = merged_data['olife_total'].apply(
                lambda x: 'High' if x > median_schizo else 'Low'
            )
            
            # Create cross-tabulation
            cross_tab = pd.crosstab(
                merged_data_with_clusters['Cluster'], 
                merged_data_with_clusters['Schizotypy_Group']
            )
            
            st.write("Cross-tabulation of Clusters and Schizotypy Groups:")
            st.dataframe(cross_tab)
            
            # Visualize cross-tabulation
            fig = px.bar(
                cross_tab.reset_index().melt(id_vars='Cluster', var_name='Schizotypy_Group', value_name='Count'),
                x='Cluster',
                y='Count',
                color='Schizotypy_Group',
                barmode='group',
                title="Distribution of Schizotypy Groups within Each Cluster"
            )
            st.plotly_chart(fig)
            
            # Chi-square test for independence
            chi2, p, dof, expected = stats.chi2_contingency(cross_tab)
            st.write(f"Chi-square test for independence: χ² = {chi2:.2f}, p = {p:.4f}")
            if p < 0.05:
                st.success("There is a significant association between clusters and schizotypy groups.")
            else:
                st.info("There is no significant association between clusters and schizotypy groups.")

with tab6:
    st.header("Cluster Insights")
    
    if 'merged_data_with_clusters' not in st.session_state:
        st.warning("Please run the clustering analysis first.")
    else:
        merged_data_with_clusters = st.session_state['merged_data_with_clusters']
        selected_features = st.session_state['selected_features']
        n_clusters = st.session_state['n_clusters']
        
        # Define high/low schizotypy groups based on median if not already defined
        if 'Schizotypy_Group' not in merged_data_with_clusters.columns:
            median_schizo = merged_data['olife_total'].median()
            merged_data_with_clusters['Schizotypy_Group'] = merged_data['olife_total'].apply(
                lambda x: 'High' if x > median_schizo else 'Low'
            )
        
        # Get cross-tabulation
        cross_tab = pd.crosstab(
            merged_data_with_clusters['Cluster'], 
            merged_data_with_clusters['Schizotypy_Group']
        )
        
        # Select cluster to analyze
        selected_cluster = st.selectbox(
            "Select cluster to analyze:",
            options=sorted(merged_data_with_clusters['Cluster'].unique()),
            index=0
        )
        
        st.subheader(f"Analysis of Cluster {selected_cluster}")
        
        # Get data for the selected cluster
        cluster_data = merged_data_with_clusters[merged_data_with_clusters['Cluster'] == selected_cluster]
        
        # Basic statistics for the cluster
        st.write(f"**Number of participants in this cluster:** {len(cluster_data)}")
        
        # Distribution of schizotypy groups in this cluster
        cluster_crosstab = cross_tab.loc[selected_cluster]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Schizotypy Group Distribution:**")
            st.write(f"- High Schizotypy: {cluster_crosstab['High']} participants")
            st.write(f"- Low Schizotypy: {cluster_crosstab['Low']} participants")
            
            # Calculate the percentage of high/low in this cluster
            total_in_cluster = cluster_crosstab.sum()
            high_percent = (cluster_crosstab['High'] / total_in_cluster) * 100
            low_percent = (cluster_crosstab['Low'] / total_in_cluster) * 100
            
            st.write(f"- High Schizotypy: {high_percent:.1f}%")
            st.write(f"- Low Schizotypy: {low_percent:.1f}%")
        
        with col2:
            # Create a pie chart
            labels = ['High Schizotypy', 'Low Schizotypy']
            values = [cluster_crosstab['High'], cluster_crosstab['Low']]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3
            )])
            fig.update_layout(title=f"Schizotypy Group Distribution in Cluster {selected_cluster}")
            st.plotly_chart(fig)
        
        # Feature analysis for this cluster
        st.subheader(f"Feature Analysis for Cluster {selected_cluster}")
        
        # Calculate mean, standard deviation and z-score for each feature
        feature_analysis = []
        
        for feature in selected_features:
            cluster_mean = cluster_data[feature].mean()
            population_mean = merged_data[feature].mean()
            population_std = merged_data[feature].std()
            z_score = (cluster_mean - population_mean) / population_std if population_std != 0 else 0
            
            feature_analysis.append({
                'Feature': feature,
                'Cluster Mean': cluster_mean,
                'Population Mean': population_mean,
                'Difference': cluster_mean - population_mean,
                'Z-score': z_score
            })
        
        feature_df = pd.DataFrame(feature_analysis)
        
        # Display feature analysis table
        st.dataframe(feature_df.style.format({
            'Cluster Mean': '{:.2f}',
            'Population Mean': '{:.2f}',
            'Difference': '{:.2f}',
            'Z-score': '{:.2f}'
        }).background_gradient(subset=['Z-score'], cmap='RdBu_r'))
        
        # Create a bar chart to visualize the z-scores
        fig = px.bar(
            feature_df,
            x='Feature',
            y='Z-score',
            labels={'Z-score': 'Standard deviations from population mean'},
            title=f"Feature Z-scores for Cluster {selected_cluster}",
            color='Z-score',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig)
        
        # Provide interpretation
        st.subheader("Cluster Interpretation")
        
        # DTD characteristics
        dtd_features = [f for f in selected_features if 'dtd' in f.lower()]
        if dtd_features:
            dtd_zscores = feature_df[feature_df['Feature'].isin(dtd_features)]['Z-score'].values
            avg_dtd_zscore = np.mean(dtd_zscores)
            
            if avg_dtd_zscore < -0.5:
                st.write("**DTD Tendency:** This cluster tends to make decisions with **fewer draws** than average (jumping to conclusions).")
            elif avg_dtd_zscore > 0.5:
                st.write("**DTD Tendency:** This cluster tends to gather **more evidence** before making decisions compared to average.")
            else:
                st.write("**DTD Tendency:** This cluster shows **typical evidence gathering** behavior.")
        
        # Schizotypy characteristics
        schizo_features = [f for f in selected_features if f in ['olife_total', 'ss1', 'ss2', 'ss3', 'ss4']]
        if schizo_features:
            schizo_rows = feature_df[feature_df['Feature'].isin(schizo_features)]
            
            # Overall schizotypy level
            if 'olife_total' in schizo_rows['Feature'].values:
                olife_zscore = schizo_rows[schizo_rows['Feature'] == 'olife_total']['Z-score'].values[0]
                
                if olife_zscore < -0.5:
                    st.write("**Schizotypy Level:** This cluster shows **lower than average** schizotypal traits.")
                    
                    # If this is Cluster 0, add specific insights
                    if selected_cluster == 0:
                        st.success("**Cluster 0 Insight:** This cluster contains exclusively low schizotypy participants (0 high, 18 low). This suggests a distinctive cognitive profile that strongly separates these participants from those with high schizotypy scores.")
                        
                        # Get specific extreme features
                        extreme_features = feature_df[abs(feature_df['Z-score']) > 0.7].sort_values(by='Z-score', ascending=False)
                        if not extreme_features.empty:
                            st.write("**Key distinctive features of this cluster:**")
                            for _, row in extreme_features.iterrows():
                                direction = "higher" if row['Z-score'] > 0 else "lower"
                                st.write(f"- {row['Feature']}: {abs(row['Z-score']):.2f} SD {direction} than population average")
                        
                        # Check DTD relationship (if DTD features are selected)
                        if 'dtd_15_85' in selected_features or 'dtd_60_40' in selected_features:
                            dtd_15_85_zscore = feature_df[feature_df['Feature'] == 'dtd_15_85']['Z-score'].values[0] if 'dtd_15_85' in selected_features else 0
                            dtd_60_40_zscore = feature_df[feature_df['Feature'] == 'dtd_60_40']['Z-score'].values[0] if 'dtd_60_40' in selected_features else 0
                            
                            if dtd_15_85_zscore > 0 or dtd_60_40_zscore > 0:
                                st.write("This cluster demonstrates the expected inverse relationship between schizotypy and DTD: low schizotypy participants gathering more evidence (higher DTD).")
                            else:
                                st.write("Interestingly, despite having low schizotypy, this cluster doesn't show higher evidence gathering as might be expected.")
                
                elif olife_zscore > 0.5:
                    st.write("**Schizotypy Level:** This cluster shows **higher than average** schizotypal traits.")
                else:
                    st.write("**Schizotypy Level:** This cluster shows **typical** levels of schizotypal traits.")
            
            # Dominant schizotypy dimension
            if len(schizo_rows) > 1:  # If we have subscales
                subscales = schizo_rows[schizo_rows['Feature'].isin(['ss1', 'ss2', 'ss3', 'ss4'])]
                if not subscales.empty:
                    dominant_subscale = subscales.loc[subscales['Z-score'].abs().idxmax()]
                    
                    subscale_names = {
                        'ss1': "Unusual Experiences",
                        'ss2': "Cognitive Disorganisation",
                        'ss3': "Introvertive Anhedonia",
                        'ss4': "Impulsive Nonconformity"
                    }
                    
                    direction = "high" if dominant_subscale['Z-score'] > 0 else "low"
                    ss_name = subscale_names.get(dominant_subscale['Feature'], dominant_subscale['Feature'])
                    st.write(f"**Dominant Schizotypy Dimension:** This cluster is characterized by {direction} scores in **{ss_name}** (z-score: {dominant_subscale['Z-score']:.2f}).")
        
        # Compare this cluster with others
        st.subheader("Comparison to Other Clusters")
        
        # Get all cluster means
        all_cluster_means = merged_data_with_clusters.groupby('Cluster')[selected_features].mean()
        
        # Reshape for plotting
        cluster_comparison_df = all_cluster_means.reset_index().melt(
            id_vars=['Cluster'],
            value_vars=selected_features,
            var_name='Feature',
            value_name='Value'
        )
        
        # Highlight the selected cluster
        cluster_comparison_df['is_selected'] = cluster_comparison_df['Cluster'] == selected_cluster
        
        # Create comparison plot
        for feature in selected_features:
            feature_df = cluster_comparison_df[cluster_comparison_df['Feature'] == feature]
            
            fig = px.bar(
                feature_df,
                x='Cluster',
                y='Value',
                color='is_selected',
                color_discrete_map={True: 'royalblue', False: 'lightgray'},
                title=f"Comparison of {feature} Across Clusters"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig)
        
        # Statistical tests
        if len(cluster_data) >= 5:  # Only if we have enough data points
            st.subheader("Statistical Comparisons")
            
            for feature in selected_features:
                # Compare this cluster vs rest of population
                cluster_values = cluster_data[feature].values
                other_values = merged_data_with_clusters[merged_data_with_clusters['Cluster'] != selected_cluster][feature].values
                
                if len(cluster_values) > 0 and len(other_values) > 0:
                    # Run t-test
                    t_stat, p_val = stats.ttest_ind(cluster_values, other_values, equal_var=False)
                    
                    if p_val < 0.05:
                        st.success(f"**{feature}**: Significantly different from other participants (t={t_stat:.2f}, p={p_val:.4f})")
                    else:
                        st.info(f"**{feature}**: No significant difference from other participants (t={t_stat:.2f}, p={p_val:.4f})")

with tab7:
    st.header("Participant-level Analysis")
    
    # Select a participant
    participant_ids = merged_data["participant_id"].tolist()
    selected_participant = st.selectbox("Select a participant", participant_ids)
    
    # Filter data for the selected participant
    participant_data = merged_data[merged_data["participant_id"] == selected_participant]
    
    if not participant_data.empty:
        st.subheader(f"Report for Participant {selected_participant}")
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Schizotypy Score", participant_data["olife_total"].values[0])
            st.metric("SS1: Unusual Experiences", participant_data["ss1"].values[0])
            st.metric("SS2: Cognitive Disorganisation", participant_data["ss2"].values[0])
        
        with col2:
            st.metric("SS3: Introvertive Anhedonia", participant_data["ss3"].values[0])
            st.metric("SS4: Impulsive Nonconformity", participant_data["ss4"].values[0])
        
        with col3:
            st.metric("DTD (15/85)", f"{participant_data['dtd_15_85'].values[0]:.1f}")
            st.metric("DTD (60/40)", f"{participant_data['dtd_60_40'].values[0]:.1f}")
            st.metric("Practice DTD", f"{participant_data['practice_dtd'].values[0]:.1f}")
        
        # Confidence metrics
        st.subheader("Confidence Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Decision Confidence (15/85)", f"{participant_data['decision_confidence_15_85'].values[0]:.1f}")
            st.metric("Decision Confidence (60/40)", f"{participant_data['decision_confidence_60_40'].values[0]:.1f}")
        
        with col2:
            st.metric("Practice Decision Confidence", 
                    f"{participant_data['decision_confidence_practice'].values[0]:.1f}" 
                    if not pd.isna(participant_data['decision_confidence_practice'].values[0]) else "N/A")
        
        # Comparison to population averages
        st.subheader("Comparison to Population Averages")
        
        # Create comparison metrics
        comparison_metrics = [
            {"metric": "DTD (15/85)", "value": participant_data["dtd_15_85"].values[0], "avg": merged_data["dtd_15_85"].mean()},
            {"metric": "DTD (60/40)", "value": participant_data["dtd_60_40"].values[0], "avg": merged_data["dtd_60_40"].mean()},
            {"metric": "Decision Confidence (15/85)", "value": participant_data["decision_confidence_15_85"].values[0], 
             "avg": merged_data["decision_confidence_15_85"].mean()},
            {"metric": "Decision Confidence (60/40)", "value": participant_data["decision_confidence_60_40"].values[0], 
             "avg": merged_data["decision_confidence_60_40"].mean()},
            {"metric": "Total Schizotypy", "value": participant_data["olife_total"].values[0], "avg": merged_data["olife_total"].mean()}
        ]
        
        # Create a comparison chart
        fig = go.Figure()
        
        for item in comparison_metrics:
            fig.add_trace(go.Bar(
                x=[item["metric"]],
                y=[item["value"]],
                name="Participant",
                marker_color="royalblue"
            ))
            
            fig.add_trace(go.Bar(
                x=[item["metric"]],
                y=[item["avg"]],
                name="Population Average",
                marker_color="lightgrey"
            ))
        
        fig.update_layout(
            barmode="group",
            title="Participant vs Population Average",
            xaxis_title="Metric",
            yaxis_title="Value",
            legend_title="Legend",
            height=500
        )
        
        st.plotly_chart(fig)
        
        # Add interpretation
        st.subheader("Interpretation")
        
        # DTD interpretation
        dtd_15_85 = participant_data["dtd_15_85"].values[0]
        dtd_60_40 = participant_data["dtd_60_40"].values[0]
        avg_dtd_15_85 = merged_data["dtd_15_85"].mean()
        avg_dtd_60_40 = merged_data["dtd_60_40"].mean()
        
        if dtd_15_85 < avg_dtd_15_85 and dtd_60_40 < avg_dtd_60_40:
            st.write("This participant tends to make decisions more quickly (with fewer draws) than average, "
                   "which might indicate a tendency toward jumping to conclusions.")
        elif dtd_15_85 > avg_dtd_15_85 and dtd_60_40 > avg_dtd_60_40:
            st.write("This participant tends to gather more evidence before making decisions compared to the average, "
                   "indicating a more cautious decision-making approach.")
        else:
            st.write("This participant shows mixed decision-making patterns across different difficulty levels.")
        
        # Schizotypy interpretation
        olife_total = participant_data["olife_total"].values[0]
        avg_olife = merged_data["olife_total"].mean()
        
        if olife_total > avg_olife:
            st.write(f"The participant's total schizotypy score ({olife_total:.1f}) is higher than the average ({avg_olife:.1f}), "
                   "indicating higher schizotypal traits compared to the sample population.")
        else:
            st.write(f"The participant's total schizotypy score ({olife_total:.1f}) is lower than or equal to the average ({avg_olife:.1f}), "
                   "indicating typical or lower schizotypal traits compared to the sample population.")
        
        # Subscale interpretation
        highest_subscale = participant_data[["ss1", "ss2", "ss3", "ss4"]].idxmax(axis=1).values[0]
        highest_value = participant_data[highest_subscale].values[0]
        
        subscale_names = {
            "ss1": "Unusual Experiences",
            "ss2": "Cognitive Disorganisation",
            "ss3": "Introvertive Anhedonia",
            "ss4": "Impulsive Nonconformity"
        }
        
        st.write(f"The highest subscale score is {subscale_names[highest_subscale]} ({highest_value}), which represents "
               f"predominant characteristics in this domain.")
        
        # Improved relationship analysis between DTD and schizotypy
        st.subheader("Relationship with Expected Pattern")
        
        # Calculate pattern scores for both DTD tasks
        pattern_score_15_85 = 0
        pattern_score_60_40 = 0
        
        # Check 15/85 ratio
        if (olife_total > avg_olife and dtd_15_85 < avg_dtd_15_85) or (olife_total < avg_olife and dtd_15_85 > avg_dtd_15_85):
            pattern_score_15_85 = 1
            
        # Check 60/40 ratio
        if (olife_total > avg_olife and dtd_60_40 < avg_dtd_60_40) or (olife_total < avg_olife and dtd_60_40 > avg_dtd_60_40):
            pattern_score_60_40 = 1
            
        total_pattern_score = pattern_score_15_85 + pattern_score_60_40
        
        # Display visual indicator of pattern match
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pattern Match Score")
            
            # Create a gauge chart for pattern match
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = total_pattern_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pattern Match Score"},
                gauge = {
                    'axis': {'range': [0, 2], 'tickwidth': 1},
                    'bar': {'color': "royalblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 2], 'color': "lightblue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Interpretation")
            
            if total_pattern_score == 2:
                st.success("This participant **strongly follows** the expected inverse relationship between schizotypy and DTD across both task difficulties.")
                st.write("This aligns with research suggesting that higher schizotypy traits are associated with quicker decisions with less evidence (jumping to conclusions bias).")
            elif total_pattern_score == 1:
                st.info("This participant **partially follows** the expected pattern in one of the task difficulties, but not in both.")
                if pattern_score_15_85 == 1:
                    st.write("The pattern is observed in the 15/85 ratio task (more distinct jars) but not in the 60/40 task.")
                else:
                    st.write("The pattern is observed in the 60/40 ratio task (more ambiguous jars) but not in the 15/85 task.")
            else:
                st.warning("This participant **does not follow** the expected inverse relationship between schizotypy and DTD in either task.")
                st.write("This suggests that for this individual, schizotypal traits may not be predictive of evidence gathering behavior in the way typically observed in research.")
            
            # Calculate z-scores for more detailed analysis
            z_schizotypy = (olife_total - avg_olife) / merged_data["olife_total"].std()
            z_dtd_15_85 = (dtd_15_85 - avg_dtd_15_85) / merged_data["dtd_15_85"].std()
            z_dtd_60_40 = (dtd_60_40 - avg_dtd_60_40) / merged_data["dtd_60_40"].std()
            
            st.write(f"**Standardized scores (z-scores):**")
            st.write(f"- Schizotypy: {z_schizotypy:.2f} SD from mean")
            st.write(f"- DTD (15/85): {z_dtd_15_85:.2f} SD from mean")
            st.write(f"- DTD (60/40): {z_dtd_60_40:.2f} SD from mean")
            
            if abs(z_schizotypy) < 0.5:
                st.write("Note: This participant's schizotypy score is close to the average, which may make pattern detection less reliable.")

# Run the app
if __name__ == "__main__":
    st.sidebar.title("About")
    st.sidebar.info(
        "This dashboard analyzes the relationship between performance on the beads drawing task "
        "and schizotypy scores. The beads task assesses decision-making under uncertainty, while "
        "schizotypy measures assess traits associated with the schizophrenia spectrum."
    )
    
    st.sidebar.title("References")
    st.sidebar.markdown("""
    - The beads drawing task is a measure of the jumping-to-conclusions bias
    - Schizotypy is measured using the O-LIFE questionnaire with four subscales:
      - Unusual Experiences (SS1)
      - Cognitive Disorganisation (SS2)
      - Introvertive Anhedonia (SS3)
      - Impulsive Nonconformity (SS4)
    """)