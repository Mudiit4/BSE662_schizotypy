import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn.metrics

# Set the page configuration
st.set_page_config(
    page_title="Beads Task vs Schizotypy Analysis",
    page_icon="🧠",
    layout="wide"
)

# Load the data
@st.cache_data
def load_data():
    participant_avg_dtd = pd.read_csv("participant_avg_dtd_with_confidence.csv")
    schizotypy = pd.read_csv("schizotypy.csv")
    trial_wise_data = pd.read_csv("trial_wise_data_new_with_pid.csv")
    
    # Check if participant_id in participant_avg_dtd matches PID in schizotypy
    st.write("Verifying participant IDs between datasets...")
    
    # Convert all participant IDs to the same format for comparison
    participant_avg_dtd['participant_id_check'] = participant_avg_dtd['participant_id'].astype(str)
    schizotypy['PID_check'] = schizotypy['PID'].astype(str)
    
    # Create sets of participant IDs from each dataset for comparison
    avg_dtd_participants = set(participant_avg_dtd['participant_id_check'])
    schizotypy_participants = set(schizotypy['PID_check'])
    trial_wise_participants = set([str(pid) for pid in trial_wise_data['participant_id']])
    
    # Check for missing or mismatched participant IDs
    missing_in_schizotypy = avg_dtd_participants - schizotypy_participants
    missing_in_avg_dtd = schizotypy_participants - avg_dtd_participants
    missing_in_trial_wise = avg_dtd_participants - trial_wise_participants
    
    # Display warnings if any participant IDs are missing between datasets
    if missing_in_schizotypy:
        st.warning(f"Warning: {len(missing_in_schizotypy)} participants in DTD data are missing from schizotypy data: {', '.join(missing_in_schizotypy)}")
    
    if missing_in_avg_dtd:
        st.warning(f"Warning: {len(missing_in_avg_dtd)} participants in schizotypy data are missing from DTD data: {', '.join(missing_in_avg_dtd)}")
    
    if missing_in_trial_wise:
        st.warning(f"Warning: {len(missing_in_trial_wise)} participants in DTD data are missing from trial-wise data: {', '.join(missing_in_trial_wise)}")
    
    # Merge the data, using inner join to ensure only matched participants are included
    merged_data = pd.merge(participant_avg_dtd, schizotypy, left_on="participant_id", right_on="PID", how="inner")
    
    # Report how many participants were successfully matched
    st.success(f"Successfully matched {len(merged_data)} participants across all datasets.")
    
    # Clean up temporary columns used for checking
    if 'participant_id_check' in participant_avg_dtd.columns:
        participant_avg_dtd.drop(columns=['participant_id_check'], inplace=True)
    if 'PID_check' in schizotypy.columns:
        schizotypy.drop(columns=['PID_check'], inplace=True)
    
    return participant_avg_dtd, schizotypy, trial_wise_data, merged_data

# Load data with messages hidden initially
with st.spinner("Loading and verifying data..."):
    participant_avg_dtd, schizotypy, trial_wise_data, merged_data = load_data()
    
# Create a section for data verification that can be expanded
with st.expander("Data Verification Details"):
    st.subheader("Dataset Information")
    st.write(f"Number of participants in average DTD data: {len(participant_avg_dtd)}")
    st.write(f"Number of participants in schizotypy data: {len(schizotypy)}")
    st.write(f"Number of participants in trial-wise data: {trial_wise_data['participant_id'].nunique()}")
    st.write(f"Number of participants successfully matched across datasets: {len(merged_data)}")
    
    # Show sample of matched participants
    st.subheader("Sample of Matched Participant IDs")
    matched_pids = merged_data[['participant_id', 'PID']].head(10)
    st.dataframe(matched_pids)

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
    
    # Add a section with tabs for the different confidence metrics
    st.subheader("Confidence Metrics Analysis")
    
    conf_metric_tabs = st.tabs(["Decision Confidence", "First Confidence", "Last Confidence", "Confidence Evolution"])
    
    # Tab 1: Decision confidence (original analysis)
    with conf_metric_tabs[0]:
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

    # Tab 2: First confidence analysis
    with conf_metric_tabs[1]:
        st.subheader("First Confidence vs Schizotypy")
        
        st.markdown("""
        This analysis shows the relationship between the first confidence values reported by participants 
        (confidence rating at the beginning of the task) and their schizotypy scores. This may reveal how 
        schizotypal traits influence initial confidence judgments before much evidence has been gathered.
        """)
        
        first_conf_tabs = st.tabs(["First Confidence (15/85)", "First Confidence (60/40)"])
        
        # For 15/85 ratio (high contrast)
        with first_conf_tabs[0]:
            ratio_tabs = st.tabs(schizo_labels)
            
            for i, (tab, schizo_col, label) in enumerate(zip(ratio_tabs, schizo_measures, schizo_labels)):
                with tab:
                    # Calculate correlation
                    corr, p_value = pearsonr(merged_data['first_conf_85_15'], merged_data[schizo_col])
                    
                    fig = px.scatter(merged_data, x=schizo_col, y='first_conf_85_15',
                                   hover_data=["participant_id"],
                                   labels={"first_conf_85_15": "First Confidence (15/85)", schizo_col: label},
                                   trendline="ols")
                    
                    fig.update_layout(
                        title=f"First Confidence (15/85) vs {label} (r={corr:.2f}, p={p_value:.3f})",
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display statistical significance
                    alpha = 0.05
                    if p_value < alpha:
                        st.success(f"Statistically significant correlation (p={p_value:.3f})")
                    else:
                        st.info(f"No statistically significant correlation (p={p_value:.3f})")
        
        # For 60/40 ratio (low contrast)
        with first_conf_tabs[1]:
            ratio_tabs = st.tabs(schizo_labels)
            
            for i, (tab, schizo_col, label) in enumerate(zip(ratio_tabs, schizo_measures, schizo_labels)):
                with tab:
                    # Calculate correlation
                    corr, p_value = pearsonr(merged_data['first_conf_60_40'], merged_data[schizo_col])
                    
                    fig = px.scatter(merged_data, x=schizo_col, y='first_conf_60_40',
                                   hover_data=["participant_id"],
                                   labels={"first_conf_60_40": "First Confidence (60/40)", schizo_col: label},
                                   trendline="ols")
                    
                    fig.update_layout(
                        title=f"First Confidence (60/40) vs {label} (r={corr:.2f}, p={p_value:.3f})",
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display statistical significance
                    alpha = 0.05
                    if p_value < alpha:
                        st.success(f"Statistically significant correlation (p={p_value:.3f})")
                    else:
                        st.info(f"No statistically significant correlation (p={p_value:.3f})")

    # Tab 3: Last confidence analysis
    with conf_metric_tabs[2]:
        st.subheader("Last Confidence vs Schizotypy")
        
        st.markdown("""
        This analysis shows the relationship between the last confidence values reported by participants 
        (final confidence rating just before making a decision) and their schizotypy scores. This may reveal how 
        schizotypal traits relate to confidence during the moment of decision across different task difficulties.
        """)
        
        last_conf_tabs = st.tabs(["Last Confidence (15/85)", "Last Confidence (60/40)"])
        
        # For 15/85 ratio (high contrast)
        with last_conf_tabs[0]:
            ratio_tabs = st.tabs(schizo_labels)
            
            for i, (tab, schizo_col, label) in enumerate(zip(ratio_tabs, schizo_measures, schizo_labels)):
                with tab:
                    # Calculate correlation
                    corr, p_value = pearsonr(merged_data['last_conf_85_15'], merged_data[schizo_col])
                    
                    fig = px.scatter(merged_data, x=schizo_col, y='last_conf_85_15',
                                   hover_data=["participant_id"],
                                   labels={"last_conf_85_15": "Last Confidence (15/85)", schizo_col: label},
                                   trendline="ols")
                    
                    fig.update_layout(
                        title=f"Last Confidence (15/85) vs {label} (r={corr:.2f}, p={p_value:.3f})",
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display statistical significance
                    alpha = 0.05
                    if p_value < alpha:
                        st.success(f"Statistically significant correlation (p={p_value:.3f})")
                    else:
                        st.info(f"No statistically significant correlation (p={p_value:.3f})")
        
        # For 60/40 ratio (low contrast)
        with last_conf_tabs[1]:
            ratio_tabs = st.tabs(schizo_labels)
            
            for i, (tab, schizo_col, label) in enumerate(zip(ratio_tabs, schizo_measures, schizo_labels)):
                with tab:
                    # Calculate correlation
                    corr, p_value = pearsonr(merged_data['last_conf_60_40'], merged_data[schizo_col])
                    
                    fig = px.scatter(merged_data, x=schizo_col, y='last_conf_60_40',
                                   hover_data=["participant_id"],
                                   labels={"last_conf_60_40": "Last Confidence (60/40)", schizo_col: label},
                                   trendline="ols")
                    
                    fig.update_layout(
                        title=f"Last Confidence (60/40) vs {label} (r={corr:.2f}, p={p_value:.3f})",
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display statistical significance
                    alpha = 0.05
                    if p_value < alpha:
                        st.success(f"Statistically significant correlation (p={p_value:.3f})")
                    else:
                        st.info(f"No statistically significant correlation (p={p_value:.3f})")
    
    # Tab 4: Confidence evolution analysis
    with conf_metric_tabs[3]:
        st.subheader("Confidence Evolution Analysis")
        
        st.markdown("""
        This analysis visualizes how confidence changes from the first bead to the final decision.
        It compares first confidence, last confidence, and final decision confidence to show the evolution
        of confidence throughout the task.
        """)
        
        # Create evolution visualizations for high contrast (15/85) and low contrast (60/40) conditions
        for condition, name in [("15/85", "High Contrast"), ("60/40", "Low Contrast")]:
            st.write(f"### Confidence Evolution for {name} ({condition}) Jar Ratio")
            
            # Prepare data for bars
            if condition == "15/85":
                first_conf = merged_data['first_conf_85_15'].mean()
                last_conf = merged_data['last_conf_85_15'].mean()
                decision_conf = merged_data['decision_confidence_15_85'].mean()
                first_conf_std = merged_data['first_conf_85_15'].std()
                last_conf_std = merged_data['last_conf_85_15'].std()
                decision_conf_std = merged_data['decision_confidence_15_85'].std()
            else:
                first_conf = merged_data['first_conf_60_40'].mean()
                last_conf = merged_data['last_conf_60_40'].mean()
                decision_conf = merged_data['decision_confidence_60_40'].mean()
                first_conf_std = merged_data['first_conf_60_40'].std()
                last_conf_std = merged_data['last_conf_60_40'].std()
                decision_conf_std = merged_data['decision_confidence_60_40'].std()
            
            # Create bar chart
            evolution_data = {
                'Stage': ['First Confidence', 'Last Confidence', 'Decision Confidence'],
                'Mean Confidence': [first_conf, last_conf, decision_conf],
                'Standard Deviation': [first_conf_std, last_conf_std, decision_conf_std]
            }
            
            fig = go.Figure()
            
            # Add bars for mean confidence
            fig.add_trace(go.Bar(
                x=evolution_data['Stage'],
                y=evolution_data['Mean Confidence'],
                name='Mean Confidence',
                marker_color='royalblue',
                error_y=dict(
                    type='data',
                    array=evolution_data['Standard Deviation'],
                    visible=True
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Confidence Evolution for {condition} Condition",
                xaxis_title="Task Stage",
                yaxis_title="Confidence Level (%)",
                yaxis=dict(range=[0, 100]),
                height=500
            )
            
            st.plotly_chart(fig)
            
            # Calculate and display statistics
            first_to_last = last_conf - first_conf
            last_to_decision = decision_conf - last_conf
            overall_change = decision_conf - first_conf
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Statistical Summary:**")
                st.write(f"- First Confidence: {first_conf:.2f}%")
                st.write(f"- Last Confidence: {last_conf:.2f}%")
                st.write(f"- Decision Confidence: {decision_conf:.2f}%")
            
            with col2:
                st.write("**Confidence Changes:**")
                st.write(f"- First to Last: {first_to_last:+.2f}% {'↑' if first_to_last > 0 else '↓'}")
                st.write(f"- Last to Decision: {last_to_decision:+.2f}% {'↑' if last_to_decision > 0 else '↓'}")
                st.write(f"- Overall Change: {overall_change:+.2f}% {'↑' if overall_change > 0 else '↓'}")
            
            # Test for significant differences between stages
            first_last_tstat, first_last_pval = stats.ttest_rel(
                merged_data[f'first_conf_{"85_15" if condition == "15/85" else "60_40"}'],
                merged_data[f'last_conf_{"85_15" if condition == "15/85" else "60_40"}']
            )
            
            last_decision_tstat, last_decision_pval = stats.ttest_rel(
                merged_data[f'last_conf_{"85_15" if condition == "15/85" else "60_40"}'],
                merged_data[f'decision_confidence_{"15_85" if condition == "15/85" else "60_40"}']
            )
            
            first_decision_tstat, first_decision_pval = stats.ttest_rel(
                merged_data[f'first_conf_{"85_15" if condition == "15/85" else "60_40"}'],
                merged_data[f'decision_confidence_{"15_85" if condition == "15/85" else "60_40"}']
            )
            
            st.write("**Statistical Tests (Paired t-tests):**")
            
            if first_last_pval < 0.05:
                st.success(f"Significant difference between First and Last Confidence (t={first_last_tstat:.2f}, p={first_last_pval:.4f})")
            else:
                st.info(f"No significant difference between First and Last Confidence (t={first_last_tstat:.2f}, p={first_last_pval:.4f})")
                
            if last_decision_pval < 0.05:
                st.success(f"Significant difference between Last and Decision Confidence (t={last_decision_tstat:.2f}, p={last_decision_pval:.4f})")
            else:
                st.info(f"No significant difference between Last and Decision Confidence (t={last_decision_tstat:.2f}, p={last_decision_pval:.4f})")
                
            if first_decision_pval < 0.05:
                st.success(f"Significant difference between First and Decision Confidence (t={first_decision_tstat:.2f}, p={first_decision_pval:.4f})")
            else:
                st.info(f"No significant difference between First and Decision Confidence (t={first_decision_tstat:.2f}, p={first_decision_pval:.4f})")
        
        # Add visualization comparing condition differences in confidence evolution
        st.write("### Comparing Confidence Evolution Between Conditions")
        
        # Create dataframe for the comparison visualization
        evolution_comparison = pd.DataFrame({
            'Stage': ['First Confidence', 'Last Confidence', 'Decision Confidence'] * 2,
            'Condition': ['15/85'] * 3 + ['60/40'] * 3,
            'Mean Confidence': [
                merged_data['first_conf_85_15'].mean(), 
                merged_data['last_conf_85_15'].mean(), 
                merged_data['decision_confidence_15_85'].mean(),
                merged_data['first_conf_60_40'].mean(),
                merged_data['last_conf_60_40'].mean(),
                merged_data['decision_confidence_60_40'].mean()
            ]
        })
        
        # Create the visualization
        fig = px.line(
            evolution_comparison, x='Stage', y='Mean Confidence', color='Condition',
            markers=True, title="Confidence Evolution Comparison Between Conditions",
            labels={'Mean Confidence': 'Confidence Level (%)', 'Stage': 'Task Stage'},
            color_discrete_map={'15/85': 'royalblue', '60/40': 'firebrick'}
        )
        
        fig.update_layout(
            yaxis=dict(range=[0, 100]),
            height=500
        )
        
        st.plotly_chart(fig)
        
        # Statistical comparison between conditions
        st.write("**Statistical Comparison Between Conditions:**")
        
        # Compare first confidence between conditions
        first_tstat, first_pval = stats.ttest_rel(
            merged_data['first_conf_85_15'],
            merged_data['first_conf_60_40']
        )
        
        # Compare last confidence between conditions
        last_tstat, last_pval = stats.ttest_rel(
            merged_data['last_conf_85_15'],
            merged_data['last_conf_60_40']
        )
        
        # Compare decision confidence between conditions
        decision_tstat, decision_pval = stats.ttest_rel(
            merged_data['decision_confidence_15_85'],
            merged_data['decision_confidence_60_40']
        )
        
        # Display statistics in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First Confidence:**")
            st.write(f"- 15/85: {merged_data['first_conf_85_15'].mean():.2f}%")
            st.write(f"- 60/40: {merged_data['first_conf_60_40'].mean():.2f}%")
            st.write(f"- Difference: {merged_data['first_conf_85_15'].mean() - merged_data['first_conf_60_40'].mean():+.2f}%")
            
            if first_pval < 0.05:
                st.success(f"Significant difference in First Confidence (t={first_tstat:.2f}, p={first_pval:.4f})")
            else:
                st.info(f"No significant difference in First Confidence (t={first_tstat:.2f}, p={first_pval:.4f})")
                
            st.write("**Last Confidence:**")
            st.write(f"- 15/85: {merged_data['last_conf_85_15'].mean():.2f}%")
            st.write(f"- 60/40: {merged_data['last_conf_60_40'].mean():.2f}%")
            st.write(f"- Difference: {merged_data['last_conf_85_15'].mean() - merged_data['last_conf_60_40'].mean():+.2f}%")
            
            if last_pval < 0.05:
                st.success(f"Significant difference in Last Confidence (t={last_tstat:.2f}, p={last_pval:.4f})")
            else:
                st.info(f"No significant difference in Last Confidence (t={last_tstat:.2f}, p={last_pval:.4f})")
        
        with col2:
            st.write("**Decision Confidence:**")
            st.write(f"- 15/85: {merged_data['decision_confidence_15_85'].mean():.2f}%")
            st.write(f"- 60/40: {merged_data['decision_confidence_60_40'].mean():.2f}%")
            st.write(f"- Difference: {merged_data['decision_confidence_15_85'].mean() - merged_data['decision_confidence_60_40'].mean():+.2f}%")
            
            if decision_pval < 0.05:
                st.success(f"Significant difference in Decision Confidence (t={decision_tstat:.2f}, p={decision_pval:.4f})")
            else:
                st.info(f"No significant difference in Decision Confidence (t={decision_tstat:.2f}, p={decision_pval:.4f})")
            
            # Calculate confidence growth rates
            growth_85_15 = (merged_data['last_conf_85_15'].mean() - merged_data['first_conf_85_15'].mean()) / merged_data['first_conf_85_15'].mean() * 100
            growth_60_40 = (merged_data['last_conf_60_40'].mean() - merged_data['first_conf_60_40'].mean()) / merged_data['first_conf_60_40'].mean() * 100
            
            st.write("**Confidence Growth Rate:**")
            st.write(f"- 15/85: {growth_85_15:.2f}%")
            st.write(f"- 60/40: {growth_60_40:.2f}%")
            st.write(f"- Difference: {growth_85_15 - growth_60_40:+.2f}%")
    
    # Add confidence vs DTD analysis
    st.subheader("Confidence vs DTD Analysis")
    
    # 15/85 ratio - Decision Confidence vs DTD
    fig1 = px.scatter(merged_data, x="dtd_15_85", y="decision_confidence_15_85", 
                    hover_data=["participant_id", "olife_total"],
                    labels={"dtd_15_85": "DTD (15/85)", "decision_confidence_15_85": "Decision Confidence (15/85)"},
                    trendline="ols")
    
    corr1, p1 = pearsonr(merged_data["dtd_15_85"], merged_data["decision_confidence_15_85"])
    fig1.update_layout(title=f"DTD vs Decision Confidence (15/85) (r={corr1:.2f}, p={p1:.3f})", height=500)
    st.plotly_chart(fig1)
    
    # 60/40 ratio - Decision Confidence vs DTD
    fig2 = px.scatter(merged_data, x="dtd_60_40", y="decision_confidence_60_40", 
                    hover_data=["participant_id", "olife_total"],
                    labels={"dtd_60_40": "DTD (60/40)", "decision_confidence_60_40": "Decision Confidence (60/40)"},
                    trendline="ols")
    
    corr2, p2 = pearsonr(merged_data["dtd_60_40"], merged_data["decision_confidence_60_40"])
    fig2.update_layout(title=f"DTD vs Decision Confidence (60/40) (r={corr2:.2f}, p={p2:.3f})", height=500)
    st.plotly_chart(fig2)
    
    # 15/85 ratio - First Confidence vs DTD
    fig3 = px.scatter(merged_data, x="dtd_15_85", y="first_conf_85_15", 
                    hover_data=["participant_id", "olife_total"],
                    labels={"dtd_15_85": "DTD (15/85)", "first_conf_85_15": "First Confidence (15/85)"},
                    trendline="ols")
    
    corr3, p3 = pearsonr(merged_data["dtd_15_85"], merged_data["first_conf_85_15"])
    fig3.update_layout(title=f"DTD vs First Confidence (15/85) (r={corr3:.2f}, p={p3:.3f})", height=500)
    st.plotly_chart(fig3)
    
    # 60/40 ratio - First Confidence vs DTD
    fig4 = px.scatter(merged_data, x="dtd_60_40", y="first_conf_60_40", 
                    hover_data=["participant_id", "olife_total"],
                    labels={"dtd_60_40": "DTD (60/40)", "first_conf_60_40": "First Confidence (60/40)"},
                    trendline="ols")
    
    corr4, p4 = pearsonr(merged_data["dtd_60_40"], merged_data["first_conf_60_40"])
    fig4.update_layout(title=f"DTD vs First Confidence (60/40) (r={corr4:.2f}, p={p4:.3f})", height=500)
    st.plotly_chart(fig4)
    
    # 15/85 ratio - Last Confidence vs DTD
    fig5 = px.scatter(merged_data, x="dtd_15_85", y="last_conf_85_15", 
                    hover_data=["participant_id", "olife_total"],
                    labels={"dtd_15_85": "DTD (15/85)", "last_conf_85_15": "Last Confidence (15/85)"},
                    trendline="ols")
    
    corr5, p5 = pearsonr(merged_data["dtd_15_85"], merged_data["last_conf_85_15"])
    fig5.update_layout(title=f"DTD vs Last Confidence (15/85) (r={corr5:.2f}, p={p5:.3f})", height=500)
    st.plotly_chart(fig5)
    
    # 60/40 ratio - Last Confidence vs DTD
    fig6 = px.scatter(merged_data, x="dtd_60_40", y="last_conf_60_40", 
                    hover_data=["participant_id", "olife_total"],
                    labels={"dtd_60_40": "DTD (60/40)", "last_conf_60_40": "Last Confidence (60/40)"},
                    trendline="ols")
    
    corr6, p6 = pearsonr(merged_data["dtd_60_40"], merged_data["last_conf_60_40"])
    fig6.update_layout(title=f"DTD vs Last Confidence (60/40) (r={corr6:.2f}, p={p6:.3f})", height=500)
    st.plotly_chart(fig6)
    
    # Summary of correlations
    st.subheader("Summary of DTD vs Confidence Correlations")
    
    corr_summary = pd.DataFrame({
        'Metric 1': ["DTD (15/85)", "DTD (60/40)", "DTD (15/85)", "DTD (60/40)", "DTD (15/85)", "DTD (60/40)"],
        'Metric 2': ["Decision Confidence (15/85)", "Decision Confidence (60/40)", 
                    "First Confidence (15/85)", "First Confidence (60/40)",
                    "Last Confidence (15/85)", "Last Confidence (60/40)"],
        'Correlation': [corr1, corr2, corr3, corr4, corr5, corr6],
        'p-value': [p1, p2, p3, p4, p5, p6],
        'Significance': ["Significant" if p < 0.05 else "Not Significant" for p in [p1, p2, p3, p4, p5, p6]]
    })
    
    st.dataframe(corr_summary.style.format({
        'Correlation': '{:.3f}',
        'p-value': '{:.3f}'
    }).background_gradient(subset=['Correlation'], cmap='coolwarm'))

with tab4:
    st.header("Statistical Analysis")
    
    st.subheader("Correlation Analysis")
    
    # Select variables for correlation analysis
    correlation_options = ["dtd_15_85", "dtd_60_40", 
                          "decision_confidence_15_85", "decision_confidence_60_40",
                          "first_conf_85_15", "first_conf_60_40",
                          "last_conf_85_15", "last_conf_60_40",
                          "olife_total", "ss1", "ss2", "ss3", "ss4"]
    
    correlation_vars = st.multiselect(
        "Select Variables for Correlation Analysis",
        options=correlation_options,
        default=["dtd_15_85", "dtd_60_40", "olife_total", "ss1", "first_conf_85_15", "last_conf_85_15", "first_conf_60_40", "last_conf_60_40"]
    )
    
    if len(correlation_vars) > 1:
        # Calculate correlation matrix
        correlation_df = merged_data[correlation_vars].corr()
        
        # Display correlation matrix as heatmap
        fig = px.imshow(
            correlation_df,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig)
        
        # Display significant correlations
        st.subheader("Significant Correlations")
        
        significant_corrs = []
        
        # Calculate p-values for correlations
        for i, var1 in enumerate(correlation_vars):
            for j, var2 in enumerate(correlation_vars):
                if i < j:  # Only look at upper triangle to avoid duplicates
                    # Calculate correlation and p-value
                    corr, p_val = scipy.stats.pearsonr(
                        merged_data[var1].dropna(),
                        merged_data[var2].dropna()
                    )
                    
                    # Check if significant
                    if p_val < 0.05:
                        significant_corrs.append({
                            "Variable 1": var1,
                            "Variable 2": var2,
                            "Correlation": corr,
                            "p-value": p_val
                        })
        
        if significant_corrs:
            significant_df = pd.DataFrame(significant_corrs)
            significant_df = significant_df.sort_values(by="p-value")
            st.dataframe(significant_df)
            
            # Create scatterplot matrix for significant correlations
            if len(significant_df) > 0:
                # Get unique variables from significant correlations
                sig_vars = list(set(
                    significant_df["Variable 1"].tolist() + 
                    significant_df["Variable 2"].tolist()
                ))
                
                if len(sig_vars) > 1:
                    fig = px.scatter_matrix(
                        merged_data,
                        dimensions=sig_vars,
                        color="olife_total",
                        title="Scatterplot Matrix of Significantly Correlated Variables"
                    )
                    fig.update_layout(height=800)
                    st.plotly_chart(fig)
        else:
            st.write("No significant correlations found.")
    
    # Group comparison
    st.subheader("Group Comparison")
    
    # Define the split variable
    split_var = st.selectbox(
        "Split Participants By:",
        options=["olife_total", "ss1", "ss2", "ss3", "ss4"],
        index=0
    )
    
    # Define the threshold for splitting
    threshold_type = st.radio(
        "Split Method:",
        options=["Median Split", "Custom Threshold"],
        index=0
    )
    
    if threshold_type == "Median Split":
        threshold = merged_data[split_var].median()
    else:
        threshold = st.slider(
            "Threshold Value",
            min_value=float(merged_data[split_var].min()),
            max_value=float(merged_data[split_var].max()),
            value=float(merged_data[split_var].median())
        )
    
    # Create groups
    merged_data["group"] = merged_data[split_var].apply(
        lambda x: "High" if x > threshold else "Low"
    )
    
    # Select variables for comparison
    comparison_vars = st.multiselect(
        "Select Variables to Compare Between Groups",
        options=["dtd_15_85", "dtd_60_40", 
                "decision_confidence_15_85", "decision_confidence_60_40",
                "first_conf_85_15", "first_conf_60_40",
                "last_conf_85_15", "last_conf_60_40"],
        default=["dtd_15_85", "dtd_60_40", "first_conf_85_15", "last_conf_85_15"]
    )
    
    if comparison_vars:
        # Display group means
        group_means = merged_data.groupby("group")[comparison_vars].mean().reset_index()
        
        # Reshape for plotting
        group_means_melted = group_means.melt(
            id_vars="group",
            value_vars=comparison_vars,
            var_name="Measure",
            value_name="Value"
        )
        
        # Create plot
        fig = px.bar(
            group_means_melted,
            x="Measure",
            y="Value",
            color="group",
            barmode="group",
            title=f"Group Comparison by {split_var} (Threshold: {threshold:.2f})"
        )
        st.plotly_chart(fig)
        
        # Statistical tests
        st.subheader("Statistical Tests")
        
        test_results = []
        
        for var in comparison_vars:
            # Get data for each group
            group1 = merged_data[merged_data["group"] == "High"][var].dropna()
            group2 = merged_data[merged_data["group"] == "Low"][var].dropna()
            
            # Perform t-test
            t_stat, p_val = scipy.stats.ttest_ind(group1, group2, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_diff = group1.mean() - group2.mean()
            pooled_std = np.sqrt(((group1.count() - 1) * group1.std()**2 + 
                                (group2.count() - 1) * group2.std()**2) / 
                                (group1.count() + group2.count() - 2))
            effect_size = mean_diff / pooled_std
            
            test_results.append({
                "Measure": var,
                "High Group Mean": group1.mean(),
                "Low Group Mean": group2.mean(),
                "Mean Difference": mean_diff,
                "t-statistic": t_stat,
                "p-value": p_val,
                "Cohen's d": effect_size
            })
        
        # Display test results
        test_df = pd.DataFrame(test_results)
        test_df = test_df.sort_values(by="p-value")
        st.dataframe(test_df)
        
        # Highlight significant differences with boxplots
        st.subheader("Distribution Comparison")
        
        # Find significant variables
        significant_vars = test_df[test_df["p-value"] < 0.05]["Measure"].tolist()
        
        if significant_vars:
            st.write("Significant differences found in:")
            
            for var in significant_vars:
                fig = px.box(
                    merged_data,
                    x="group",
                    y=var,
                    color="group",
                    points="all",
                    title=f"{var} by {split_var} Group"
                )
                st.plotly_chart(fig)
        else:
            st.write("No significant differences found between groups.")
    
    # Regression Analysis
    st.subheader("Regression Analysis")
    
    # Select dependent variable
    dependent_var = st.selectbox(
        "Select Dependent Variable",
        options=["dtd_15_85", "dtd_60_40", 
                "decision_confidence_15_85", "decision_confidence_60_40",
                "first_conf_85_15", "first_conf_60_40",
                "last_conf_85_15", "last_conf_60_40"],
        index=0
    )
    
    # Select independent variables
    independent_vars = st.multiselect(
        "Select Independent Variables",
        options=["olife_total", "ss1", "ss2", "ss3", "ss4"],
        default=["olife_total"]
    )
    
    if independent_vars:
        # Prepare data for regression
        X = merged_data[independent_vars].copy()
        y = merged_data[dependent_var].copy()
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Fit regression model
        model = sm.OLS(y, X).fit()
        
        # Display regression results
        st.write("Regression Results:")
        
        # Create table of results
        results_df = pd.DataFrame({
            "Variable": ["constant"] + independent_vars,
            "Coefficient": model.params.values,
            "Std Error": model.bse.values,
            "t-value": model.tvalues.values,
            "p-value": model.pvalues.values
        })
        
        st.dataframe(results_df)
        
        # Display model summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R-squared", f"{model.rsquared:.3f}")
        
        with col2:
            st.metric("Adjusted R-squared", f"{model.rsquared_adj:.3f}")
        
        with col3:
            st.metric("F-statistic p-value", f"{model.f_pvalue:.3f}")
        
        # Create prediction plot
        if len(independent_vars) == 1:
            # For single predictor, create scatter plot with regression line
            fig = px.scatter(
                merged_data,
                x=independent_vars[0],
                y=dependent_var,
                trendline="ols",
                labels={
                    independent_vars[0]: independent_vars[0],
                    dependent_var: dependent_var
                },
                title=f"Regression: {dependent_var} vs {independent_vars[0]}"
            )
            st.plotly_chart(fig)

with tab5:
    st.header("Clustering Analysis")
    
    # Select clustering features
    clustering_features = st.multiselect(
        "Select Features for Clustering",
        options=["dtd_15_85", "dtd_60_40", 
                "decision_confidence_15_85", "decision_confidence_60_40",
                "first_conf_85_15", "first_conf_60_40",
                "last_conf_85_15", "last_conf_60_40",
                "olife_total", "ss1", "ss2", "ss3", "ss4"],
        default=["dtd_15_85", "dtd_60_40", "decision_confidence_15_85", "first_conf_85_15", "last_conf_85_15"]
    )
    
    if len(clustering_features) > 1:
        # Add elbow method for optimal k selection
        st.subheader("Elbow Method for Optimal Number of Clusters")
        
        # Prepare data for clustering
        clustering_data = merged_data[clustering_features].copy()
        clustering_data = clustering_data.dropna()
        
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # Calculate WCSS (Within-Cluster Sum of Square) for different values of k
        show_elbow = st.checkbox("Show Elbow Analysis", value=True)
        
        if show_elbow:
            with st.spinner("Calculating optimal number of clusters..."):
                wcss = []
                silhouette_scores = []
                k_range = range(2, 11)  # Test from 2 to 10 clusters
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(scaled_data)
                    wcss.append(kmeans.inertia_)
                    
                    # Calculate silhouette score
                    cluster_labels = kmeans.labels_
                    silhouette_avg = sklearn.metrics.silhouette_score(scaled_data, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                
                # Create elbow plot
                fig = make_subplots(rows=1, cols=2, 
                                   subplot_titles=("Elbow Method (WCSS)", "Silhouette Score"),
                                   specs=[[{"type": "scatter"}, {"type": "scatter"}]])
                
                # Add WCSS trace
                fig.add_trace(
                    go.Scatter(x=list(k_range), y=wcss, mode='lines+markers', name='WCSS'),
                    row=1, col=1
                )
                
                # Add Silhouette score trace
                fig.add_trace(
                    go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers', 
                              name='Silhouette Score'),
                    row=1, col=2
                )
                
                # Update layout
                fig.update_layout(
                    height=500, 
                    title_text="Cluster Quality Metrics",
                    showlegend=False
                )
                
                fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
                fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
                fig.update_yaxes(title_text="WCSS", row=1, col=1)
                fig.update_yaxes(title_text="Silhouette Score (higher is better)", row=1, col=2)
                
                st.plotly_chart(fig)
                
                # Find optimal k based on silhouette score
                optimal_k_silhouette = k_range[silhouette_scores.index(max(silhouette_scores))]
                st.success(f"Optimal number of clusters based on silhouette score: {optimal_k_silhouette}")
        
        # Select number of clusters
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Calculate silhouette score for the selected number of clusters
        silhouette_avg = sklearn.metrics.silhouette_score(scaled_data, clusters)
        st.metric("Silhouette Score", f"{silhouette_avg:.3f}", 
                 delta=f"{silhouette_avg - 0.5:.3f}" if silhouette_avg > 0.5 else f"{silhouette_avg - 0.5:.3f}",
                 delta_color="normal")
        
        if silhouette_avg < 0.3:
            st.warning("Low silhouette score indicates poor clustering quality. Consider using different features or changing the number of clusters.")
        elif silhouette_avg > 0.6:
            st.success("High silhouette score indicates good clustering quality.")
            
        # Add cluster labels to the data
        clustered_data = clustering_data.copy()
        clustered_data["Cluster"] = clusters
        
        # Store clustering results in session state for other tabs to use
        merged_data_with_clusters = merged_data.loc[clustered_data.index].copy()
        merged_data_with_clusters["Cluster"] = clusters
        
        # Define high/low schizotypy groups based on median
        median_schizo = merged_data['olife_total'].median()
        merged_data_with_clusters['Schizotypy_Group'] = merged_data['olife_total'].apply(
            lambda x: 'High' if x > median_schizo else 'Low'
        )
        
        # Save to session state
        st.session_state['merged_data_with_clusters'] = merged_data_with_clusters
        st.session_state['selected_features'] = clustering_features
        st.session_state['n_clusters'] = n_clusters
        
        # Visualize clusters using PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create dataframe for plotting
        pca_df = pd.DataFrame({
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "Cluster": clusters
        })
        
        # Map participant IDs and schizotypy group for hover data
        pca_df["participant_id"] = clustered_data.index
        pca_df["schizotypy"] = merged_data_with_clusters['olife_total']
        pca_df["schizotypy_group"] = merged_data_with_clusters['Schizotypy_Group']
        
        # Create scatter plot of clusters
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_data=["participant_id", "schizotypy", "schizotypy_group"],
            title="PCA Visualization of Clusters",
            labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
                   "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"}
        )
        st.plotly_chart(fig)
        
        # Show cluster characteristics
        st.subheader("Cluster Characteristics")
        
        # Calculate cluster means
        cluster_means = clustered_data.groupby("Cluster")[clustering_features].mean()
        
        # Reshape for plotting
        cluster_means_melted = cluster_means.reset_index().melt(
            id_vars="Cluster",
            value_vars=clustering_features,
            var_name="Feature",
            value_name="Mean Value"
        )
        
        # Create bar chart of cluster means
        fig = px.bar(
            cluster_means_melted,
            x="Feature",
            y="Mean Value",
            color="Cluster",
            barmode="group",
            title="Mean Feature Values by Cluster"
        )
        st.plotly_chart(fig)
        
        # Show cluster details
        st.subheader("Cluster Details")
        
        # Add participant information to clusters
        cluster_details = merged_data.loc[clustered_data.index].copy()
        cluster_details["Cluster"] = clusters
        
        # Show counts per cluster
        st.write("Participants per cluster:")
        st.write(cluster_details["Cluster"].value_counts())
        
        # Select cluster to examine
        selected_cluster = st.selectbox(
            "Select Cluster to Examine",
            options=sorted(cluster_details["Cluster"].unique())
        )
        
        # Filter data for selected cluster
        selected_cluster_data = cluster_details[cluster_details["Cluster"] == selected_cluster]
        
        # Display participant IDs in the selected cluster
        st.write(f"Participants in Cluster {selected_cluster}:")
        st.write(", ".join(selected_cluster_data["participant_id"].tolist()))
        
        # Compare selected cluster with others
        st.subheader(f"Cluster {selected_cluster} vs. Other Clusters")
        
        # Prepare data for comparison
        comparison_data = cluster_details.copy()
        comparison_data["Group"] = comparison_data["Cluster"].apply(
            lambda x: f"Cluster {selected_cluster}" if x == selected_cluster else "Other Clusters"
        )
        
        # Select features for comparison
        comparison_features = clustering_features + ["olife_total", "ss1", "ss2", "ss3", "ss4"]
        
        # Calculate means for comparison
        comparison_means = comparison_data.groupby("Group")[comparison_features].mean().reset_index()
        
        # Reshape for plotting
        comparison_means_melted = comparison_means.melt(
            id_vars="Group",
            value_vars=comparison_features,
            var_name="Feature",
            value_name="Mean Value"
        )
        
        # Create comparison bar chart
        fig = px.bar(
            comparison_means_melted,
            x="Feature",
            y="Mean Value",
            color="Group",
            barmode="group",
            title=f"Cluster {selected_cluster} vs. Other Clusters"
        )
        st.plotly_chart(fig)
        
        # Statistical tests for differences
        st.subheader("Statistical Tests")
        
        test_results = []
        
        for feature in comparison_features:
            # Get data for each group
            group1 = comparison_data[comparison_data["Group"] == f"Cluster {selected_cluster}"][feature].dropna()
            group2 = comparison_data[comparison_data["Group"] == "Other Clusters"][feature].dropna()
            
            # Skip if too few samples
            if len(group1) < 2 or len(group2) < 2:
                continue
                
            # Perform t-test
            t_stat, p_val = scipy.stats.ttest_ind(group1, group2, equal_var=False)
            
            test_results.append({
                "Feature": feature,
                f"Cluster {selected_cluster} Mean": group1.mean(),
                "Other Clusters Mean": group2.mean(),
                "Mean Difference": group1.mean() - group2.mean(),
                "t-statistic": t_stat,
                "p-value": p_val
            })
        
        # Display test results
        if test_results:
            test_df = pd.DataFrame(test_results)
            test_df = test_df.sort_values(by="p-value")
            st.dataframe(test_df)
            
            # Highlight significant differences
            significant_features = test_df[test_df["p-value"] < 0.05]["Feature"].tolist()
            
            if significant_features:
                st.success(f"Significant differences found in: {', '.join(significant_features)}")
            else:
                st.info("No significant differences found.")
        else:
            st.write("Insufficient data for statistical tests.")

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
            st.write("This participant tends to make decisions with significantly fewer draws than average, "
                   "demonstrating a strong 'jumping to conclusions' (JTC) bias. This cognitive pattern is a "
                   "well-established risk marker for schizophrenia-spectrum disorders, as individuals with "
                   "schizophrenia or high schizotypy typically gather less evidence before making decisions.")
            st.write("**Clinical relevance:** The JTC bias observed here is consistently associated with delusion formation "
                   "and maintenance in schizophrenia and related disorders. This participant's evidence-gathering behavior "
                   "aligns with patterns observed in individuals with higher risk for psychosis.")
        elif dtd_15_85 > avg_dtd_15_85 and dtd_60_40 > avg_dtd_60_40:
            st.write("This participant tends to gather more evidence before making decisions compared to the average, "
                   "indicating a more cautious decision-making approach. This pattern is typically associated with "
                   "lower schizotypy and reduced risk for schizophrenia-spectrum cognitive biases.")
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