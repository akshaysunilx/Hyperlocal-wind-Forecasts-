#!/usr/bin/env python3
"""
Wind Forecast Visualization (v4.2 Compatible - Notebook Safe)
=============================================================
Optimized plotting for the integrated wind forecast script.
Works seamlessly in Jupyter/Colab and terminal.

Usage:
    # In Jupyter/Colab - just run the cell
    %run plot_wind_v4_notebook.py
    
    # Or import and use
    from plot_wind_v4_notebook import plot_forecast
    plot_forecast()  # Auto-detect latest
    plot_forecast("path/to/file.csv")  # Specific file
"""

import sys
from pathlib import Path
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime

warnings.filterwarnings('ignore')

# Try plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120

# Model colors
COLORS = {
    'GFS': '#E74C3C', 
    'ECMWF': '#3498DB', 
    'ICON': '#2ECC71', 
    'Ensemble': '#9B59B6'
}


def find_latest_csv(pattern="wind_forecast_data/csv/wind_10m_forecast_*UTC.csv"):
    """Find most recent wind forecast CSV"""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda p: Path(p).stat().st_mtime)


def load_forecast(csv_path):
    """Load forecast CSV with metadata extraction"""
    df = pd.read_csv(csv_path)
    
    # Detect time column
    if 'Timestamp_UTC' in df.columns:
        df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC'])
        df = df.set_index('Timestamp_UTC')
        time_col = 'UTC'
    else:
        # Fallback to first datetime column
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                time_col = col
                break
            except:
                continue
    
    # Parse IST if present
    if 'Timestamp_IST' in df.columns:
        df['Timestamp_IST'] = pd.to_datetime(df['Timestamp_IST'])
    
    # Identify model columns
    model_cols = [c for c in df.columns if c.endswith('_10m_ms')]
    
    # Extract metadata
    metadata = {
        'gfs_init': df['GFS_Init_UTC'].iloc[0] if 'GFS_Init_UTC' in df.columns else 'N/A',
        'ecmwf_init': df['ECMWF_Init_UTC'].iloc[0] if 'ECMWF_Init_UTC' in df.columns else 'N/A',
        'icon_init': df['ICON_Init_UTC'].iloc[0] if 'ICON_Init_UTC' in df.columns else 'N/A',
        'ecmwf_stream': df['ECMWF_Stream'].iloc[0] if 'ECMWF_Stream' in df.columns else 'N/A',
    }
    
    # Convert numeric
    for col in model_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("=" * 80)
    print(f"LOADED: {Path(csv_path).name}")
    print("=" * 80)
    print(f"Period: {df.index.min()} to {df.index.max()} ({time_col})")
    print(f"Duration: {(df.index.max() - df.index.min()).total_seconds() / 3600:.1f} hours")
    print(f"Records: {len(df)}")
    print(f"Models: {', '.join([c.replace('_10m_ms', '') for c in model_cols])}")
    print(f"\nModel Initializations:")
    print(f"  GFS:   {metadata['gfs_init']}")
    print(f"  ECMWF: {metadata['ecmwf_init']} [{metadata['ecmwf_stream']}]")
    print(f"  ICON:  {metadata['icon_init']}")
    print("=" * 80 + "\n")
    
    return df, model_cols, metadata, time_col


def calculate_ensemble(df, model_cols):
    """Calculate ensemble statistics"""
    if len(model_cols) < 2:
        print("⚠ Only one model - skipping ensemble\n")
        return df, {}
    
    df['Ensemble_Mean'] = df[model_cols].mean(axis=1)
    df['Ensemble_Median'] = df[model_cols].median(axis=1)
    df['Ensemble_Std'] = df[model_cols].std(axis=1)
    df['Ensemble_Min'] = df[model_cols].min(axis=1)
    df['Ensemble_Max'] = df[model_cols].max(axis=1)
    df['Model_Spread'] = df['Ensemble_Max'] - df['Ensemble_Min']
    df['CI_Lower'] = df['Ensemble_Mean'] - 1.96 * df['Ensemble_Std']
    df['CI_Upper'] = df['Ensemble_Mean'] + 1.96 * df['Ensemble_Std']
    
    # Calculate metrics
    metrics = {}
    
    # Per-model stats
    for col in model_cols:
        name = col.replace('_10m_ms', '')
        data = df[col].dropna()
        if len(data) > 0:
            metrics[name] = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'coverage': 100 * len(data) / len(df)
            }
    
    # Ensemble stats
    metrics['Ensemble'] = {
        'mean': df['Ensemble_Mean'].mean(),
        'avg_spread': df['Model_Spread'].mean(),
        'max_spread': df['Model_Spread'].max(),
        'avg_std': df['Ensemble_Std'].mean(),
    }
    
    # Correlations
    if len(model_cols) >= 2:
        corr = df[model_cols].corr()
        metrics['correlations'] = {}
        for i, c1 in enumerate(model_cols):
            for c2 in model_cols[i+1:]:
                n1 = c1.replace('_10m_ms', '')
                n2 = c2.replace('_10m_ms', '')
                metrics['correlations'][f'{n1}-{n2}'] = corr.loc[c1, c2]
    
    # Agreement thresholds
    metrics['agreement'] = {}
    for thresh in [0.5, 1.0, 2.0]:
        agreement = []
        for _, row in df.iterrows():
            vals = [row[c] for c in model_cols if pd.notna(row[c])]
            if len(vals) >= 2:
                mean_val = np.mean(vals)
                within = all(abs(v - mean_val) <= thresh for v in vals)
                agreement.append(within)
        if agreement:
            metrics['agreement'][f'{thresh}'] = 100 * sum(agreement) / len(agreement)
    
    # Bias vs ensemble
    metrics['bias'] = {}
    for col in model_cols:
        name = col.replace('_10m_ms', '')
        bias = (df[col] - df['Ensemble_Mean']).mean()
        metrics['bias'][name] = bias
    
    print("✓ Ensemble calculated\n")
    print_metrics(metrics)
    
    return df, metrics


def print_metrics(metrics):
    """Print metrics summary"""
    print("-" * 80)
    print("MODEL STATISTICS")
    print("-" * 80)
    
    for model, stats in metrics.items():
        if model in ['Ensemble', 'correlations', 'agreement', 'bias']:
            continue
        print(f"\n{model}:")
        print(f"  Mean: {stats['mean']:.2f} m/s")
        print(f"  Std:  {stats['std']:.2f} m/s")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} m/s")
        print(f"  Coverage: {stats['coverage']:.1f}%")
    
    if 'Ensemble' in metrics:
        print(f"\nEnsemble:")
        print(f"  Mean: {metrics['Ensemble']['mean']:.2f} m/s")
        print(f"  Avg Spread: {metrics['Ensemble']['avg_spread']:.2f} m/s")
        print(f"  Max Spread: {metrics['Ensemble']['max_spread']:.2f} m/s")
    
    if 'correlations' in metrics:
        print(f"\nCorrelations:")
        for pair, corr in metrics['correlations'].items():
            print(f"  {pair}: {corr:.3f}")
    
    if 'agreement' in metrics:
        print(f"\nAgreement:")
        for thresh, pct in metrics['agreement'].items():
            print(f"  Within ±{thresh} m/s: {pct:.1f}%")
    
    if 'bias' in metrics:
        print(f"\nBias (vs Ensemble):")
        for model, bias in metrics['bias'].items():
            sign = '+' if bias >= 0 else ''
            print(f"  {model}: {sign}{bias:.2f} m/s")
    
    print("\n" + "=" * 80 + "\n")


def plot_comprehensive(df, model_cols, metrics, metadata, time_col, save_path=None):
    """Create comprehensive 6-panel visualization"""
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 2, height_ratios=[2.5, 1, 1, 1], hspace=0.35, wspace=0.3)
    
    has_ensemble = 'Ensemble_Mean' in df.columns
    
    # Panel 1: Main forecast
    ax1 = fig.add_subplot(gs[0, :])
    
    for col in model_cols:
        name = col.replace('_10m_ms', '')
        if not df[col].isna().all():
            ax1.plot(df.index, df[col], label=name, 
                    linewidth=2.5, alpha=0.85, color=COLORS[name])
    
    if has_ensemble:
        ax1.fill_between(df.index, df['CI_Lower'], df['CI_Upper'],
                        alpha=0.2, color=COLORS['Ensemble'], label='95% CI')
        ax1.plot(df.index, df['Ensemble_Mean'], label='Ensemble Mean', 
                linewidth=3.5, color=COLORS['Ensemble'])
    
    ax1.set_ylabel('Wind Speed (m/s)', fontweight='bold', fontsize=13)
    ax1.set_title(
        f'Multi-Model 10m Wind Speed Forecast\n'
        f'Target: 23.375°-24.625°N, 68.625°-69.875°E  |  Time: {time_col}',
        fontweight='bold', fontsize=15
    )
    ax1.legend(loc='upper left', frameon=True, shadow=True, ncol=3, fontsize=11)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    
    # Info box
    info_text = f"GFS: {metadata['gfs_init']}\n"
    info_text += f"ECMWF: {metadata['ecmwf_init']} [{metadata['ecmwf_stream']}]\n"
    info_text += f"ICON: {metadata['icon_init']}"
    if has_ensemble and 'Ensemble' in metrics:
        info_text += f"\n\nEnsemble: {metrics['Ensemble']['mean']:.2f} m/s"
        info_text += f"\nSpread: {metrics['Ensemble']['avg_spread']:.2f} m/s"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    # Panel 2: Spread
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    
    if has_ensemble:
        ax2.fill_between(df.index, 0, df['Model_Spread'],
                        alpha=0.5, color='orange', label='Inter-Model Range')
        ax2.plot(df.index, df['Model_Spread'], color='darkorange', linewidth=2.5)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='1 m/s')
        ax2.axhline(y=2.0, color='darkred', linestyle='--', alpha=0.6, linewidth=1.5, label='2 m/s')
    
    ax2.set_ylabel('Spread (m/s)', fontweight='bold', fontsize=12)
    ax2.set_title('Forecast Uncertainty', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # Panel 3: Deviations
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    
    if has_ensemble:
        for col in model_cols:
            name = col.replace('_10m_ms', '')
            deviation = df[col] - df['Ensemble_Mean']
            ax3.plot(df.index, deviation, label=name, alpha=0.8, linewidth=2, color=COLORS[name])
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        ax3.fill_between(df.index, -1, 1, alpha=0.15, color='gray')
    
    ax3.set_ylabel('Deviation (m/s)', fontweight='bold', fontsize=11)
    ax3.set_title('Bias vs Ensemble', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9, ncol=len(model_cols))
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    # Panel 4: Agreement
    ax4 = fig.add_subplot(gs[2, 1], sharex=ax1)
    
    if has_ensemble:
        window = max(5, min(40, len(df) // 20))
        agreement = (df['Model_Spread'] <= 1.0).astype(float)
        rolling_agr = agreement.rolling(window=window, center=True, min_periods=1).mean() * 100
        
        ax4.plot(df.index, rolling_agr, color='green', linewidth=2.5)
        ax4.fill_between(df.index, 0, rolling_agr, alpha=0.3, color='green')
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax4.set_ylabel('Agreement (%)', fontweight='bold', fontsize=11)
    ax4.set_title(f'Model Agreement (±1 m/s, {window}-pt window)', fontweight='bold', fontsize=12)
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.get_xticklabels(), visible=False)
    
    # Panel 5: Distributions
    ax5 = fig.add_subplot(gs[3, 0])
    
    plot_data = [df[col].dropna() for col in model_cols if len(df[col].dropna()) > 10]
    plot_labels = [col.replace('_10m_ms', '') for col in model_cols if len(df[col].dropna()) > 10]
    
    if plot_data:
        bp = ax5.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                        widths=0.6, showmeans=True)
        for patch, label in zip(bp['boxes'], plot_labels):
            patch.set_facecolor(COLORS.get(label, 'gray'))
            patch.set_alpha(0.7)
    
    ax5.set_ylabel('Wind Speed (m/s)', fontweight='bold', fontsize=11)
    ax5.set_title('Statistical Distribution', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Metrics table
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    
    table_text = "PERFORMANCE METRICS\n" + "─" * 35 + "\n\n"
    
    if 'agreement' in metrics:
        table_text += "Agreement:\n"
        for thresh, pct in metrics['agreement'].items():
            table_text += f"  ±{thresh} m/s: {pct:5.1f}%\n"
    
    table_text += "\nCorrelations:\n"
    if 'correlations' in metrics:
        for pair, corr in metrics['correlations'].items():
            table_text += f"  {pair:12s}: {corr:5.3f}\n"
    
    table_text += "\nBias vs Ensemble:\n"
    if 'bias' in metrics:
        for model, bias in metrics['bias'].items():
            sign = '+' if bias >= 0 else ''
            table_text += f"  {model:12s}: {sign}{bias:5.2f} m/s\n"
    
    if has_ensemble and 'Ensemble' in metrics:
        table_text += f"\nEnsemble Stats:\n"
        table_text += f"  Mean:       {metrics['Ensemble']['mean']:5.2f} m/s\n"
        table_text += f"  Avg Spread: {metrics['Ensemble']['avg_spread']:5.2f} m/s\n"
        table_text += f"  Max Spread: {metrics['Ensemble']['max_spread']:5.2f} m/s\n"
    
    ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    ax5.set_xlabel(f'Time ({time_col})', fontweight='bold', fontsize=11)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    
    plt.suptitle(f'Wind Forecast Analysis',
                fontsize=17, fontweight='bold', y=0.9975)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_interactive(df, model_cols, save_path=None):
    """Interactive Plotly visualization"""
    if not PLOTLY_OK:
        print("⚠ Plotly not available")
        return
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=('Wind Speed Forecast', 'Model Spread'),
        row_heights=[0.7, 0.3]
    )
    
    for col in model_cols:
        name = col.replace('_10m_ms', '')
        if not df[col].isna().all():
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=name, mode='lines',
                          line=dict(width=2.5, color=COLORS[name]),
                          hovertemplate='%{x}<br>%{y:.2f} m/s<extra></extra>'),
                row=1, col=1
            )
    
    if 'Ensemble_Mean' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['CI_Upper'], mode='lines',
                      line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['CI_Lower'], mode='lines',
                      line=dict(width=0), fillcolor='rgba(155, 89, 182, 0.2)',
                      fill='tonexty', name='95% CI', hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Ensemble_Mean'], name='Ensemble',
                      mode='lines', line=dict(width=3.5, color=COLORS['Ensemble']),
                      hovertemplate='%{x}<br>%{y:.2f} m/s<extra></extra>'), row=1, col=1)
    
    if 'Model_Spread' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Model_Spread'], name='Spread',
                      mode='lines', fill='tozeroy', line=dict(width=2.5, color='orange'),
                      hovertemplate='%{x}<br>%{y:.2f} m/s<extra></extra>'), row=2, col=1)
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Wind Speed (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (m/s)", row=2, col=1)
    
    fig.update_layout(title='Interactive Wind Forecast', height=850,
                     hovermode='x unified', template='plotly_white')
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    
    fig.show()


def plot_forecast(csv_file=None):
    """
    Main plotting function - notebook safe
    
    Args:
        csv_file: Path to CSV file, or None to auto-detect latest
    """
    # Auto-detect if not provided
    if csv_file is None:
        csv_file = find_latest_csv()
        if csv_file is None:
            print("\n❌ No CSV found in wind_forecast_data/csv/")
            print("\nRun the forecast script first to generate data.")
            return
        print(f"📂 Auto-detected: {csv_file}\n")
    
    # Load
    df, model_cols, metadata, time_col = load_forecast(csv_file)
    
    # Calculate ensemble
    df, metrics = calculate_ensemble(df, model_cols)
    
    # Create output directory
    output_dir = Path("wind_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating visualizations...\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    plot_comprehensive(
        df, model_cols, metrics, metadata, time_col,
        save_path=output_dir / f"wind_analysis_{timestamp}.png"
    )
    
    plot_interactive(
        df, model_cols,
        save_path=output_dir / f"wind_interactive_{timestamp}.html"
    )
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs in: {output_dir.absolute()}")
    for f in sorted(output_dir.glob(f'*{timestamp}*')):
        print(f"  • {f.name}")
    print()


# Safe main that handles notebook arguments
def main():
    """Main function - handles both terminal and notebook execution"""
    # Filter out notebook-specific arguments
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-') and Path(arg).suffix == '.csv']
    
    if args:
        plot_forecast(args[0])
    else:
        plot_forecast()


if __name__ == '__main__':
    main()
