import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy import stats
import plotly.graph_objects as go

st.set_page_config(page_title="Voltammetric Analysis App", layout="wide")

def process_voltammogram(potential, current):
   baseline = savgol_filter(current, window_length=21, polyorder=2)
   corrected_current = current - baseline
   peaks, properties = find_peaks(current, height=5, distance=50, prominence=2)
   return {
       'peaks': peaks,
       'peak_heights': current[peaks],
       'peak_potentials': potential[peaks],
       'corrected_current': corrected_current,
       'baseline': baseline
   }

def generate_synthetic_data(base_peak_height, n_points=8, n_replicates=3, noise_level=0.01):
   concentrations = np.linspace(0.1, 2.0, n_points)
   all_data = []
   for conc in concentrations:
       for _ in range(n_replicates):
           current = base_peak_height * conc * (1 + np.random.normal(0, noise_level))
           all_data.append({'concentration': conc, 'current': current})
   return pd.DataFrame(all_data)

def analyze_calibration(df):
   grouped = df.groupby('concentration').agg({
       'current': ['mean', 'std']
   }).reset_index()
   
   slope, intercept, r_value, p_value, std_err = stats.linregress(
       grouped['concentration'], 
       grouped['current']['mean']
   )
   
   r_squared = r_value**2
   lowest_std = grouped['current']['std'].iloc[0]
   LOD = 3.3 * lowest_std / slope
   LOQ = 10 * lowest_std / slope
   
   return {
       'grouped_data': grouped,
       'slope': slope,
       'intercept': intercept,
       'r_squared': r_squared,
       'LOD': LOD,
       'LOQ': LOQ,
       'std_err': std_err
   }

def main():
   st.title('Voltammetric Analysis App')
   
   with st.sidebar:
       st.header('Settings')
       n_replicates = st.slider('Number of Replicates', 3, 10, 3)
       noise_level = st.slider('Noise Level', 0.001, 0.05, 0.01, format="%.3f")
   
   uploaded_file = st.file_uploader("Upload voltammogram data (CSV/Excel)", type=['csv', 'xlsx'])
   
   if uploaded_file:
       try:
           # Modified file reading with different encodings
           if uploaded_file.name.endswith('.csv'):
               try:
                   data = pd.read_csv(uploaded_file, encoding='utf-8')
               except UnicodeDecodeError:
                   try:
                       data = pd.read_csv(uploaded_file, encoding='latin1')
                   except UnicodeDecodeError:
                       data = pd.read_csv(uploaded_file, encoding='cp1252')
           else:
               data = pd.read_excel(uploaded_file)
           
           st.subheader('Data Preview')
           st.dataframe(data.head())
           
           # Process voltammogram
           voltammogram = process_voltammogram(data['Potential'].values, data['Current'].values)
           
           # Generate synthetic calibration data
           base_peak_height = np.max(voltammogram['peak_heights'])
           synthetic_data = generate_synthetic_data(base_peak_height, n_replicates=n_replicates, noise_level=noise_level)
           
           # Analyze calibration
           results = analyze_calibration(synthetic_data)
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.subheader('Voltammogram')
               fig_volt = go.Figure()
               fig_volt.add_trace(go.Scatter(x=data['Potential'], y=data['Current'], name='Raw Data'))
               fig_volt.add_trace(go.Scatter(x=data['Potential'], y=voltammogram['corrected_current'], name='Baseline Corrected'))
               fig_volt.add_trace(go.Scatter(x=data['Potential'][voltammogram['peaks']], 
                                           y=data['Current'][voltammogram['peaks']], 
                                           mode='markers', name='Peaks'))
               fig_volt.update_layout(xaxis_title='Potential (V)', yaxis_title='Current (µA)')
               st.plotly_chart(fig_volt)
           
           with col2:
               st.subheader('Calibration Curve')
               fig_cal = go.Figure()
               grouped = results['grouped_data']
               fig_cal.add_trace(go.Scatter(
                   x=grouped['concentration'],
                   y=grouped['current']['mean'],
                   error_y=dict(type='data', array=grouped['current']['std'], visible=True),
                   mode='markers+lines',
                   name='Data with Error Bars'
               ))
               
               x_range = np.linspace(min(grouped['concentration']), max(grouped['concentration']), 100)
               fig_cal.add_trace(go.Scatter(
                   x=x_range,
                   y=results['slope'] * x_range + results['intercept'],
                   mode='lines',
                   line=dict(dash='dash'),
                   name=f'R² = {results["r_squared"]:.4f}'
               ))
               fig_cal.update_layout(xaxis_title='Relative Concentration', yaxis_title='Peak Current (µA)')
               st.plotly_chart(fig_cal)
           
           st.subheader('Analysis Results')
           col3, col4 = st.columns(2)
           with col3:
               st.metric('Sensitivity', f"{results['slope']:.2f} ± {results['std_err']:.2f} µA/conc")
               st.metric('R²', f"{results['r_squared']:.4f}")
           with col4:
               st.metric('LOD', f"{results['LOD']:.3f} conc")
               st.metric('LOQ', f"{results['LOQ']:.3f} conc")
           
           # Download results
           results_df = pd.DataFrame({
               'Parameter': ['Sensitivity', 'R²', 'LOD', 'LOQ'],
               'Value': [
                   f"{results['slope']:.2f} ± {results['std_err']:.2f}",
                   f"{results['r_squared']:.4f}",
                   f"{results['LOD']:.3f}",
                   f"{results['LOQ']:.3f}"
               ]
           })
           
           csv = results_df.to_csv(index=False).encode('utf-8')
           st.download_button(
               "Download Results",
               csv,
               "analysis_results.csv",
               "text/csv",
               key='download-csv'
           )
           
       except Exception as e:
           st.error(f'Error processing file: {str(e)}')
           st.error("Please ensure your CSV columns are named 'Potential' and 'Current'")
   
   else:
       st.info('Please upload a CSV or Excel file containing voltammetric data')

if __name__ == '__main__':
   main()
