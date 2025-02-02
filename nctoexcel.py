import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import io
import tempfile
import os
import plotly.express as px
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import logging

# Memory and Performance Configuration
APP_CONFIG = {
    'MAX_FILE_SIZE': 5120,  # 5GB in MB
    'MAX_PLOT_POINTS': 2000,
    'CHUNK_SIZE': 500,
    'CACHE_TTL': 7200,
}

# Memory optimization settings
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'
os.environ['PYTHONMALLOC'] = 'malloc'

# Initialize Dask client with clean output
@st.cache_resource(show_spinner=False)
def setup_dask():
    with st.spinner('Initializing processing engine...'):
        cluster = LocalCluster(n_workers=4, 
                             threads_per_worker=2,
                             memory_limit='4GB',
                             silence_logs=logging.ERROR)
        return Client(cluster)

client = setup_dask()

# Page configuration
st.set_page_config(
    page_title="NetCDF File Manager",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.update({
        'page': 'home',
        'datasets': [],
        'merged_ds': None,
        'time_var': None,
        'single_ds': None
    })

@st.cache_data
def load_large_nc_file(file_content):
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=True) as tmp:
        tmp.write(file_content)
        chunks = {'time': 1000, 'latitude': 100, 'longitude': 100}
        ds = xr.open_dataset(tmp.name, chunks=chunks, engine='netcdf4')
    return ds

def find_time_variable(ds):
    time_vars = ['valid_time', 'time', 'TIME', 'datetime', 'date', 'Time']
    return next((var for var in time_vars if var in ds.variables), None)

def get_variable_units(ds, var_name):
    try:
        return ds[var_name].attrs.get('units', 'No unit specified')
    except:
        return 'No unit specified'

def convert_point_nc_to_excel(ds, selected_vars, lat, lon, time_var):
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    
    time_values = pd.to_datetime(ds_point[time_var].values)
    formatted_time = time_values.strftime('%Y-%m-%d %H:%M')
    
    df = pd.DataFrame({
        'observation_time (UTC)': formatted_time,
        'longitude (DD)': [lon] * len(time_values),
        'latitude (DD)': [lat] * len(time_values)
    })
    
    for var_name in selected_vars:
        unit = get_variable_units(ds, var_name)
        column_name = f"{var_name} ({unit})"
        var_data = ds_point[var_name].values
        df[column_name] = var_data.flatten() if var_data.size > 1 else var_data.item()
    
    return df

def merge_datasets(datasets):
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    merged = xr.merge(datasets, compat='override', join='outer')
    return merged.chunk({'time': 1000})

def visualize_data(df, selected_vars):
    if len(df) > APP_CONFIG['MAX_PLOT_POINTS']:
        sampling_rate = len(df) // APP_CONFIG['MAX_PLOT_POINTS']
        df = df.iloc[::sampling_rate].copy()
    
    df['formatted_time'] = pd.to_datetime(df['observation_time (UTC)']).dt.strftime('%H:%M %Y-%m-%d')
    
    fig = px.line(df, x='formatted_time', y=selected_vars,
                  title='Variables Over Time')
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            type='category',
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        width=900,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    buffer = io.StringIO()
    fig.write_html(buffer)
    st.download_button(
        label="📊 Download Plot",
        data=buffer.getvalue(),
        file_name="plot.html",
        mime="text/html"
    )

def process_dataset(ds):
    time_var = find_time_variable(ds)
    if not time_var:
        st.error("No time variable found in the dataset.")
        return
    
    st.write("### Dataset Information:")
    st.write(f"Dimensions: {dict(ds.dims)}")
    
    var_info = [{
        'Variable': var,
        'Unit': get_variable_units(ds, var),
        'Dimensions': ', '.join(ds[var].dims),
        'Description': ds[var].attrs.get('long_name', 'No description available')
    } for var in ds.variables]
    
    st.table(pd.DataFrame(var_info))
    
    available_vars = [var for var in ds.variables
                     if var not in ['latitude', 'longitude', 'lat', 'lon', time_var]]
    
    selected_vars = st.multiselect("Select variables to extract:", available_vars)
    
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude (DD)",
                                 value=float(ds.latitude.mean()),
                                 min_value=float(ds.latitude.min()),
                                 max_value=float(ds.latitude.max()))
    with col2:
        longitude = st.number_input("Longitude (DD)",
                                  value=float(ds.longitude.mean()),
                                  min_value=float(ds.longitude.min()),
                                  max_value=float(ds.longitude.max()))
    
    action = st.radio("Choose action:", ["Visualize", "Excel"])
    
    if selected_vars and st.button("Generate"):
        with st.spinner('Processing data...'):
            df = convert_point_nc_to_excel(ds, selected_vars, latitude, longitude, time_var)
            
            if action == "Visualize":
                visualize_data(df, df.columns[3:])
            else:
                st.write("### Data Preview:")
                st.dataframe(df.head())
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=output.getvalue(),
                    file_name=f"data_{latitude}_{longitude}.xlsx",
                    mime="application/vnd.ms-excel"
                )

# UI Layout
st.title('🌟 NetCDF File Manager')

# Navigation
cols = st.columns([1,1,1,1])
with cols[0]:
    if st.button('🏠 Home'): st.session_state['page'] = 'home'
with cols[1]:
    if st.button('📥 Convert NC to Excel'): st.session_state['page'] = 'convert'
with cols[2]:
    if st.button('🔄 Merge NC Files'): st.session_state['page'] = 'merge'
with cols[3]:
    if st.button('🔄 Reset'):
        for key in ['page', 'datasets', 'merged_ds', 'time_var', 'single_ds']:
            st.session_state[key] = None
        st.session_state['page'] = 'home'
        st.experimental_rerun()

st.markdown("---")

# Page Content
if st.session_state['page'] == 'home':
    st.header("Welcome to NetCDF File Manager!")
    st.write("""
    ### Features Available:
    - Handle large files (up to 5GB)
    - Fast processing with parallel computing
    - Interactive visualization
    - Excel export
    - Multiple file merging
    - Time series analysis
    """)

elif st.session_state['page'] == 'convert':
    st.header("Convert NC File to Excel")
    st.info(f"Maximum file size: {APP_CONFIG['MAX_FILE_SIZE']}MB")
    
    uploaded_file = st.file_uploader("Upload your .nc file", type='nc')
    
    if uploaded_file:
        file_size = uploaded_file.size / (1024 * 1024)
        
        if file_size > APP_CONFIG['MAX_FILE_SIZE']:
            st.error(f"File too large! Maximum size is {APP_CONFIG['MAX_FILE_SIZE']}MB")
        else:
            with st.spinner(f'Processing {file_size:.1f}MB NC file...'):
                try:
                    st.session_state['single_ds'] = load_large_nc_file(uploaded_file.read())
                    process_dataset(st.session_state['single_ds'])
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

elif st.session_state['page'] == 'merge':
    st.header("Merge Multiple NC Files")
    st.info(f"Upload multiple NC files (Max {APP_CONFIG['MAX_FILE_SIZE']}MB each)")
    
    uploaded_files = st.file_uploader("Upload NC files", type='nc', accept_multiple_files=True)
    
    if uploaded_files:
        total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)
        st.write(f"Total size: {total_size:.1f}MB")
        
        if any(file.size / (1024 * 1024) > APP_CONFIG['MAX_FILE_SIZE'] for file in uploaded_files):
            st.error(f"One or more files exceed the {APP_CONFIG['MAX_FILE_SIZE']}MB limit!")
        else:
            if len(uploaded_files) < 2:
                st.warning("Please upload at least 2 files")
            else:
                if st.button("🔄 Merge Files"):
                    progress_bar = st.progress(0)
                    datasets = []
                    
                    for i, file in enumerate(uploaded_files):
                        try:
                            ds = load_large_nc_file(file.read())
                            datasets.append(ds)
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            break
                    
                    if len(datasets) == len(uploaded_files):
                        st.session_state['merged_ds'] = merge_datasets(datasets)
                        if st.session_state['merged_ds'] is not None:
                            st.success(f"Successfully merged {len(datasets)} files!")
                            process_dataset(st.session_state['merged_ds'])

# Footer
st.markdown("---")
st.markdown("### 📝 NetCDF File Manager - Optimized for Large Files")
