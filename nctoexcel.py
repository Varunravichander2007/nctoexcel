import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import io
import concurrent.futures
import plotly.express as px
import zipfile
import gzip
from pathlib import Path

# Increase memory efficiency with dask
import dask

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

def load_compressed_file(uploaded_file):
    """Handle compressed or uncompressed NC files"""
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as z:
            nc_file = [f for f in z.namelist() if f.endswith('.nc')][0]
            with z.open(nc_file) as f:
                return xr.open_dataset(f, chunks={'time': 1000}, decode_times=False)
    
    elif filename.endswith('.gz'):
        with gzip.open(io.BytesIO(uploaded_file.getvalue())) as f:
            return xr.open_dataset(f, chunks={'time': 1000}, decode_times=False)
    
    else:  # Regular .nc file
        return xr.open_dataset(io.BytesIO(uploaded_file.getvalue()), 
                             chunks={'time': 1000}, 
                             decode_times=False)

def process_multiple_files(uploaded_files):
    """Process multiple files in parallel"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        datasets = list(executor.map(load_compressed_file, uploaded_files))
    return xr.merge(datasets)

# Your existing helper functions remain the same
[find_time_variable, get_variable_units, convert_point_nc_to_excel, merge_datasets, visualize_data]

def process_dataset(ds):
    """Optimized dataset processing"""
    try:
        time_var = find_time_variable(ds)
        if not time_var:
            st.error("No time variable found in the dataset.")
            return
            
        # Load only necessary variables
        available_vars = [var for var in ds.variables
                         if var not in ['latitude', 'longitude', 'lat', 'lon', time_var]]
        
        selected_vars = st.multiselect("Select variables to extract:", available_vars)
        
        if selected_vars:
            # Process in chunks
            ds = ds[selected_vars + ['latitude', 'longitude', time_var]]
            
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
            
            if st.button("Generate"):
                with st.spinner("Processing data chunks..."):
                    df = convert_point_nc_to_excel(ds, selected_vars, latitude, longitude, time_var)
                    
                    if action == "Visualize":
                        visualize_data(df, df.columns[3:])
                    else:
                        st.write("### Data Preview:")
                        st.dataframe(df.head())
                        
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False)
                        output.seek(0)
                        
                        st.download_button(
                            label="📥 Download Excel",
                            data=output.getvalue(),
                            file_name=f"data_{latitude}_{longitude}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                        st.success("✅ File ready for download!")

    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")

# Main navigation and UI
st.title('🌟 NetCDF File Manager')

# Navigation buttons
cols = st.columns([1,1,1,1])
nav_buttons = {
    '🏠 Home': 'home',
    '📥 Convert NC to Excel': 'convert',
    '🔄 Merge NC Files': 'merge',
    '🔄 Reset': 'reset'
}

for col, (button_text, page) in zip(cols, nav_buttons.items()):
    with col:
        if st.button(button_text):
            if page == 'reset':
                st.session_state.clear()
                st.session_state['page'] = 'home'
            else:
                st.session_state['page'] = page
            st.rerun()

st.markdown("---")

# Page content
if st.session_state['page'] == 'home':
    st.header("Welcome to NetCDF File Manager!")
    st.write("""
    ### Supported File Formats:
    - .nc (NetCDF files)
    - .zip (Compressed NC files)
    - .gz (Gzipped NC files)
    
    ### Features:
    - Efficient file processing
    - Parallel file merging
    - Chunk-based processing
    - Compressed file support
    - Data visualization
    - Excel export
    """)

elif st.session_state['page'] == 'convert':
    st.header("Convert NC File to Excel")
    uploaded_file = st.file_uploader("Upload .nc file (supports .nc, .zip, .gz)", 
                                   type=['nc', 'zip', 'gz'])
    
    if uploaded_file:
        with st.spinner('Processing file...'):
            st.session_state['single_ds'] = load_compressed_file(uploaded_file)
            process_dataset(st.session_state['single_ds'])

elif st.session_state['page'] == 'merge':
    st.header("Merge Multiple NC Files")
    uploaded_files = st.file_uploader("Upload NC files (supports .nc, .zip, .gz)", 
                                    type=['nc', 'zip', 'gz'],
                                    accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) < 2:
            st.warning("Please upload at least 2 files to merge")
        else:
            if st.button("🔄 Merge Files"):
                with st.spinner('Processing files in parallel...'):
                    st.session_state['merged_ds'] = process_multiple_files(uploaded_files)
                    if st.session_state['merged_ds'] is not None:
                        process_dataset(st.session_state['merged_ds'])

# Footer
st.markdown("---")
st.markdown("""
### 📝 Contact Support:
- Harshitha Gunnam (gunnamharshitha2@gmail.com)
- Varun Ravichander (varunravichander2007@gmail.com)
""")
