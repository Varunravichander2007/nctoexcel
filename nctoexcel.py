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

def find_time_variable(ds):
    """Find the time variable in the dataset"""
    time_vars = ['valid_time', 'time', 'TIME', 'datetime', 'date', 'Time']
    for var in time_vars:
        if var in ds.variables:
            return var
    return None

def get_variable_units(ds, var_name):
    """Get original units from NC file"""
    try:
        return ds[var_name].attrs.get('units', 'No unit specified')
    except:
        return 'No unit specified'

def load_compressed_file(uploaded_file):
    """Handle compressed or uncompressed NC files"""
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as z:
            nc_file = [f for f in z.namelist() if f.endswith('.nc')][0]
            with z.open(nc_file) as f:
                return xr.open_dataset(f, decode_times=False)
    
    elif filename.endswith('.gz'):
        with gzip.open(io.BytesIO(uploaded_file.getvalue())) as f:
            return xr.open_dataset(f, decode_times=False)
    
    else:  # Regular .nc file
        return xr.open_dataset(io.BytesIO(uploaded_file.getvalue()), decode_times=False)

def convert_point_nc_to_excel(ds, selected_vars, lat, lon, time_var):
    """Convert selected variables to Excel"""
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    
    time_values = pd.to_datetime(ds_point[time_var].values)
    formatted_time = time_values.strftime('%Y-%m-%d %H:%M')
    max_length = len(time_values)
    
    df = pd.DataFrame({
        'observation_time (UTC)': formatted_time,
        'longitude (DD)': [lon] * max_length,
        'latitude (DD)': [lat] * max_length
    })
    
    for var_name in selected_vars:
        unit = get_variable_units(ds, var_name)
        column_name = f"{var_name} ({unit})"
        var_data = ds_point[var_name].values
        
        if var_data.size == 1:
            df[column_name] = var_data.item()
        else:
            df[column_name] = var_data.flatten()
    
    return df

def merge_datasets(datasets):
    """Merge multiple datasets"""
    if len(datasets) == 0:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return xr.merge(datasets)

def visualize_data(df, selected_vars):
    """Create interactive plots with multiple variables on single plot"""
    
    df['formatted_time'] = pd.to_datetime(df['observation_time (UTC)']).dt.strftime('%d-%m-%y %H:%M')
    
    fig = px.line(df, x='formatted_time', y=selected_vars, title='Variables Over Time')
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="white",
            bordercolor="Black",
            borderwidth=1
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            type='category',
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            tickangle=-90,
            title_text=''
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        margin=dict(l=80, r=150, t=100, b=100),
        width=900,
        height=600,
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color='black',
                    width=2,
                )
            )
        ]
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
    try:
        time_var = find_time_variable(ds)
        if not time_var:
            st.error("No time variable found in the dataset.")
            return
        st.write("### Dataset Information:")
        st.write(f"Dimensions: {dict(ds.dims)}")
        
        var_info = []
        for var in ds.variables:
            var_obj = ds[var]
            original_unit = get_variable_units(ds, var)
            var_info.append({
                'Variable': var,
                'Unit': original_unit,
                'Dimensions': ', '.join(var_obj.dims),
                'Description': var_obj.attrs.get('long_name', 'No description available')
            })
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
                    label="📥 Download",
                    data=output.getvalue(),
                    file_name=f"data_{latitude}_{longitude}.xlsx",
                    mime="application/vnd.ms-excel"
                )
                st.success("✅ File ready for download!")
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")

# Main navigation
st.title('🌟 NetCDF File Manager')

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

# Page Content
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
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        datasets = list(executor.map(load_compressed_file, uploaded_files))
                    st.session_state['merged_ds'] = merge_datasets(datasets)
                    if st.session_state['merged_ds'] is not None:
                        st.success(f"Successfully merged {len(datasets)} files!")
                        process_dataset(st.session_state['merged_ds'])

# Footer
st.markdown("---")
st.markdown("""
### 📝 Contact Support:
- Harshitha Gunnam (gunnamharshitha2@gmail.com)
- Varun Ravichander (varunravichander2007@gmail.com)
""")
