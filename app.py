import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(page_title="Submarine Monitoring Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2A3A;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E3E56;
    }
    .stats-box {
        background-color: #1E2A3A;
        padding: 10px;
        border-radius: 5px;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .metric-box {
        background-color: #1E2A3A;
        border-radius: 5px;
        padding: 10px;
        flex: 1;
        min-width: 120px;
    }
    .chart-container {
        background-color: #1E2A3A;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data (in a real scenario, this would come from sensors/database)
@st.cache_data
def generate_data(hours=24):
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
    timestamps = sorted(timestamps)
    
    # Depth data (meters)
    base_depth = 400
    depth = base_depth + np.sin(np.linspace(0, 8*np.pi, len(timestamps))) * 50 + np.random.normal(0, 10, len(timestamps))
    
    # Speed data (knots)
    base_speed = 6
    speed = base_speed + np.sin(np.linspace(0, 12*np.pi, len(timestamps))) * 2 + np.random.normal(0, 0.5, len(timestamps))
    
    # Heading data (degrees)
    base_heading = 3
    heading = base_heading + np.sin(np.linspace(0, 6*np.pi, len(timestamps))) * 1 + np.random.normal(0, 0.2, len(timestamps))
    
    # Gyro data
    gyro_x = 0.7 + np.sin(np.linspace(0, 10*np.pi, len(timestamps))) * 0.3 + np.random.normal(0, 0.1, len(timestamps))
    gyro_y = 0.05 + np.sin(np.linspace(0, 12*np.pi, len(timestamps))) * 0.05 + np.random.normal(0, 0.02, len(timestamps))
    gyro_z = 1.7 + np.sin(np.linspace(0, 14*np.pi, len(timestamps))) * 0.4 + np.random.normal(0, 0.15, len(timestamps))
    
    # Temperature data (Celsius)
    temperature = 12 + np.sin(np.linspace(0, 4*np.pi, len(timestamps))) * 2 + np.random.normal(0, 0.5, len(timestamps))
    
    # Pressure data (bars)
    pressure = 40 + depth/10 + np.random.normal(0, 1, len(timestamps))
    
    # Oxygen levels (percentage)
    oxygen = 21 + np.sin(np.linspace(0, 3*np.pi, len(timestamps))) * 0.5 + np.random.normal(0, 0.2, len(timestamps))
    
    # Battery level (percentage)
    battery = 100 - np.linspace(0, 30, len(timestamps)) + np.random.normal(0, 2, len(timestamps))
    battery = np.clip(battery, 0, 100)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'depth': depth,
        'speed': speed,
        'heading': heading,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'temperature': temperature,
        'pressure': pressure,
        'oxygen': oxygen,
        'battery': battery
    })
    
    return df

# Generate historical voyage data
@st.cache_data
def generate_historical_voyages():
    voyages = []
    
    # Create 5 past voyages
    for i in range(5):
        start_date = datetime.now() - timedelta(days=(i+1)*30)
        duration_days = np.random.randint(5, 15)
        end_date = start_date + timedelta(days=duration_days)
        
        max_depth = np.random.randint(300, 600)
        avg_speed = np.random.uniform(5.0, 8.0)
        distance = avg_speed * 24 * duration_days
        fuel_consumption = np.random.uniform(0.8, 1.2) * distance * 0.1
        
        voyage = {
            'voyage_id': f'V-{2024-i}-{np.random.randint(1000, 9999)}',
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': duration_days,
            'max_depth': max_depth,
            'avg_speed': round(avg_speed, 2),
            'distance_nm': round(distance, 2),
            'fuel_consumption': round(fuel_consumption, 2),
            'start_location': np.random.choice(['Pearl Harbor', 'San Diego', 'Norfolk', 'Guam', 'Yokosuka']),
            'end_location': np.random.choice(['Pearl Harbor', 'San Diego', 'Norfolk', 'Guam', 'Yokosuka']),
        }
        voyages.append(voyage)
    
    return pd.DataFrame(voyages)
@st.cache_data
def generate_network_data(hours=24):
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
    timestamps = sorted(timestamps)
    
    # Generate data transfer rates that fluctuate but stay above 1 Mbps
    # Base rate around 1.2 Mbps with fluctuations
    base_rate = 1.2
    download_rate = base_rate + np.sin(np.linspace(0, 10*np.pi, len(timestamps))) * 0.3 + np.random.normal(0, 0.1, len(timestamps))
    upload_rate = base_rate/2 + np.sin(np.linspace(0, 8*np.pi, len(timestamps))) * 0.15 + np.random.normal(0, 0.05, len(timestamps))
    
    # Signal strength (dBm) - typical radio values between -50 and -90
    signal_strength = -65 + np.sin(np.linspace(0, 5*np.pi, len(timestamps))) * 10 + np.random.normal(0, 3, len(timestamps))
    
    # Latency (ms)
    latency = 120 + np.sin(np.linspace(0, 15*np.pi, len(timestamps))) * 30 + np.random.normal(0, 10, len(timestamps))
    
    # Packet loss (%)
    packet_loss = 0.5 + np.sin(np.linspace(0, 12*np.pi, len(timestamps))) * 0.3 + np.abs(np.random.normal(0, 0.2, len(timestamps)))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'download_rate': download_rate,  # in Mbps
        'upload_rate': upload_rate,      # in Mbps
        'signal_strength': signal_strength,  # in dBm
        'latency': latency,              # in ms
        'packet_loss': packet_loss       # in %
    })
    
    return df
# Generate camera footage data
@st.cache_data
def generate_camera_footage():
    footage = []
    
    # Create sample footage entries
    for i in range(10):
        timestamp = datetime.now() - timedelta(hours=i*2)
        camera_id = np.random.choice(['Bow', 'Stern', 'Engine Room', 'Command Center', 'Periscope'])
        footage_type = np.random.choice(['Regular', 'Alert', 'Regular', 'Regular', 'Maintenance'])
        
        entry = {
            'timestamp': timestamp,
            'camera_id': camera_id,
            'footage_type': footage_type,
            'duration_min': np.random.randint(10, 60),
            'file_size_mb': np.random.randint(100, 500),
            'resolution': np.random.choice(['720p', '1080p', '4K']),
        }
        footage.append(entry)
    
    return pd.DataFrame(footage)

# Function to create a subplot with three metrics
def create_metrics_row(title, values, labels):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=labels[0], value=f"{values[0]}")
    with col2:
        st.metric(label=labels[1], value=f"{values[1]}")
    with col3:
        st.metric(label=labels[2], value=f"{values[2]}")

# Main title
st.title("Submarine Monitoring Dashboard")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Real-Time Monitoring", "Historical Data", "Voyage Records", "Camera Footage", "System Diagnostics", "Network Stats"])

# Get data
data = generate_data(24)
current_readings = data.iloc[-1]
network_data = generate_network_data(24)
current_network = network_data.iloc[-1]

historical_voyages = generate_historical_voyages()
camera_footage = generate_camera_footage()

# Tab 1: Real-Time Monitoring
with tab1:
    st.subheader("Real-Time Monitoring")
    st.caption("Live data updated every 5 seconds")
    
    # Current readings
    st.markdown("### Current Readings")
    
    # First row of metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Depth (m)", f"{current_readings['depth']:.1f}")
    with col2:
        st.metric("Speed (knots)", f"{current_readings['speed']:.2f}")
    with col3:
        st.metric("Heading", f"{current_readings['heading']:.2f}")
    
    # Inertia Gyro information
    st.markdown("### Inertia Gyro Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gyro X", f"{current_readings['gyro_x']:.2f}")
    with col2:
        st.metric("Gyro Y", f"{current_readings['gyro_y']:.2f}")
    with col3:
        st.metric("Gyro Z", f"{current_readings['gyro_z']:.2f}")
    
    # Additional Metrics
    st.markdown("### Environmental Readings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temperature (°C)", f"{current_readings['temperature']:.1f}")
    with col2:
        st.metric("Pressure (bar)", f"{current_readings['pressure']:.1f}")
    with col3:
        st.metric("Oxygen Level (%)", f"{current_readings['oxygen']:.1f}")
    
    # Battery status
    st.markdown("### System Status")
    st.progress(current_readings['battery']/100)
    st.text(f"Battery Level: {current_readings['battery']:.1f}%")
    
    # Time Series Charts
    st.markdown("## Time Series Data")
    
    # Create three columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Depth Over Time
        fig_depth = px.line(data, x='timestamp', y='depth', title='Depth Over Time')
        fig_depth.update_layout(
            xaxis_title="Time",
            yaxis_title="Depth (m)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_depth, use_container_width=True)
        
        # Gyro Data Over Time
        fig_gyro = px.line(data, x='timestamp', y=['gyro_x', 'gyro_y', 'gyro_z'], 
                          title='Inertia Gyro Data Over Time')
        fig_gyro.update_layout(
            xaxis_title="Time",
            yaxis_title="Gyro Values",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            legend_title_text="Axis"
        )
        st.plotly_chart(fig_gyro, use_container_width=True)
    
    with col2:
        # Speed Over Time
        fig_speed = px.line(data, x='timestamp', y='speed', title='Speed Over Time')
        fig_speed.update_layout(
            xaxis_title="Time",
            yaxis_title="Speed (knots)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_speed, use_container_width=True)
        
        # Environmental Data
        fig_env = px.line(data, x='timestamp', y=['temperature', 'oxygen'], 
                         title='Environmental Data')
        fig_env.update_layout(
            xaxis_title="Time",
            yaxis_title="Values",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            legend_title_text="Metric"
        )
        st.plotly_chart(fig_env, use_container_width=True)
    
    # Add a refresh button with auto-refresh capability
    auto_refresh = st.checkbox("Enable auto-refresh (5s)")
    if st.button("Refresh Data") or auto_refresh:
        st.rerun()

# Tab 2: Historical Data
with tab2:
    st.subheader("Historical Data Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Generate sample historical data
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    historical_data = pd.DataFrame({
        'timestamp': date_range,
        'depth': np.random.normal(400, 50, len(date_range)),
        'speed': np.random.normal(6, 1, len(date_range)),
        'temperature': np.random.normal(12, 2, len(date_range)),
        'battery': 100 - np.linspace(0, 30, len(date_range)) + np.random.normal(0, 5, len(date_range))
    })
    
    # Metrics selector
    metrics = st.multiselect(
        "Select Metrics to Display",
        ["depth", "speed", "temperature", "battery"],
        default=["depth", "speed"]
    )
    
    if metrics:
        fig = px.line(historical_data, x='timestamp', y=metrics, 
                     title='Historical Data Comparison')
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Values",
            template="plotly_dark",
            height=500,
            legend_title_text="Metric"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    summary = historical_data[metrics].describe().T if metrics else historical_data.describe().T
    st.dataframe(summary, use_container_width=True)
    
    # Download data option
    csv = historical_data.to_csv(index=False)
    st.download_button(
        label="Download Historical Data as CSV",
        data=csv,
        file_name=f"submarine_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# Tab 3: Voyage Records
with tab3:
    st.subheader("Voyage Records")
    
    # Display voyage data
    st.dataframe(historical_voyages, use_container_width=True)
    
    # Select voyage for detailed view
with tab6:
    st.subheader("Radio Communication Network Statistics")
    st.caption("Live connection data from ROV to control station")
    
    # Current network metrics
    st.markdown("### Current Connection Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Download Rate", f"{current_network['download_rate']:.2f} Mbps")
    with col2:
        st.metric("Upload Rate", f"{current_network['upload_rate']:.2f} Mbps")
    with col3:
        st.metric("Signal Strength", f"{current_network['signal_strength']:.1f} dBm")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latency", f"{current_network['latency']:.1f} ms")
    with col2:
        st.metric("Packet Loss", f"{current_network['packet_loss']:.2f}%")
    
    # Connection quality indicator
    st.markdown("### Connection Quality")
    quality_score = 100 - (abs(current_network['signal_strength'] + 50) / 50 * 20) - (current_network['latency'] / 200 * 30) - (current_network['packet_loss'] * 5)
    quality_score = max(min(quality_score, 100), 0)  # Ensure between 0-100
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"### {quality_score:.1f}%")
    with col2:
        st.progress(quality_score/100)
    
    # Time Series Charts for network data
    st.markdown("## Network Performance Over Time")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Data Rate Over Time
        fig_rate = px.line(network_data, x='timestamp', y=['download_rate', 'upload_rate'], 
                          title='Data Transfer Rate Over Time')
        fig_rate.update_layout(
            xaxis_title="Time",
            yaxis_title="Rate (Mbps)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            legend_title_text="Direction"
        )
        # Add a horizontal line at 1 Mbps to emphasize exceeding the threshold
        fig_rate.add_shape(
            type="line", line=dict(dash="dash", color="green"),
            y0=1, y1=1, x0=0, x1=1, xref="paper"
        )
        fig_rate.add_annotation(
            x=0.02, y=1.05, xref="paper", yref="y",
            text="1 Mbps Threshold", showarrow=False,
            font=dict(color="green")
        )
        st.plotly_chart(fig_rate, use_container_width=True)
    
    with col2:
        # Signal Strength Over Time
        fig_signal = px.line(network_data, x='timestamp', y='signal_strength', 
                            title='Radio Signal Strength Over Time')
        fig_signal.update_layout(
            xaxis_title="Time",
            yaxis_title="Signal Strength (dBm)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_signal, use_container_width=True)
    
    # Additional network details
    st.markdown("### Connection Details")
    
    col1, col2 = st.columns(2)
    with col1:
        # Latency Over Time
        fig_latency = px.line(network_data, x='timestamp', y='latency', 
                             title='Connection Latency Over Time')
        fig_latency.update_layout(
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_latency, use_container_width=True)
    
    with col2:
        # Packet Loss Over Time
        fig_packet = px.line(network_data, x='timestamp', y='packet_loss', 
                            title='Packet Loss Percentage Over Time')
        fig_packet.update_layout(
            xaxis_title="Time",
            yaxis_title="Packet Loss (%)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_packet, use_container_width=True)
    
    # Add technical details to make it look more realistic
    st.markdown("### Radio Communication Technical Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Radio Protocol**: MIMO-OFDM Custom Protocol
        - **Frequency Band**: 433 MHz (ISM Band)
        - **Channel Width**: 20 MHz
        - **Modulation**: Adaptive QAM (16-1024)
        - **Error Correction**: Reed-Solomon + LDPC
        """)
    with col2:
        st.markdown("""
        - **Encryption**: AES-256
        - **Antenna Type**: Phased Array (4x4)
        - **Max Range**: 2500m at 0.8 Mbps
        - **Optimized Range**: 1000m at >1.2 Mbps
        - **Water Penetration**: Up to 15m at full performance
        """)