import streamlit as st
import json
import gzip
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Any

# Page configuration
st.set_page_config(
    page_title="Sleep Session Viewer",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("😴 Sleep Session Viewer")
st.markdown("---")

@st.cache_data
def load_json_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Load JSON or JSON.gz file
    
    Args:
        uploaded_file: Uploaded file object from Streamlit
        
    Returns:
        Parsed JSON data or None if failed
    """
    try:
        if uploaded_file.name.endswith('.gz'):
            # Decompress gzip file
            content = gzip.decompress(uploaded_file.read())
            data = json.loads(content.decode('utf-8'))
        else:
            # Load regular JSON
            content = uploaded_file.read()
            data = json.loads(content.decode('utf-8'))
        
        return data
    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")
        return None

def format_timestamp(timestamp_ms: int) -> str:
    """
    Format timestamp in milliseconds to readable string
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        
    Returns:
        Formatted date time string
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def format_seconds_to_hms(seconds: int) -> str:
    """
    Format seconds to hours:minutes:seconds format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string in "X시간 Y분 Z초" format
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}시간")
    if minutes > 0:
        parts.append(f"{minutes}분")
    if secs > 0 or len(parts) == 0:
        parts.append(f"{secs}초")
    
    return " ".join(parts)

def get_session_display_name(session: Dict[str, Any]) -> str:
    """
    Get display name for session (start time)
    
    Args:
        session: Session data dictionary
        
    Returns:
        Display name string
    """
    start_time = format_timestamp(session['startTime'])
    end_time = format_timestamp(session['endTime']) if session.get('endTime') else "진행중"
    return f"{start_time} ~ {end_time}"

def display_session_info(session: Dict[str, Any]):
    """
    Display session information in a formatted way
    
    Args:
        session: Session data dictionary
    """
    st.subheader("📋 세션 정보")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("사용자", session['userName'])
        st.metric("세션 ID", session['sessionId'][:8] + "...")
        st.metric("상태", session['status'])
        st.metric("날짜", session['date'])
    
    with col2:
        st.metric("시작 시간", format_timestamp(session['startTime']))
        if session.get('endTime'):
            st.metric("종료 시간", format_timestamp(session['endTime']))
            duration_sec = (session['endTime'] - session['startTime']) / 1000
            duration_hours = duration_sec / 3600
            st.metric("수면 시간", f"{duration_hours:.1f} 시간")
        else:
            st.metric("종료 시간", "진행중")
    
    with col3:
        st.metric("평균 심박수", f"{session['hrAvg']} bpm")
        st.metric("SpO2 < 90% 시간", format_seconds_to_hms(session['spo2B90TimeSec']))
        st.metric("코골이 시간", format_seconds_to_hms(session['snoreTimeSec']))
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("코골이 최대 dB", f"{session['snoreMaxDb']} dB")
        st.metric("코골이 평균 dB", f"{session['snoreAvgDb']} dB")
    
    with col2:
        st.metric("무호흡 예측 횟수", session['apneaPredCount'])
        st.metric("코골이 예측 횟수", session['snorePredCount'])
    
    with col3:
        st.metric("베개 제어 횟수", session['pillowControlCount'])
        st.metric("베개 제어 지연", f"{session['pillowControlDelayMin']} 분")
    
    with col4:
        pillow_disabled = "비활성화" if session['pillowControlDisabled'] else "활성화"
        st.metric("베개 제어 상태", pillow_disabled)

def plot_sensor_data(data_list: List[Dict[str, Any]], session_id: str, data_type: str, title: str, y_label: str):
    """
    Plot sensor data (HR, SpO2, or Sound) over time
    
    Args:
        data_list: List of sensor data dictionaries
        session_id: Session ID to filter
        data_type: Type of data ('hr', 'spo2', 'sound')
        title: Chart title
        y_label: Y-axis label
    """
    # Filter data for the selected session
    session_data = [d for d in data_list if d['sessionId'] == session_id]
    
    if not session_data:
        st.warning(f"No {data_type} data available for this session")
        return
    
    # Sort by timestamp
    session_data.sort(key=lambda x: x['timestamp'])
    
    # Expand data: each entry has 60 data points
    all_timestamps = []
    all_values = []
    
    for entry in session_data:
        base_timestamp = entry['timestamp']
        data_points = entry['data']
        
        # Each data point represents 1 second
        for i, value in enumerate(data_points):
            timestamp = base_timestamp + (i * 1000)  # Add milliseconds
            all_timestamps.append(timestamp)
            all_values.append(value)
    
    # Convert timestamps to datetime
    datetime_list = [datetime.fromtimestamp(ts / 1000) for ts in all_timestamps]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': datetime_list,
        'Value': all_values
    })
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Value'],
        mode='lines',
        name=data_type.upper(),
        line=dict(width=1.5)
    ))
    
    # Add reference line based on data type
    if data_type == 'spo2':
        # SpO2: 90% 기준선
        fig.add_hline(
            y=90,
            line_dash="dash",
            line_color="red",
            annotation_text="기준: 90%",
            annotation_position="right"
        )
    elif data_type == 'sound':
        # Sound: 40dB 기준선
        fig.add_hline(
            y=40,
            line_dash="dash",
            line_color="red",
            annotation_text="기준: 40dB",
            annotation_position="right"
        )
    else:
        # Other types (e.g., hr): average line
        avg_value = df['Value'].mean()
        fig.add_hline(
            y=avg_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"평균: {avg_value:.1f}",
            annotation_position="right"
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="시간",
        yaxis_title=y_label,
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    if data_type == 'spo2':
        # SpO2 specific statistics with < 90% time
        below_90_count = len(df[df['Value'] < 90])
        below_90_minutes = below_90_count // 60
        below_90_seconds = below_90_count % 60
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("최소값", f"{df['Value'].min():.1f}")
        with col2:
            st.metric("최대값", f"{df['Value'].max():.1f}")
        with col3:
            st.metric("평균값", f"{df['Value'].mean():.1f}")
        with col4:
            st.metric("표준편차", f"{df['Value'].std():.1f}")
        with col5:
            st.metric("90% 미만 시간", f"{below_90_minutes}분 {below_90_seconds}초")
    
    elif data_type == 'sound':
        # Sound specific statistics with >= 40dB time in 5-second intervals
        values_list = df['Value'].tolist()
        time_above_40db = 0
        
        # Process in 5-second intervals
        for i in range(0, len(values_list), 5):
            interval = values_list[i:i+5]
            # If any value in this 5-second interval is >= 40dB, add 5 seconds
            if any(v >= 40 for v in interval):
                time_above_40db += 5
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("최소값", f"{df['Value'].min():.1f}")
        with col2:
            st.metric("최대값", f"{df['Value'].max():.1f}")
        with col3:
            st.metric("평균값", f"{df['Value'].mean():.1f}")
        with col4:
            st.metric("표준편차", f"{df['Value'].std():.1f}")
        with col5:
            st.metric("40dB 이상 시간", format_seconds_to_hms(time_above_40db))
    
    else:
        # Default statistics for other data types (e.g., hr)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("최소값", f"{df['Value'].min():.1f}")
        with col2:
            st.metric("최대값", f"{df['Value'].max():.1f}")
        with col3:
            st.metric("평균값", f"{df['Value'].mean():.1f}")
        with col4:
            st.metric("표준편차", f"{df['Value'].std():.1f}")

def plot_inference_data(inference_list: List[Dict[str, Any]], session_id: str):
    """
    Plot inference data (snorePred, apneaPred, severityPred) over time
    
    Args:
        inference_list: List of inference data dictionaries
        session_id: Session ID to filter
    """
    # Filter data for the selected session
    session_data = [d for d in inference_list if d['sessionId'] == session_id]
    
    if not session_data:
        st.warning("No inference data available for this session")
        return
    
    # Sort by timestamp
    session_data.sort(key=lambda x: x['timestamp'])
    
    # Extract data
    timestamps = [datetime.fromtimestamp(d['timestamp'] / 1000) for d in session_data]
    snore_pred = [d['snorePred'] for d in session_data]
    apnea_pred = [d['apneaPred'] for d in session_data]
    severity_pred = [d['severityPred'] for d in session_data]
    
    # Prepare sensor data for hover tooltip
    # Each inference entry contains hrData, spo2Data, soundData (300 elements)
    # Extract last 240 elements (4 minutes) from each
    custom_data = []
    for inference in session_data:
        sensor_blocks = []
        
        # Head Position and Cell Levels block (combined, no blank line between them)
        head_position = inference.get('headPosition', 0)
        cell_levels = inference.get('ruleBasedCellLevels', [])
        
        position_and_levels = f"머리 위치: {head_position}"
        if cell_levels:
            cell_levels_str = ", ".join([str(level) for level in cell_levels])
            position_and_levels += f"<br>셀 레벨: [{cell_levels_str}]"
        
        sensor_blocks.append(position_and_levels)
        
        # HR data block
        hr_data = inference.get('hrData', [])
        if hr_data and len(hr_data) >= 240:
            hr_vals = hr_data[-240:]  # Last 240 points (4 minutes)
            hr_stats = f"HR: 평균 {int(sum(hr_vals)/len(hr_vals))}, 최소 {int(min(hr_vals))}, 최대 {int(max(hr_vals))}"
            # Format 240 data points (60 values per line)
            hr_formatted = []
            for j in range(0, len(hr_vals), 60):
                chunk = hr_vals[j:j+60]
                hr_formatted.append(", ".join([f"{int(v)}" for v in chunk]))
            hr_data_lines = "<br>".join(hr_formatted)
            hr_block = f"{hr_stats}<br>HR 데이터:<br>{hr_data_lines}"
            sensor_blocks.append(hr_block)
        
        # SpO2 data block
        spo2_data = inference.get('spo2Data', [])
        if spo2_data and len(spo2_data) >= 240:
            spo2_vals = spo2_data[-240:]  # Last 240 points (4 minutes)
            spo2_stats = f"SpO2: 평균 {int(sum(spo2_vals)/len(spo2_vals))}, 최소 {int(min(spo2_vals))}, 최대 {int(max(spo2_vals))}"
            # Format 240 data points
            spo2_formatted = []
            for j in range(0, len(spo2_vals), 60):
                chunk = spo2_vals[j:j+60]
                spo2_formatted.append(", ".join([f"{int(v)}" for v in chunk]))
            spo2_data_lines = "<br>".join(spo2_formatted)
            spo2_block = f"{spo2_stats}<br>SpO2 데이터:<br>{spo2_data_lines}"
            sensor_blocks.append(spo2_block)
        
        # Sound data block
        sound_data = inference.get('soundData', [])
        if sound_data and len(sound_data) >= 240:
            sound_vals = sound_data[-240:]  # Last 240 points (4 minutes)
            sound_stats = f"Sound: 평균 {int(sum(sound_vals)/len(sound_vals))}, 최소 {int(min(sound_vals))}, 최대 {int(max(sound_vals))}"
            # Format 240 data points
            sound_formatted = []
            for j in range(0, len(sound_vals), 60):
                chunk = sound_vals[j:j+60]
                sound_formatted.append(", ".join([f"{int(v)}" for v in chunk]))
            sound_data_lines = "<br>".join(sound_formatted)
            sound_block = f"{sound_stats}<br>Sound 데이터:<br>{sound_data_lines}"
            sensor_blocks.append(sound_block)
        
        # Combine blocks: different sensors separated by blank line
        sensor_info = "<br><br>".join(sensor_blocks)
        custom_data.append(sensor_info)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'SnorePred': snore_pred,
        'ApneaPred': apnea_pred,
        'SeverityPred': severity_pred
    })
    
    # Create subplots: 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('코골이 & 무호흡 예측', '심각도'),
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    # Row 1: Snore and Apnea Prediction
    fig.add_trace(
        go.Scatter(
            x=df['Timestamp'],
            y=df['SnorePred'],
            mode='lines+markers',
            name='코골이 예측',
            line=dict(color='orange', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Timestamp'],
            y=df['ApneaPred'],
            mode='lines+markers',
            name='무호흡 단계',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Row 2: Severity Prediction
    fig.add_trace(
        go.Scatter(
            x=df['Timestamp'],
            y=df['SeverityPred'],
            mode='lines+markers',
            name='심각도 단계',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            customdata=custom_data,
            hovertemplate='<b>시간</b>: %{x}<br>' +
                         '<b>심각도 단계</b>: %{y}<br>' +
                         '%{customdata}' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="시간", row=2, col=1)
    fig.update_yaxes(title_text="단계", row=1, col=1)
    fig.update_yaxes(title_text="단계", row=2, col=1)
    
    # Set y-axis range and ticks to show only 0, 1, 2, 3 for both subplots
    fig.update_yaxes(
        range=[-0.5, 3.5],
        tickmode='array',
        tickvals=[0, 1, 2, 3],
        ticktext=['0', '1', '2', '3']
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background (80% opacity)
            font_size=12,
            font_family="sans-serif"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    st.markdown("### 📊 추론 통계")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("코골이 예측 평균", f"{df['SnorePred'].mean():.2f}")
        st.metric("코골이 최대값", f"{df['SnorePred'].max()}")
    
    with col2:
        st.metric("무호흡 단계 평균", f"{df['ApneaPred'].mean():.2f}")
        st.metric("무호흡 최대 단계", f"{df['ApneaPred'].max()}")
    
    with col3:
        st.metric("심각도 평균", f"{df['SeverityPred'].mean():.2f}")
        st.metric("최대 심각도", f"{df['SeverityPred'].max()}")
    
    # Additional info
    st.markdown("---")
    st.markdown("### 📌 추가 정보")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        head_positions = [d['headPosition'] for d in session_data]
        st.metric("평균 머리 위치", f"{sum(head_positions) / len(head_positions):.1f}")
    
    with col2:
        pillow_controls = sum(1 for d in session_data if d['isPillowControlNeeded'])
        st.metric("베개 제어 필요 횟수", pillow_controls)
    
    with col3:
        accumulated_normal = [d['accumulatedNormalCount'] for d in session_data]
        st.metric("최종 정상 누적", accumulated_normal[-1] if accumulated_normal else 0)

# Main application
def main():
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 파일 업로드")
        uploaded_file = st.file_uploader(
            "JSON 또는 JSON.gz 파일 선택",
            type=['json', 'gz'],
            help="ExportDatabaseToJsonUseCase로 생성된 파일을 업로드하세요"
        )
        
        st.markdown("---")
        st.markdown("### 사용 방법")
        st.markdown("""
        1. JSON 또는 JSON.gz 파일 업로드
        2. 세션 선택
        3. 데이터 및 차트 확인
        """)
    
    # Load and display data
    if uploaded_file is not None:
        data = load_json_file(uploaded_file)
        
        if data is None:
            st.error("파일을 로드할 수 없습니다. 올바른 형식인지 확인하세요.")
            return
        
        # Display export info
        export_info = data.get('exportInfo', {})
        st.info(f"""
        **데이터베이스 내보내기 정보**
        - 버전: {export_info.get('version', 'N/A')}
        - 내보내기 시간: {format_timestamp(export_info.get('exportTimestamp', 0))}
        - 총 세션 수: {export_info.get('totalSessions', 0)}
        """)
        
        # Display total records
        total_records = export_info.get('totalRecords', {})
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("세션", total_records.get('sessions', 0))
        with col2:
            st.metric("HR 데이터", total_records.get('hrData', 0))
        with col3:
            st.metric("SpO2 데이터", total_records.get('spo2Data', 0))
        with col4:
            st.metric("Sound 데이터", total_records.get('soundData', 0))
        with col5:
            st.metric("추론 데이터", total_records.get('inferenceData', 0))
        
        st.markdown("---")
        
        # Session selection
        sessions = data.get('sessions', [])
        
        if not sessions:
            st.warning("세션 데이터가 없습니다.")
            return
        
        # Create session options
        session_options = {get_session_display_name(session): session for session in sessions}
        
        selected_session_name = st.selectbox(
            "세션 선택",
            options=list(session_options.keys()),
            help="시작 시간을 기준으로 세션을 선택하세요"
        )
        
        selected_session = session_options[selected_session_name]
        session_id = selected_session['sessionId']
        
        st.markdown("---")
        
        # Display session info
        display_session_info(selected_session)
        
        # Tabs for different data views
        tab1, tab2, tab3, tab4 = st.tabs(["💓 심박수", "🫁 산소포화도", "🔊 소리", "🧠 추론"])
        
        with tab1:
            st.header("💓 심박수 (HR) 데이터")
            plot_sensor_data(
                data.get('hrData', []),
                session_id,
                'hr',
                '심박수 추이',
                'Heart Rate (bpm)'
            )
        
        with tab2:
            st.header("🫁 산소포화도 (SpO2) 데이터")
            plot_sensor_data(
                data.get('spo2Data', []),
                session_id,
                'spo2',
                '산소포화도 추이',
                'SpO2 (%)'
            )
        
        with tab3:
            st.header("🔊 소리 (Sound) 데이터")
            plot_sensor_data(
                data.get('soundData', []),
                session_id,
                'sound',
                '소리 레벨 추이',
                'Sound Level (dB)'
            )
        
        with tab4:
            st.header("🧠 추론 (Inference) 데이터")
            plot_inference_data(
                data.get('inferenceData', []),
                session_id
            )
    
    else:
        st.info("👈 왼쪽 사이드바에서 JSON 또는 JSON.gz 파일을 업로드하세요.")
        
        st.markdown("### 📖 파일 형식")
        st.markdown("""
        이 뷰어는 `ExportDatabaseToJsonUseCase`로 생성된 데이터베이스 덤프 파일을 읽습니다.
        
        **지원 형식:**
        - `.json` - 압축되지 않은 JSON 파일
        - `.json.gz` - GZIP으로 압축된 JSON 파일
        
        **데이터 구조:**
        - `sessions`: 수면 세션 정보
        - `hrData`: 심박수 데이터 (60초 단위)
        - `spo2Data`: 산소포화도 데이터 (60초 단위)
        - `soundData`: 소리 데이터 (60초 단위)
        - `inferenceData`: AI 추론 결과 (240초 단위)
        """)

if __name__ == "__main__":
    main()

