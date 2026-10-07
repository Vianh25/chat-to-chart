import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import chardet
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from google import genai
from google.genai import types

# 1. Cấu hình giao diện Streamlit
st.set_page_config(page_title="AI Chat to Interactive Chart", page_icon="📊", layout="wide")
st.title("📊 Chat to Interactive Chart & Data Insights")

# 2. Định nghĩa Schema trả về cho Gemini (Structured Outputs)
class ChartSeries(BaseModel):
    name: str = Field(description="Tên chuỗi dữ liệu (ví dụ: Doanh số, Lợi nhuận, Tỷ lệ)")
    values: List[float] = Field(description="Danh sách giá trị số tương ứng với từng danh mục trên trục X")

class ChartPayload(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter", "area"] = Field(
        description="Kiểu biểu đồ thích hợp nhất: bar, line, pie, scatter, hoặc area"
    )
    title: str = Field(description="Tiêu đề biểu đồ ngắn gọn, rõ ràng")
    x_axis_title: str = Field(description="Nhãn trục hoành (hoặc tên nhóm danh mục)")
    y_axis_title: str = Field(description="Nhãn trục tung")
    categories: List[str] = Field(description="Danh sách danh mục / thời gian phân loại trên trục X")
    series: List[ChartSeries] = Field(description="Danh sách các tập dữ liệu đo lường")
    insights: str = Field(
        description="Đoạn văn phân tích ngắn gọn: xu hướng, chỉ số nổi bật nhất, tỷ trọng và nhận định hữu ích."
    )

# 3. Hàm render biểu đồ tương tác kèm công cụ tùy biến
def render_interactive_chart(payload: ChartPayload, key_suffix: str = "main", source_label: str = ""):
    if source_label:
        st.caption(f"📌 **Nguồn dữ liệu:** {source_label}")

    categories = payload.categories
    data_dict = {payload.x_axis_title: categories}
    for s in payload.series:
        data_dict[s.name] = s.values
    df = pd.DataFrame(data_dict)
    all_series_names = [s.name for s in payload.series]

    # Khung tùy chỉnh biểu đồ
    with st.expander("🛠️ Tùy chỉnh biểu đồ tương tác", expanded=False):
        col1, col2, col3 = st.columns([2, 3, 2])

        chart_options = ["bar", "line", "area", "pie", "scatter"]
        default_index = chart_options.index(payload.chart_type) if payload.chart_type in chart_options else 0
        selected_chart_type = col1.selectbox(
            "Kiểu biểu đồ:",
            chart_options,
            index=default_index,
            key=f"chart_type_{key_suffix}"
        )

        if selected_chart_type == "pie":
            selected_series = [col2.selectbox(
                "Chọn chỉ số để tính tỷ trọng (%):",
                all_series_names,
                key=f"pie_series_{key_suffix}"
            )]
        else:
            selected_series = col2.multiselect(
                "Lọc chỉ số hiển thị:",
                all_series_names,
                default=all_series_names,
                key=f"series_select_{key_suffix}"
            )

        barmode = "group"
        if selected_chart_type == "bar":
            barmode = col3.selectbox("Dạng cột:", ["group", "stack", "relative"], key=f"barmode_{key_suffix}")
        else:
            col3.write("")

    if not selected_series:
        st.warning("Vui lòng chọn ít nhất một chỉ số để hiển thị.")
        return

    # Vẽ biểu đồ Plotly
    if selected_chart_type == "pie":
        fig = px.pie(
            df,
            names=payload.x_axis_title,
            values=selected_series[0],
            title=payload.title,
            hole=0.35
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
    elif selected_chart_type == "bar":
        fig = px.bar(
            df,
            x=payload.x_axis_title,
            y=selected_series,
            barmode=barmode,
            title=payload.title,
            labels={"value": payload.y_axis_title, "variable": "Chỉ số"}
        )
    elif selected_chart_type == "line":
        fig = px.line(
            df,
            x=payload.x_axis_title,
            y=selected_series,
            markers=True,
            title=payload.title,
            labels={"value": payload.y_axis_title, "variable": "Chỉ số"}
        )
    elif selected_chart_type == "area":
        fig = px.area(
            df,
            x=payload.x_axis_title,
            y=selected_series,
            title=payload.title,
            labels={"value": payload.y_axis_title, "variable": "Chỉ số"}
        )
    elif selected_chart_type == "scatter":
        fig = px.scatter(
            df,
            x=payload.x_axis_title,
            y=selected_series,
            title=payload.title,
            labels={"value": payload.y_axis_title, "variable": "Chỉ số"}
        )

    if selected_chart_type != "pie":
        fig.update_xaxes(rangeslider_visible=True, rangemode="normal")
        fig.update_layout(hovermode="x unified")

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": payload.title.replace(" ", "_"),
                "height": 600,
                "width": 1000,
                "scale": 2
            }
        }
    )

    # Nút tải dữ liệu về CSV
    csv_bytes = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 Tải dữ liệu biểu đồ này (.csv)",
        data=csv_bytes,
        file_name=f"{payload.title.replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"download_{key_suffix}"
    )

# 4. Hàm đọc file (CSV, XLS, XLSX)
def load_uploaded_file(file):
    filename = file.name.lower()
    if filename.endswith(".csv"):
        raw_data = file.read(50000)
        file.seek(0)
        detected = chardet.detect(raw_data)
        encoding = detected.get("encoding") or "utf-8"
        try:
            return pd.read_csv(file, encoding=encoding)
        except Exception:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")
            
    elif filename.endswith(".xlsx"):
        excel_file = pd.ExcelFile(file, engine="openpyxl")
        sheet_names = excel_file.sheet_names
        selected_sheet = sheet_names[0]
        if len(sheet_names) > 1:
            selected_sheet = st.sidebar.selectbox("Chọn trang tính (Sheet):", sheet_names)
        return pd.read_excel(file, sheet_name=selected_sheet, engine="openpyxl")
        
    elif filename.endswith(".xls"):
        excel_file = pd.ExcelFile(file, engine="xlrd")
        sheet_names = excel_file.sheet_names
        selected_sheet = sheet_names[0]
        if len(sheet_names) > 1:
            selected_sheet = st.sidebar.selectbox("Chọn trang tính (Sheet):", sheet_names)
        return pd.read_excel(file, sheet_name=selected_sheet, engine="xlrd")
        
    return None

# 5. Sidebar: Quản lý API Key & Upload file
api_key = os.getenv("GEMINI_API_KEY")
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    if not api_key:
        api_key = st.text_input("Nhập Google Gemini API Key:", type="password")

    st.divider()
    st.subheader("📁 Tải tệp dữ liệu (Tùy chọn)")
    uploaded_file = st.file_uploader(
        "Tải file (.csv, .xls, .xlsx):",
        type=["csv", "xls", "xlsx"],
        help="Bạn có thể upload file để phân tích, hoặc không cần upload mà chat yêu cầu tạo biểu đồ trực tiếp."
    )
    
    current_df = None
    use_file_data = False
    if uploaded_file is not None:
        try:
            current_df = load_uploaded_file(uploaded_file)
            if current_df is not None:
                st.success(f"Đã đọc file: `{uploaded_file.name}`\n\n({current_df.shape[0]} dòng × {current_df.shape[1]} cột)")
                use_file_data = st.checkbox("Sử dụng dữ liệu từ file này cho câu hỏi", value=True)
                with st.expander("👀 Xem trước bảng dữ liệu (5 dòng đầu)"):
                    st.dataframe(current_df.head(5))
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {str(e)}")
    
    st.divider()
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử trò chuyện
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if "chart_data" in msg:
            render_interactive_chart(
                ChartPayload(**msg["chart_data"]),
                key_suffix=f"hist_{idx}",
                source_label=msg.get("source_label", "")
            )
            st.markdown(msg["chart_data"]["insights"])
        else:
            st.markdown(msg["content"])

# 6. Gợi ý nút thao tác nhanh nếu có file
prompt_to_send = None
if current_df is not None and use_file_data and len(st.session_state.messages) == 0:
    st.info("💡 Bạn đã tải file. Có thể bấm nút dưới đây để tạo nhanh biểu đồ từ file, hoặc nhập yêu cầu bất kỳ vào ô chat bên dưới.")
    if st.button("✨ Tự động phân tích và tạo biểu đồ tổng quan từ file"):
        prompt_to_send = "Hãy chọn lọc các chỉ số cốt lõi và giá trị nhất trong bảng dữ liệu để vẽ biểu đồ và phân tích."

# Ô nhập chat
user_input = st.chat_input("Nhập yêu cầu (ví dụ: 'Vẽ biểu đồ doanh số theo vùng từ file' HOẶC 'So sánh GDP Việt Nam và Thái Lan 5 năm qua')...")
if user_input:
    prompt_to_send = user_input

# 7. Xử lý yêu cầu tạo biểu đồ
if prompt_to_send:
    if not api_key:
        st.warning("Vui lòng cung cấp Gemini API Key trong thanh bên để tiếp tục.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    client = genai.Client(api_key=api_key)

    with st.chat_message("assistant"):
        with st.spinner("Gemini đang phân tích và khởi tạo biểu đồ..."):
            try:
                data_context = ""
                source_tag = "Tổng hợp trực tiếp từ Gemini"
                
                if current_df is not None and use_file_data:
                    sample_size = min(len(current_df), 40)
                    df_sample = current_df.head(sample_size).to_markdown(index=False)
                    data_context = (
                        f"\n\n[BẢNG DỮ LIỆU ĐÍNH KÈM TỪ FILE NGƯỜI DÙNG]:\n"
                        f"- Tên file: {uploaded_file.name}\n"
                        f"- Kích thước: {len(current_df)} dòng x {len(current_df.columns)} cột\n"
                        f"- Danh sách cột: {list(current_df.columns)}\n"
                        f"- 40 dòng mẫu đầu tiên:\n{df_sample}\n"
                    )
                    source_tag = f"Tệp đính kèm ({uploaded_file.name})"

                full_prompt = (
                    f"Người dùng có yêu cầu:\n\"{prompt_to_send}\""
                    f"{data_context}\n\n"
                    f"Quy tắc phản hồi:\n"
                    f"1. NẾU CÓ BẢNG DỮ LIỆU ĐÍNH KÈM: Trích xuất và tổng hợp số liệu chuẩn xác dựa trên các cột và dòng trong bảng.\n"
                    f"2. NẾU KHÔNG CÓ BẢNG DỮ LIỆU: Tự động tra cứu, ước tính hoặc tổng hợp dữ liệu thực tế / logic phù hợp nhất với chủ đề được yêu cầu.\n"
                    f"3. Dữ liệu trục hoành đưa vào mảng 'categories'. Mỗi thước đo định lượng đưa vào một phần tử trong mảng 'series'.\n"
                    f"4. Chọn kiểu biểu đồ thích hợp nhất (bar, line, pie, scatter, area).\n"
                    f"5. Đưa ra nhận xét insights súc tích, làm rõ xu hướng, điểm đột biến hoặc tương quan quan trọng."
                )

                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ChartPayload,
                        temperature=0.1,
                    ),
                )

                chart_data = json.loads(response.text)
                payload_obj = ChartPayload(**chart_data)

                render_interactive_chart(
                    payload_obj,
                    key_suffix=f"new_{len(st.session_state.messages)}",
                    source_label=source_tag
                )
                st.markdown(payload_obj.insights)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": payload_obj.insights,
                    "chart_data": chart_data,
                    "source_label": source_tag
                })

            except Exception as e:
                error_msg = f"Đã có lỗi xảy ra: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
