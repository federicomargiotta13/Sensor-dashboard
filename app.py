import streamlit as st
import pandas as pd
import gpxpy
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap

# Page configuration for mobile responsiveness
st.set_page_config(page_title="Sensor & Strava Unified Dashboard", layout="wide")

st.title("Sensor & Strava Unified Dashboard")
st.write("Upload multiple sensor CSV files or Strava GPX track files simultaneously.")

# File uploader supporting multiple CSV and GPX files
uploaded_files = st.file_uploader("Choose CSV or GPX files", type=["csv", "gpx"], accept_multiple_files=True)

if uploaded_files:
    combined_dfs = []
    
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        df_clean = None
        
        try:
            # 1. Handle CSV files (Hardware sensors) with robust multi-language coordinate detection
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
                df.columns = df.columns.str.strip()
                
                # Smart detection for Latitude and Longitude supporting English and Italian headers
                lat_candidates = [col for col in df.columns if any(k in col.lower() for k in ['lat', 'latitudine', 'gps_lat'])]
                lon_candidates = [col for col in df.columns if any(k in col.lower() for k in ['lon', 'lng', 'longitudine', 'gps_lon', 'gps_long'])]
                
                lat_col = lat_candidates[0] if lat_candidates else df.columns[0]
                lon_col = lon_candidates[0] if lon_candidates else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
                
                df[lat_col] = pd.to_numeric(df[lat_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                df[lon_col] = pd.to_numeric(df[lon_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                
                df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
                df_clean = df_clean.rename(columns={lat_col: "lat", lon_col: "lon"})
                df_clean["source_file"] = uploaded_file.name
                
            # 2. Handle GPX files (Strava tracks)
            elif file_extension == "gpx":
                gpx_data = gpxpy.parse(uploaded_file)
                points = []
                for track in gpx_data.tracks:
                    for segment in track.segments:
                        for point in segment.points:
                            points.append({
                                "lat": point.latitude,
                                "lon": point.longitude,
                                "elevation": point.elevation if point.elevation is not None else 0.0,
                                "time": point.time,
                                "source_file": uploaded_file.name
                            })
                df_clean = pd.DataFrame(points)
            
            if df_clean is not None and not df_clean.empty:
                combined_dfs.append(df_clean)
                
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            
    if combined_dfs:
        # Combine all uploaded files into a master dataset
        master_df = pd.concat(combined_dfs, ignore_index=True)
        
        with st.expander("View Combined Data Preview"):
            st.dataframe(master_df.head(10))
            
        # Clean invalid coordinate bounds and zeroes
        master_df = master_df[
            (master_df["lat"].between(-90, 90)) & 
            (master_df["lon"].between(-180, 180)) &
            (master_df["lat"].notna()) & 
            (master_df["lon"].notna()) &
            (master_df["lat"] != 0.0) & 
            (master_df["lon"] != 0.0)
        ].copy()
        
        # Chronological sorting if time/timestamp column exists to guarantee correct Start/End order
        time_cols = [c for c in master_df.columns if any(k in c.lower() for k in ['time', 'data', 'ora', 'timestamp'])]
        if time_cols:
            try:
                master_df[time_cols[0]] = pd.to_datetime(master_df[time_cols[0]], errors='coerce')
                master_df = master_df.sort_values(by=time_cols[0]).reset_index(drop=True)
            except Exception:
                master_df = master_df.reset_index(drop=True)
        else:
            master_df = master_df.reset_index(drop=True)
            
        # Smart GPS Jump / Spike Outlier Removal (location-agnostic, cleans stray points without breaking the track)
        st.sidebar.header("Spatial Settings")
        remove_outliers = st.sidebar.checkbox("Remove GPS jump outliers (spike filter)", value=True)
        
        if remove_outliers and len(master_df) > 5:
            lat_diff = master_df["lat"].diff().fillna(0)
            lon_diff = master_df["lon"].diff().fillna(0)
            prev_dist = (lat_diff**2 + lon_diff**2)**0.5
            
            next_lat_diff = master_df["lat"].diff(-1).fillna(0)
            next_lon_diff = master_df["lon"].diff(-1).fillna(0)
            next_dist = (next_lat_diff**2 + next_lon_diff**2)**0.5
            
            median_step = prev_dist[prev_dist > 0].median()
            if not pd.isna(median_step) and median_step > 0:
                jump_threshold = median_step * 20.0
                mask = ~((prev_dist > jump_threshold) & (next_dist > jump_threshold))
                master_df = master_df[mask].reset_index(drop=True)
            
        # Extract available numeric variables for selection
        excluded_terms = ["lat", "lon", "gps", "time", "data", "ora", "timestamp", "unnamed:0", "source_file"]
        available_targets = [col for col in master_df.columns if not any(ex in col.lower() for ex in excluded_terms) and pd.api.types.is_numeric_dtype(master_df[col])]
        
        if not available_targets:
            available_targets = [c for c in master_df.columns if c not in ["lat", "lon", "source_file"]]
            
        st.subheader("Multi-Variable Configuration")
        selected_variables = st.multiselect("Select variables to include in analysis & popups:", available_targets, default=available_targets[:min(3, len(available_targets))])
        
        if selected_variables:
            # Clean numeric values for selected targets first
            for var in selected_variables:
                master_df[var] = pd.to_numeric(master_df[var].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            
            # Sidebar Range Sliders to tighten value ranges for each variable
            st.sidebar.header("Value Range Filters")
            filtered_df = master_df.copy()
            
            for var in selected_variables:
                min_v = float(filtered_df[var].min()) if not filtered_df[var].isna().all() else 0.0
                max_v = float(filtered_df[var].max()) if not filtered_df[var].isna().all() else 1.0
                if min_v < max_v:
                    val_range = st.sidebar.slider(f"Range for {var}", min_value=min_v, max_value=max_v, value=(min_v, max_v))
                    filtered_df = filtered_df[filtered_df[var].between(val_range[0], val_range[1])]
            
            color_variable = st.selectbox("Select primary variable for map coloration:", selected_variables)
            filtered_df = filtered_df.dropna(subset=[color_variable]).reset_index(drop=True)
            
            if filtered_df.empty:
                st.warning("Warning: No data points match the selected value range filters.")
            else:
                total_valid_points = len(filtered_df)
                
                # Map initialization centered on data mean
                center_lat = filtered_df["lat"].mean()
                center_lon = filtered_df["lon"].mean()
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")
                
                min_val = filtered_df[color_variable].min()
                max_val = filtered_df[color_variable].max()
                
                if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                    min_val, max_val = 0, 1
                    
                # Color scale generation for markers
                colormap = LinearColormap(
                    colors=["blue", "green", "yellow", "orange", "red"],
                    vmin=min_val, vmax=max_val,
                    caption=f"Color Metric: {color_variable}"
                )
                colormap.add_to(m)
                
                # Plot points with custom popups displaying all selected metrics
                for idx, row in filtered_df.iterrows():
                    lat, lon = row["lat"], row["lon"]
                    val = row[color_variable]
                    clipped_val = max(min_val, min(val, max_val))
                    point_color = colormap(clipped_val)
                    
                    popup_text = f"<b>Source File:</b> {row.get('source_file', 'Unknown')}<br>"
                    for var in selected_variables:
                        popup_text += f"<b>{var}:</b> {row.get(var, 'N/A')}<br>"
                        
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=5.0,
                        color="#000000",
                        weight=0.5,
                        fill=True,
                        fill_color=point_color,
                        fill_opacity=0.85,
                        popup=folium.Popup(popup_text, max_width=300)
                    ).add_to(m)
                    
                # Start and End markers correctly positioned on chronological bounds
                folium.Marker(
                    location=[filtered_df.iloc[0]["lat"], filtered_df.iloc[0]["lon"]],
                    popup="Start Point",
                    icon=folium.Icon(color="green", icon="play", prefix="fa")
                ).add_to(m)
                
                folium.Marker(
                    location=[filtered_df.iloc[-1]["lat"], filtered_df.iloc[-1]["lon"]],
                    popup="End Point",
                    icon=folium.Icon(color="red", icon="stop", prefix="fa")
                ).add_to(m)
                
                # Automatically fit map bounds to the exact track coordinates
                sw = [filtered_df["lat"].min(), filtered_df["lon"].min()]
                ne = [filtered_df["lat"].max(), filtered_df["lon"].max()]
                m.fit_bounds([sw, ne])
                
                st.subheader(f"Spatial Map Visualisation ({total_valid_points} points analyzed)")
                st_folium(m, width=None, height=650)
else:
    st.info("Please upload one or more CSV sensor files or GPX Strava files to begin.")
