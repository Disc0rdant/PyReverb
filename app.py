import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from io import StringIO 

# Import functions
from material_parser import parse_materials_csv, FREQ_BANDS_HZ, FREQ_BANDS_STR, AIR_ABSORPTION_M_LIST
from acoustic_calculator import (
    calculate_rt_sabine, calculate_rt_eyring, calculate_rt_fitzroy,
    calculate_schroeder_freq
)

# --- AS/NZS 2107:2016 Room Type Recommendations ---
ASNZS_ROOM_TYPES = {
    "Not Specified": {"target_tmf": None, "description": "No specific AS/NZS 2107 target selected."},
    "Office - General Office Space": {"target_tmf": 0.7, "description": "Typically < 0.7s, aim for lower in open plan."},
    "Office - Private Office, <50m³": {"target_tmf": 0.6, "description": "≤ 0.6s"},
    "Office - Conference Room, <50m³": {"target_tmf": 0.6, "description": "≤ 0.6s"},
    "Office - Conference Room, 50-200m³": {"target_tmf": 0.7, "description": "0.5 to 0.7s"},
    "Educational - Classroom, <85m³ (Primary/Secondary)": {"target_tmf": 0.5, "description": "≤ 0.5s"}, 
    "Educational - Classroom, >85m³ (Primary/Secondary)": {"target_tmf": 0.6, "description": "≤ 0.6s"},
    "Educational - Lecture Theatre, <100 seats": {"target_tmf": 0.8, "description": "0.6 to 0.8s"},
    "Educational - Lecture Theatre, >100 seats": {"target_tmf": 1.0, "description": "0.8 to 1.0s"},
    "Residential - Living Room": {"target_tmf": 0.6, "description": "0.4 to 0.6s"},
    "Residential - Bedroom": {"target_tmf": 0.5, "description": "0.4 to 0.5s"},
    "Restaurant/Cafeteria": {"target_tmf": 0.8, "description": "≤ 0.8s (can be higher if lively atmosphere desired)"},
    "Hospital - Ward (Multi-bed)": {"target_tmf": 0.8, "description": "≤ 0.8s"},
    "Hospital - Single Patient Room": {"target_tmf": 0.7, "description": "≤ 0.7s"},
    "Sports Hall - Multi-purpose": {"target_tmf": 1.5, "description": "1.2 to 1.5s (speech); up to 2.0s (sport)"},
    "Music - Rehearsal Room (Small, Pop/Rock)": {"target_tmf": 0.4, "description": "0.3 to 0.5s"},
    "Music - Auditorium (Classical)": {"target_tmf": 1.8, "description": "Highly variable, e.g., 1.6 to 2.0s for orchestral"},
}
ROOM_TYPE_OPTIONS = list(ASNZS_ROOM_TYPES.keys())

# --- CSV Data Loading ---
CSV_FILENAME = "reverberation_common_material_absorption_coefficient.csv"
try:
    with open(CSV_FILENAME, "r", encoding="utf-8-sig") as f:
        CSV_FILE_CONTENT = f.read()
    MATERIALS_DB, MATERIAL_DESCRIPTIONS = parse_materials_csv(CSV_FILE_CONTENT)
except FileNotFoundError:
    st.error(f"ERROR: '{CSV_FILENAME}' not found. Please place it in the same directory as the app.")
    CSV_FILE_CONTENT = "Surface Type,Code,Description,125 Hz,250 Hz,500 Hz,1k Hz,2k Hz,4k Hz\nDefault,0,No element selected,0,0,0,0,0,0" # Note the (0) is added by parser
    MATERIALS_DB, MATERIAL_DESCRIPTIONS = parse_materials_csv(CSV_FILE_CONTENT)
    if not MATERIAL_DESCRIPTIONS: 
        MATERIAL_DESCRIPTIONS.append("No element selected (0)")
    if "No element selected (0)" not in MATERIALS_DB:
         MATERIALS_DB["No element selected (0)"] = {"coeffs": [0.0]*6, "category": "Default", "code": "0", "description_only":"No element selected"}


DEFAULT_MATERIAL_DISPLAY_NAME = "No element selected (0)" 
if DEFAULT_MATERIAL_DISPLAY_NAME not in MATERIAL_DESCRIPTIONS:
    MATERIAL_DESCRIPTIONS.insert(0, DEFAULT_MATERIAL_DISPLAY_NAME)
    if DEFAULT_MATERIAL_DISPLAY_NAME not in MATERIALS_DB: 
        MATERIALS_DB[DEFAULT_MATERIAL_DISPLAY_NAME] = {"coeffs": [0.0] * 6, "category": "Default", "code": "0", "description_only":"No element selected"}

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Room Reverberation Calculator")
st.title("ROOM REVERBERATION CALCULATOR")

# --- Input Section ---
with st.sidebar:
    st.header("Room Configuration")
    selected_room_type = st.selectbox(
        "Select AS/NZS 2107:2016 Room Type (for Tmf target)",
        options=ROOM_TYPE_OPTIONS, index=0 
    )
    if ASNZS_ROOM_TYPES[selected_room_type]["target_tmf"] is not None:
        st.caption(f"Target Tmf: {ASNZS_ROOM_TYPES[selected_room_type]['target_tmf']:.2f} s. "
                   f"({ASNZS_ROOM_TYPES[selected_room_type]['description']})")
    else: st.caption(ASNZS_ROOM_TYPES[selected_room_type]['description'])

    st.header("Room Dimensions (meters)")
    if 'L_val' not in st.session_state: st.session_state.L_val = 10.0
    if 'W_val' not in st.session_state: st.session_state.W_val = 7.0
    if 'H_val' not in st.session_state: st.session_state.H_val = 3.0

    def update_dim_val_from_slider(slider_key, val_key): st.session_state[val_key] = st.session_state[slider_key]
    def update_slider_from_dim_val(val_key, slider_key): 
        if st.session_state[val_key] >= 2.0 and st.session_state[val_key] <= 40.0: # Check if within slider range
            st.session_state[slider_key] = st.session_state[val_key]
        # else: slider remains as is, number input is master

    L_val_ui = st.number_input("Length (L)", min_value=0.1, max_value=100.0, value=st.session_state.L_val, step=0.1, key="L_num_ui", on_change=update_slider_from_dim_val, args=("L_num_ui", "L_slider_ui"))
    L_slider_ui = st.slider("L slider", min_value=2.0, max_value=40.0, value=st.session_state.L_val, step=0.1, key="L_slider_ui", on_change=update_dim_val_from_slider, args=("L_slider_ui", "L_num_ui"))
    st.session_state.L_val = L_val_ui 

    W_val_ui = st.number_input("Width (W)", min_value=0.1, max_value=100.0, value=st.session_state.W_val, step=0.1, key="W_num_ui", on_change=update_slider_from_dim_val, args=("W_num_ui", "W_slider_ui"))
    W_slider_ui = st.slider("W slider", min_value=2.0, max_value=40.0, value=st.session_state.W_val, step=0.1, key="W_slider_ui", on_change=update_dim_val_from_slider, args=("W_slider_ui", "W_num_ui"))
    st.session_state.W_val = W_val_ui

    H_val_ui = st.number_input("Height (H)", min_value=0.1, max_value=100.0, value=st.session_state.H_val, step=0.1, key="H_num_ui", on_change=update_slider_from_dim_val, args=("H_num_ui", "H_slider_ui"))
    H_slider_ui = st.slider("H slider", min_value=2.0, max_value=40.0, value=st.session_state.H_val, step=0.1, key="H_slider_ui", on_change=update_dim_val_from_slider, args=("H_slider_ui", "H_num_ui"))
    st.session_state.H_val = H_val_ui
    
    L_calc, W_calc, H_calc = st.session_state.L_val, st.session_state.W_val, st.session_state.H_val

    st.header("Surface Materials")
    material_options = MATERIAL_DESCRIPTIONS 
    default_idx = material_options.index(DEFAULT_MATERIAL_DISPLAY_NAME) if DEFAULT_MATERIAL_DISPLAY_NAME in material_options else 0

    wall_rear_mat = st.selectbox("Wall Rear Material", options=material_options, index=default_idx, key="mat_wr")
    front_wall_mat = st.selectbox("Front Wall Material", options=material_options, index=default_idx, key="mat_fw")
    wall_left_mat = st.selectbox("Wall Left Material", options=material_options, index=default_idx, key="mat_wl")
    wall_right_mat = st.selectbox("Wall Right Material", options=material_options, index=default_idx, key="mat_wrg")
    ceiling_mat = st.selectbox("Ceiling Material", options=material_options, index=default_idx, key="mat_ceil")
    floor_mat = st.selectbox("Floor Material", options=material_options, index=default_idx, key="mat_floor")

    st.header("Windows")
    window_area_m2 = st.number_input("Total Window Area (m²)", min_value=0.0, value=0.0, step=0.1, key="win_area")
    
    window_material_subset = [m for m in material_options if "window" in m.lower() or "glass" in m.lower() or m == DEFAULT_MATERIAL_DISPLAY_NAME]
    if not window_material_subset: window_material_subset = [DEFAULT_MATERIAL_DISPLAY_NAME] # Ensure not empty
    
    # Try to find specific window materials, otherwise use general default
    default_win_mat_key = "Window - up to 4mm glass (120)" # Check from your full CSV if this code is right
    if default_win_mat_key not in window_material_subset: default_win_mat_key = "Window - up to 4mm glass (5)" # Another common one
    if default_win_mat_key not in window_material_subset: default_win_mat_key = DEFAULT_MATERIAL_DISPLAY_NAME # Ultimate fallback
    
    window_mat_idx = window_material_subset.index(default_win_mat_key) if default_win_mat_key in window_material_subset else 0
    window_mat = st.selectbox("Window Material", options=window_material_subset, index=window_mat_idx, key="mat_win")
    
    st.header("Ambient Sound Level")
    ambient_LpA_1k = st.number_input("Ambient Sound Level (dBA at 1kHz or Overall)", min_value=0, max_value=120, value=40, step=1)
    provide_octave_ambient = st.checkbox("Provide Octave Band Ambient Levels (dB)", value=False)
    ambient_octave_levels = {}
    if provide_octave_ambient:
        st.caption("Enter unweighted (Z) or A-weighted levels if known:")
        for freq_hz, freq_str_display in zip(FREQ_BANDS_HZ, FREQ_BANDS_STR):
            ambient_octave_levels[freq_hz] = st.number_input(f"Ambient at {freq_str_display}", min_value=0, max_value=120, value=ambient_LpA_1k - 10, step=1, key=f"amb_{freq_hz}")

# --- Calculations ---
volume_m3 = L_calc * W_calc * H_calc if L_calc > 0 and W_calc > 0 and H_calc > 0 else 0.0

surfaces_details = {
    "Wall Rear":  {"name": "Wall Rear",  "material": wall_rear_mat,  "area": W_calc * H_calc},
    "Front Wall": {"name": "Front Wall", "material": front_wall_mat, "area": W_calc * H_calc},
    "Wall Left":  {"name": "Wall Left",  "material": wall_left_mat,  "area": L_calc * H_calc},
    "Wall Right": {"name": "Wall Right", "material": wall_right_mat, "area": L_calc * H_calc},
    "Ceiling":    {"name": "Ceiling",    "material": ceiling_mat,    "area": L_calc * W_calc},
    "Floor":      {"name": "Floor",      "material": floor_mat,      "area": L_calc * W_calc},
}
if window_area_m2 > 0 and "Front Wall" in surfaces_details and surfaces_details["Front Wall"]["area"] >= window_area_m2:
    surfaces_details["Front Wall"]["area"] -= window_area_m2
    surfaces_details["Windows"] = {"name": "Windows", "material": window_mat, "area": window_area_m2}
elif window_area_m2 > 0: # If front wall too small or not there, subtract from largest wall or log warning
    st.warning(f"Window area ({window_area_m2}m²) exceeds available Front Wall area or Front Wall not defined. Windows not fully accounted for in surface area subtraction.")
    # Simplistic: Add window as separate, don't subtract if complex
    surfaces_details["Windows"] = {"name": "Windows", "material": window_mat, "area": window_area_m2}


total_surface_absorption_list = [0.0] * len(FREQ_BANDS_HZ) 
total_surface_area_m2 = 0.0

for surf_name, details in surfaces_details.items():
    if details["area"] < 0 : details["area"] = 0 # Ensure no negative area
    total_surface_area_m2 += details["area"]
    mat_data = MATERIALS_DB.get(details["material"], MATERIALS_DB[DEFAULT_MATERIAL_DISPLAY_NAME]) 
    mat_coeffs = mat_data.get("coeffs", [0.0]*6)
    for i in range(len(FREQ_BANDS_HZ)):
        total_surface_absorption_list[i] += details["area"] * mat_coeffs[i]

air_absorption_term_list = [4 * m * volume_m3 for m in AIR_ABSORPTION_M_LIST] if volume_m3 > 0 else [0.0]*len(FREQ_BANDS_HZ)
total_sabine_absorption_list = [s_a + air_a for s_a, air_a in zip(total_surface_absorption_list, air_absorption_term_list)]
avg_alpha_surf_list = [tsa / total_surface_area_m2 if total_surface_area_m2 > 1e-6 else 0.0 for tsa in total_surface_absorption_list]

rt_sabine_list = calculate_rt_sabine(volume_m3, total_sabine_absorption_list)
rt_eyring_list = calculate_rt_eyring(volume_m3, total_surface_area_m2, avg_alpha_surf_list, air_absorption_term_list)

fitzroy_material_data = {
    'Ceiling_Material': ceiling_mat, 'Floor_Material': floor_mat,
    'FrontWall_Material': front_wall_mat, 'RearWall_Material': wall_rear_mat,
    'LeftWall_Material': wall_left_mat, 'RightWall_Material': wall_right_mat
}
rt_fitzroy_list = calculate_rt_fitzroy(volume_m3, L_calc, W_calc, H_calc, fitzroy_material_data, MATERIALS_DB)
sch_freq_list = calculate_schroeder_freq(rt_sabine_list, volume_m3)

# --- AS/NZS 2107 Compliance Check ---
target_tmf_value = ASNZS_ROOM_TYPES[selected_room_type]["target_tmf"]
compliance_message = ""
meets_recommendation = None
current_tmf = -1.0

if target_tmf_value is not None and volume_m3 > 1e-6:
    rt_500_idx = FREQ_BANDS_HZ.index(500)
    rt_1k_idx = FREQ_BANDS_HZ.index(1000)
    
    current_rt_500 = rt_sabine_list[rt_500_idx] if not (math.isinf(rt_sabine_list[rt_500_idx]) or math.isnan(rt_sabine_list[rt_500_idx])) else float('inf')
    current_rt_1k = rt_sabine_list[rt_1k_idx] if not (math.isinf(rt_sabine_list[rt_1k_idx]) or math.isnan(rt_sabine_list[rt_1k_idx])) else float('inf')

    if current_rt_500 == float('inf') or current_rt_1k == float('inf'):
        current_tmf = float('inf')
        compliance_message = f"Cannot determine Tmf compliance due to extremely high/infinite calculated RT at 500Hz or 1kHz."
        meets_recommendation = False
    else:
        current_tmf = (current_rt_500 + current_rt_1k) / 2.0
        if current_tmf <= target_tmf_value * 1.05: # Adding 5% tolerance, adjust as needed
            compliance_message = (f"This room **MEETS** the recommended mid-frequency reverberation time (Tmf) "
                                  f"of ≤ {target_tmf_value:.2f}s for a '{selected_room_type}'. Current calculated Tmf: {current_tmf:.2f}s.")
            meets_recommendation = True
        else:
            compliance_message = (f"This room **DOES NOT MEET** the recommended mid-frequency reverberation time (Tmf) "
                                  f"of ≤ {target_tmf_value:.2f}s for a '{selected_room_type}'. Current calculated Tmf: {current_tmf:.2f}s.")
            meets_recommendation = False
elif volume_m3 <= 1e-6:
    compliance_message = "Room volume is zero or invalid. Please enter valid dimensions."
    meets_recommendation = False 
else:
    compliance_message = "No specific AS/NZS 2107 room type selected for Tmf compliance check."

# --- Output Section ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Room Sketch")
    fig_3d, ax_3d = plt.subplots(figsize=(6,5))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    if volume_m3 > 1e-6:
        ax_3d.set_title(f"Room: {L_calc:.1f}m x {W_calc:.1f}m x {H_calc:.1f}m")
        v = np.array([[0,0,0],[L_calc,0,0],[L_calc,W_calc,0],[0,W_calc,0], [0,0,H_calc],[L_calc,0,H_calc],[L_calc,W_calc,H_calc],[0,W_calc,H_calc]])
        faces = [[v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]], [v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]], [v[1],v[2],v[6],v[5]], [v[0],v[3],v[7],v[4]]]
        ax_3d.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        ax_3d.set_xlabel('L'); ax_3d.set_ylabel('W'); ax_3d.set_zlabel('H')
        max_dim = max(L_calc, W_calc, H_calc, 1.0) # Ensure max_dim is at least 1 for proper scaling
        ax_3d.set_xlim([0, max_dim]); ax_3d.set_ylim([0, max_dim]); ax_3d.set_zlim([0, max_dim])
        ax_3d.view_init(elev=20, azim=30)
    else:
        ax_3d.set_title("Invalid Dimensions for Sketch")
        ax_3d.text(0.5, 0.5, 0.5, "Enter valid room dimensions", ha='center', va='center', transform=ax_3d.transAxes)
    plt.tight_layout()
    st.pyplot(fig_3d)

with col2:
    st.subheader("Surface Configuration (Unfolded)")
    fig_2d_unfolded, ax_2d = plt.subplots(figsize=(6, 5))
    ax_2d.set_xticks([]); ax_2d.set_yticks([]); ax_2d.set_xlim(0, 4); ax_2d.set_ylim(-0.6, 3.1) 
    ax_2d.set_title("Room Surface Materials")
    # Use description_only for display
    rect_props = {
        "Ceiling": (1, 2, 1, 1, MATERIALS_DB.get(ceiling_mat,{}).get("description_only","N/A")), 
        "Front Wall": (1, 1, 1, 1, MATERIALS_DB.get(front_wall_mat,{}).get("description_only","N/A")),
        "Wall Left": (0, 1, 1, 1, MATERIALS_DB.get(wall_left_mat,{}).get("description_only","N/A")), 
        "Wall Right": (2, 1, 1, 1, MATERIALS_DB.get(wall_right_mat,{}).get("description_only","N/A")),
        "Rear Wall": (3, 1, 1, 1, MATERIALS_DB.get(wall_rear_mat,{}).get("description_only","N/A")), 
        "Floor": (1, 0, 1, 1, MATERIALS_DB.get(floor_mat,{}).get("description_only","N/A")),
    }
    if "Windows" in surfaces_details: rect_props["Windows"] = (0, -0.5, 1, 0.4, MATERIALS_DB.get(window_mat,{}).get("description_only","N/A"))
    
    for name, (x, y, w, h, mat_desc_only) in rect_props.items():
        rect = plt.Rectangle((x, y), w, h, edgecolor='black', facecolor='lightgray', alpha=0.7)
        ax_2d.add_patch(rect)
        display_mat_name = (mat_desc_only[:15] + '...') if len(mat_desc_only) > 18 else mat_desc_only
        ax_2d.text(x + w/2, y + h/2, f"{name}\n({display_mat_name})", ha='center', va='center', fontsize=7)
    ax_2d.axis('off'); plt.tight_layout(); st.pyplot(fig_2d_unfolded)

st.markdown("---")
st.subheader("Reverberation Time Results (RT60 in seconds) - Untreated Room")
results_data = {
    "Frequency (Hz)": FREQ_BANDS_HZ,
    "Sabine RT60 (s)": [f"{rt:.2f}" if not (math.isinf(rt) or math.isnan(rt)) else "N/A" for rt in rt_sabine_list],
    "Eyring RT60 (s)": [f"{rt:.2f}" if not (math.isinf(rt) or math.isnan(rt)) else "N/A" for rt in rt_eyring_list],
    "Fitzroy RT60 (s)": [f"{rt:.2f}" if not (math.isinf(rt) or math.isnan(rt)) else "N/A" for rt in rt_fitzroy_list],
    "Schroeder Freq (Hz)": [f"{f:.0f}" if not math.isnan(f) else "N/A" for f in sch_freq_list]
}
results_df = pd.DataFrame(results_data)
st.dataframe(results_df.set_index("Frequency (Hz)"))

if volume_m3 > 1e-6:
    rt_plot_df_data = {"Sabine": rt_sabine_list, "Eyring": rt_eyring_list, "Fitzroy": rt_fitzroy_list}
    plot_df = pd.DataFrame(rt_plot_df_data, index=pd.Index(FREQ_BANDS_HZ, name="Frequency (Hz)"))
    plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    st.line_chart(plot_df)
else:
    st.warning("Cannot plot RT chart as room volume is zero or invalid.")

st.markdown("---")
st.subheader("AS/NZS 2107:2016 Compliance & Treatment Estimation")
st.markdown(compliance_message)

if meets_recommendation is False and current_tmf != float('inf') and target_tmf_value is not None and current_tmf > target_tmf_value :
    st.markdown("---")
    st.subheader("Acoustic Treatment Estimation")
    st.markdown(f"The current Tmf ({current_tmf:.2f}s) is higher than the target ({target_tmf_value:.2f}s). "
                "Let's estimate the treatment needed.")

    st.markdown("**Choose a treatment material to estimate area:**")
    absorptive_material_options = [m for m in MATERIAL_DESCRIPTIONS if m != DEFAULT_MATERIAL_DISPLAY_NAME]
    if not absorptive_material_options: absorptive_material_options = [DEFAULT_MATERIAL_DISPLAY_NAME] 

    default_treatment_name = "50mm m/wool on solid (138)" # Check code from CSV
    if default_treatment_name not in absorptive_material_options: default_treatment_name = "Suspended acoustic tile (good) (8)" # Fallback
    if default_treatment_name not in absorptive_material_options and absorptive_material_options: default_treatment_name = absorptive_material_options[0] 
    
    default_treatment_idx = absorptive_material_options.index(default_treatment_name) if default_treatment_name in absorptive_material_options else 0

    treatment_material_key = st.selectbox(
        "Treatment Material Type", options=absorptive_material_options, index=default_treatment_idx,
        help="Select a material to be notionally added to walls/ceiling."
    )
    treatment_material_data = MATERIALS_DB.get(treatment_material_key, MATERIALS_DB[DEFAULT_MATERIAL_DISPLAY_NAME])
    treatment_material_coeffs = treatment_material_data.get("coeffs", [0.0]*6)
    treatment_material_desc_only = treatment_material_data.get("description_only", treatment_material_key)


    if volume_m3 > 1e-6 and target_tmf_value > 1e-6:
        A_target_for_tmf = (0.161 * volume_m3) / target_tmf_value
        current_A_500 = total_sabine_absorption_list[FREQ_BANDS_HZ.index(500)]
        current_A_1k = total_sabine_absorption_list[FREQ_BANDS_HZ.index(1000)]
        current_A_avg_mid_freq = (current_A_500 + current_A_1k) / 2.0
        additional_sabins_needed_at_mid_freq_avg = A_target_for_tmf - current_A_avg_mid_freq
        
        if additional_sabins_needed_at_mid_freq_avg > 0:
            alpha_treatment_500 = treatment_material_coeffs[FREQ_BANDS_HZ.index(500)]
            alpha_treatment_1k = treatment_material_coeffs[FREQ_BANDS_HZ.index(1000)]
            alpha_treatment_avg_mid_freq = (alpha_treatment_500 + alpha_treatment_1k) / 2.0

            if alpha_treatment_avg_mid_freq > 1e-3: 
                estimated_treatment_area_m2 = additional_sabins_needed_at_mid_freq_avg / alpha_treatment_avg_mid_freq
                st.success(f"Estimated **{estimated_treatment_area_m2:.2f} m²** of '{treatment_material_desc_only}' "
                           f"(avg mid-freq alpha: {alpha_treatment_avg_mid_freq:.2f}) "
                           f"is needed to reach the target Tmf of {target_tmf_value:.2f}s.")
                st.caption(f"(This adds ~{additional_sabins_needed_at_mid_freq_avg:.2f} Sabins at mid-frequencies).")

                absorption_added_by_treatment_list = [estimated_treatment_area_m2 * alpha for alpha in treatment_material_coeffs]
                total_sabine_absorption_treated_list = [
                    current_total_sab + added_treat_abs
                    for current_total_sab, added_treat_abs in zip(total_sabine_absorption_list, absorption_added_by_treatment_list)
                ]
                rt_sabine_treated_list = calculate_rt_sabine(volume_m3, total_sabine_absorption_treated_list)

                st.subheader("Estimated Reverberation Time WITH Treatment")
                treated_results_data = {
                    "Frequency (Hz)": FREQ_BANDS_HZ,
                    "Sabine RT60 (s) - Treated": [f"{rt:.2f}" if not (math.isinf(rt)or math.isnan(rt)) else "N/A" for rt in rt_sabine_treated_list],
                }
                treated_results_df = pd.DataFrame(treated_results_data).set_index("Frequency (Hz)")
                st.dataframe(treated_results_df)

                plot_treated_df = pd.DataFrame({
                    "Sabine (Untreated)": rt_sabine_list,
                    "Sabine (Treated)": rt_sabine_treated_list
                }, index=pd.Index(FREQ_BANDS_HZ, name="Frequency (Hz)")).replace([np.inf, -np.inf], np.nan)
                st.line_chart(plot_treated_df)

                treated_tmf_500_val = rt_sabine_treated_list[FREQ_BANDS_HZ.index(500)]
                treated_tmf_1k_val = rt_sabine_treated_list[FREQ_BANDS_HZ.index(1000)]
                if not (math.isinf(treated_tmf_500_val) or math.isnan(treated_tmf_500_val) or math.isinf(treated_tmf_1k_val) or math.isnan(treated_tmf_1k_val)):
                    estimated_tmf_after_treatment = (treated_tmf_500_val + treated_tmf_1k_val) / 2.0
                    st.caption(f"Estimated Tmf after treatment: {estimated_tmf_after_treatment:.2f}s")
                else:
                    st.caption("Estimated Tmf after treatment could not be precisely determined due to high RT values.")
            else:
                st.warning(f"The selected treatment material '{treatment_material_desc_only}' has very low "
                           f"(or zero: {alpha_treatment_avg_mid_freq:.3f}) "
                           f"average absorption at mid-frequencies. Please choose a more effective material for estimation.")
        elif target_tmf_value is not None and additional_sabins_needed_at_mid_freq_avg <=0 : 
            st.info("No additional absorption seems to be needed to meet the Tmf target based on average mid-frequency values, or the room is already below target. "
                    "Consider specific frequency band RTs if intelligibility or other acoustic issues persist.")
    elif volume_m3 <= 1e-6:
        st.info("Cannot calculate treatment as room volume is zero or invalid.")
    elif target_tmf_value is None: # Should be caught by outer if, but as safeguard
         st.info("No target Tmf selected, cannot estimate treatment.")
    else: # other unforeseen cases
        st.info("Could not calculate treatment requirement (e.g. target RT or volume invalid).")


st.markdown("---")
st.subheader("Method Descriptions")
st.markdown("""
- **Sabine Formula:** Classic formula, best for rooms with evenly distributed absorption and diffuse sound fields. Assumes low average absorption.
- **Eyring Formula:** Modification of Sabine, generally more accurate for rooms with higher average absorption coefficients.
- **Fitzroy Formula:** Considers non-uniform absorption distribution across opposing pairs of surfaces. (Note: Simplifications made for window area).
- **Schroeder Frequency:** The approximate frequency above which the modal density of a room is high enough for statistical room acoustics to be generally valid.
- **Tmf (Mid-frequency Reverberation Time):** Typically the arithmetic average of the reverberation times in the 500 Hz and 1 kHz octave bands. Used in AS/NZS 2107:2016 for many room type recommendations.
""")

st.caption(f"Room Volume: {volume_m3:.2f} m³")
st.caption(f"Total Main Surface Area (adjusted for windows): {total_surface_area_m2:.2f} m²")

with st.expander("View Selected Material Coefficients (Untreated Room)"):
    for surf_name, details in surfaces_details.items():
        # details["material"] is the display name key like "Material Name (Code)"
        mat_data = MATERIALS_DB.get(details["material"], MATERIALS_DB[DEFAULT_MATERIAL_DISPLAY_NAME])
        coeffs = mat_data.get("coeffs", [0.0]*6)
        coeffs_str = ", ".join([f"{c:.2f}" for c in coeffs])
        mat_desc_only = mat_data.get('description_only', details["material"]) # Fallback to full key if desc_only missing
        st.text(f"{details['name']} ({details['area']:.2f}m², {mat_desc_only}): {coeffs_str} (for {', '.join(FREQ_BANDS_STR)})")
        
# --- Footer ---
st.markdown("---")
# Using HTML for right alignment and mailto link
footer_html = """
<div style="text-align: right;">
  <p>Contact: <a href="mailto:bruno.marion@sydney.edu.au">bruno.marion@sydney.edu.au</a></p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)