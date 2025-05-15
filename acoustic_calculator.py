import math
import numpy as np

# Frequencies must match the order in material_parser
FREQ_BANDS_HZ = [125, 250, 500, 1000, 2000, 4000]

def calculate_rt_sabine(volume_m3, total_absorption_sabins_list):
    """ Total_absorption_sabins_list: list of total absorption (A_sabine) for each freq band """
    rt60 = []
    if volume_m3 <= 1e-6:
        return [float('inf')] * len(total_absorption_sabins_list)
    for A_sab in total_absorption_sabins_list:
        if A_sab <= 1e-6: 
            rt60.append(float('inf')) 
        else:
            rt60.append((0.161 * volume_m3) / A_sab)
    return rt60

def calculate_rt_eyring(volume_m3, total_surface_area_m2, avg_alpha_list, air_absorption_term_list):
    """
    avg_alpha_list: list of average absorption coefficient (surfaces only) for each freq band
    air_absorption_term_list: list of 4mV term for each freq band
    """
    rt60 = []
    if total_surface_area_m2 <= 1e-6 or volume_m3 <= 1e-6:
        return [float('inf')] * len(avg_alpha_list)

    for i, alpha_avg_surf in enumerate(avg_alpha_list):
        # Ensure alpha_avg_surf is a float and within range for log
        try:
            alpha_val = float(alpha_avg_surf)
        except (ValueError, TypeError):
            alpha_val = 0.0 # Default if not convertible
            
        if alpha_val >= 1.0: 
            alpha_val = 0.99999 
        elif alpha_val < 0.0: # Should not happen if clamped in parser
            alpha_val = 0.0

        # Surface part of Eyring absorption
        if (1.0 - alpha_val) <= 1e-9: # Avoid log(0) or log(negative)
            A_eyring_surface_part = float('inf') # Effectively infinite absorption from surfaces
        else:
            A_eyring_surface_part = -total_surface_area_m2 * math.log(1.0 - alpha_val)
        
        # Total Eyring absorption including air
        A_eyring_total = A_eyring_surface_part + air_absorption_term_list[i]
        
        if A_eyring_total <= 1e-6 or A_eyring_total == float('inf'):
            rt60.append(float('inf'))
        else:
            rt60.append((0.161 * volume_m3) / A_eyring_total)
    return rt60

def calculate_rt_fitzroy(volume_m3, L, W, H, surface_material_data, materials_db):
    """
    surface_material_data: dict like {'Ceiling_Material': 'Material Name (Code)', ...}
    materials_db: The parsed material database.
    L, W, H: room dimensions
    """
    rt60_fitzroy = [0.0] * len(FREQ_BANDS_HZ)
    if volume_m3 <= 1e-6:
        return [float('inf')] * len(FREQ_BANDS_HZ)

    # Gross areas of main surfaces (Fitzroy typically uses these)
    area_LW = L * W # Ceiling/Floor
    area_WH = W * H # Front/Rear
    area_LH = L * H # Left/Right

    total_room_surface_area = 2 * (area_LW + area_WH + area_LH)
    if total_room_surface_area == 0:
        return [float('inf')] * len(FREQ_BANDS_HZ)

    # Areas of PAIRS:
    Sx_pair_area = 2 * area_WH  # Front and Rear walls
    Sy_pair_area = 2 * area_LH  # Left and Right walls
    Sz_pair_area = 2 * area_LW  # Ceiling and Floor
    
    default_coeffs = [0.0] * len(FREQ_BANDS_HZ)
    default_material_key = "No element selected (0)" # Ensure this key exists in materials_db

    for band_idx in range(len(FREQ_BANDS_HZ)):
        # Get alpha for each surface at the current frequency band
        # Use .get(materials_db.get(DEFAULT_KEY)) to ensure a fallback if material key is missing
        alpha_ceil  = materials_db.get(surface_material_data['Ceiling_Material'], materials_db.get(default_material_key, {"coeffs":default_coeffs})).get('coeffs', default_coeffs)[band_idx]
        alpha_floor = materials_db.get(surface_material_data['Floor_Material'], materials_db.get(default_material_key, {"coeffs":default_coeffs})).get('coeffs', default_coeffs)[band_idx]
        alpha_front = materials_db.get(surface_material_data['FrontWall_Material'], materials_db.get(default_material_key, {"coeffs":default_coeffs})).get('coeffs', default_coeffs)[band_idx]
        alpha_rear  = materials_db.get(surface_material_data['RearWall_Material'], materials_db.get(default_material_key, {"coeffs":default_coeffs})).get('coeffs', default_coeffs)[band_idx]
        alpha_left  = materials_db.get(surface_material_data['LeftWall_Material'], materials_db.get(default_material_key, {"coeffs":default_coeffs})).get('coeffs', default_coeffs)[band_idx]
        alpha_right = materials_db.get(surface_material_data['RightWall_Material'], materials_db.get(default_material_key, {"coeffs":default_coeffs})).get('coeffs', default_coeffs)[band_idx]

        # Average absorption for each pair of surfaces
        alpha_x_pair_avg = (area_WH * alpha_front + area_WH * alpha_rear) / Sx_pair_area if Sx_pair_area > 0 else 1.0
        alpha_y_pair_avg = (area_LH * alpha_left + area_LH * alpha_right) / Sy_pair_area if Sy_pair_area > 0 else 1.0
        alpha_z_pair_avg = (area_LW * alpha_ceil + area_LW * alpha_floor) / Sz_pair_area if Sz_pair_area > 0 else 1.0
        
        # Avoid division by zero if average alpha is 0 for a pair
        term_x = (Sx_pair_area / alpha_x_pair_avg) if alpha_x_pair_avg > 1e-6 else (Sx_pair_area / 1e-6)
        term_y = (Sy_pair_area / alpha_y_pair_avg) if alpha_y_pair_avg > 1e-6 else (Sy_pair_area / 1e-6)
        term_z = (Sz_pair_area / alpha_z_pair_avg) if alpha_z_pair_avg > 1e-6 else (Sz_pair_area / 1e-6)

        if total_room_surface_area**2 < 1e-6 :
             rt60_fitzroy[band_idx] = float('inf')
        else:
            rt_val = (0.161 * volume_m3 / (total_room_surface_area**2)) * (term_x + term_y + term_z)
            rt60_fitzroy[band_idx] = rt_val
            
    return rt60_fitzroy


def calculate_schroeder_freq(rt60_list, volume_m3):
    """ rt60_list: list of RT60 values (e.g., Sabine) for each freq band """
    sch_freq = []
    if volume_m3 <= 1e-6:
        return [float('nan')] * len(rt60_list) 
        
    for rt in rt60_list:
        # Ensure rt is a valid number for sqrt
        rt_val = float('inf')
        try:
            rt_val = float(rt)
        except (ValueError, TypeError):
            pass # rt_val remains inf

        if rt_val <= 1e-6 or rt_val == float('inf') or math.isnan(rt_val) or volume_m3 <= 1e-6:
            sch_freq.append(float('nan')) 
        else:
            try:
                val = 2000 * math.sqrt(rt_val / volume_m3)
                sch_freq.append(val)
            except (ValueError, TypeError): 
                 sch_freq.append(float('nan'))
    return sch_freq