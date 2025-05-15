import pandas as pd
from io import StringIO

# Standard octave bands
FREQ_BANDS_HZ = [125, 250, 500, 1000, 2000, 4000]
FREQ_BANDS_STR_CSV_MATCH = ["125 Hz", "250 Hz", "500 Hz", "1k Hz", "2k Hz", "4k Hz"] # Matches CSV headers

# For display/internal use
FREQ_BANDS_STR_DISPLAY = ["125 Hz", "250 Hz", "500 Hz", "1 kHz", "2 kHz", "4 kHz"]

# Expose FREQ_BANDS_STR_DISPLAY as FREQ_BANDS_STR for external use by app.py
FREQ_BANDS_STR = FREQ_BANDS_STR_DISPLAY


def parse_materials_csv(csv_content_string):
    """
    Parses the material absorption CSV content with the new "Surface Type" column.
    Returns a dictionary of materials:
    { "Description (Code)": {"coeffs": [c1, c2, ...], "category": "Surface Type", "code": "X", "description_only": "Description"}, ... }
    and a list of all material descriptions (formatted as "Description (Code)") for dropdowns.
    """
    materials_data = {}
    material_display_list = []

    try:
        lines = csv_content_string.splitlines()
        header_row_index = -1
        expected_header_start = "Surface Type,Code,Description"
        
        # Search for the header, trying a few common starting lines if the primary one isn't first
        possible_starts = [
            "Surface Type,Code,Description",
            ",Absorption Coefficient Listing,,Absorption Coefficients" # This is often line 1
        ]
        
        found_main_header = False
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith(possible_starts[0]): # Direct match for "Surface Type,Code,Description"
                header_row_index = i
                found_main_header = True
                break
            elif stripped_line.startswith(possible_starts[1]): # Match for ",Absorption Coefficient Listing..."
                # The actual data header is usually 1 or 2 lines below this
                if len(lines) > i + 1 and lines[i+1].strip().startswith(possible_starts[0]):
                    header_row_index = i + 1
                    found_main_header = True
                    break
                # Add more checks if there can be more empty lines between the title and the actual header
                elif len(lines) > i + 2 and lines[i+2].strip().startswith(possible_starts[0]):
                    header_row_index = i + 2
                    found_main_header = True
                    break


        if not found_main_header:
            # If still not found, raise an error or try a more lenient search if needed
            raise ValueError(f"Could not find the CSV data header row starting with '{possible_starts[0]}'. Please check CSV format.")

        csv_for_pandas = "\n".join(lines[header_row_index:])
        df = pd.read_csv(StringIO(csv_for_pandas))
        df.columns = [col.strip() for col in df.columns]

    except Exception as e:
        print(f"Error reading or parsing CSV with pandas: {e}")
        # Fallback
        default_display_name_fb = "No element selected (0)"
        materials_data[default_display_name_fb] = {
            "coeffs": [0.0] * len(FREQ_BANDS_HZ), 
            "category": "Default", 
            "code": "0",
            "description_only": "No element selected"
        }
        material_display_list.append(default_display_name_fb)
        return materials_data, material_display_list

    default_desc = "No element selected"
    default_code = "0"
    default_display_name = f"{default_desc} ({default_code})"
    
    materials_data[default_display_name] = {
        "coeffs": [0.0] * len(FREQ_BANDS_HZ),
        "category": "Default", 
        "code": default_code,
        "description_only": default_desc
    }

    for index, row in df.iterrows():
        try:
            surface_type = str(row.get("Surface Type", "Uncategorized")).strip()
            code_val_raw = row.get("Code")
            description_val = str(row.get("Description", "")).strip()

            # Handle code: might be float (e.g., "0.0"), string, or NaN
            if pd.isna(code_val_raw):
                code_val = ""
            else:
                try: # Attempt to convert to int then str to remove ".0"
                    code_val = str(int(float(str(code_val_raw).strip())))
                except ValueError: # If not a number, use as is (stripped)
                    code_val = str(code_val_raw).strip()
            
            # Skip rows that are category headers or lack essential data
            is_category_header_row = (surface_type == description_val and (not code_val or code_val == "nan")) or \
                                     (description_val == "Description" and code_val == "Code") # Skip "Code,Description" subheaders

            if not code_val or not description_val or is_category_header_row :
                if description_val == "Air" and surface_type == "Ceilings": # Allow 'Air' to be skipped later
                    pass
                else:
                    continue
            
            if description_val == "Air" and surface_type == "Ceilings":
                continue

            coeffs = []
            for band_str_csv in FREQ_BANDS_STR_CSV_MATCH:
                val_str = str(row.get(band_str_csv, "0.0")).strip()
                try:
                    val_str = val_str.replace(',', '.')
                    coeff = float(val_str) if val_str else 0.0
                    coeffs.append(max(0.0, min(1.0, coeff))) 
                except ValueError:
                    coeffs.append(0.0)
            
            display_name = f"{description_val} ({code_val})"
            
            if code_val == default_code and description_val == default_desc:
                materials_data[display_name]["category"] = surface_type # Update category for the pre-added default
                # Don't add to material_display_list here, handled at the end
                continue

            if display_name not in material_display_list: # Add if not the default one already processed
                 material_display_list.append(display_name)

            materials_data[display_name] = {
                "coeffs": coeffs,
                "category": surface_type,
                "code": code_val,
                "description_only": description_val 
            }

        except Exception as e:
            # print(f"Skipping row {index} due to error: {e}, Row data: {row.to_dict()}") # For debugging
            continue
            
    material_display_list.sort()
    
    if default_display_name in material_display_list:
        material_display_list.remove(default_display_name)
    material_display_list.insert(0, default_display_name)
    
    if default_display_name not in materials_data: # Should be there from init, but as a safeguard
        materials_data[default_display_name] = {
            "coeffs": [0.0] * len(FREQ_BANDS_HZ),
            "category": "Default", "code": "0", "description_only": "No element selected"
        }

    return materials_data, material_display_list

# Air absorption coefficient 'm' (Nepers/meter) at 20°C, 50% RH
AIR_ABSORPTION_M_VALUES = {
    125: 0.0003, 250: 0.0006, 500: 0.0011,
    1000: 0.0019, 2000: 0.0037, 4000: 0.0086
}
AIR_ABSORPTION_M_LIST = [AIR_ABSORPTION_M_VALUES[f] for f in FREQ_BANDS_HZ]