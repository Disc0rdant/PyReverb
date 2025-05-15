# Room Reverberation Time Calculator

A simple web-based calculator to estimate room reverberation times using various acoustic formulas. Built with Python and Streamlit.

## Features

*   **Room Dimensions:** Input length, width, and height.
*   **Material Selection:** Choose materials for walls, ceiling, floor, and windows from an extensive list.
*   **Reverberation Time Calculations:**
    *   Sabine Formula
    *   Eyring Formula
    *   Fitzroy Formula
*   **Schroeder Frequency:** Estimated.
*   **AS/NZS 2107:2016 Compliance (Tmf):**
    *   Select a standard room type to get a target mid-frequency reverberation time (Tmf).
    *   Compares calculated Tmf against the standard's recommendation.
*   **Acoustic Treatment Estimation:** If Tmf is too high, estimates the area of a selected absorptive material needed to meet the target.

## Setup & Usage
**Run via https://pyreverb.streamlit.app/**

or

1.  **Clone the repository (or download the files):**
    ```bash
    # git clone [URL_OF_YOUR_REPO]
    # cd [REPO_NAME]
    ```
    Or, if you downloaded the files, navigate to the folder.

2.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas numpy matplotlib
    ```

3.  **Prepare Material Data:**
    *   Ensure the `reverberation_common_material_absorption_coefficient.csv` file is in the same directory as the Python scripts. This file contains the absorption coefficients for various materials.

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
    The calculator will open in your web browser.

## Files

*   `app.py`: The main Streamlit application script.
*   `material_parser.py`: Handles parsing the material absorption coefficient CSV file.
*   `acoustic_calculator.py`: Contains the core acoustic calculation functions.
*   `reverberation_common_material_absorption_coefficient.csv`: Data file for material acoustic properties.

## How it Works

The calculator uses the Sabine, Eyring, and Fitzroy formulas to predict reverberation times based on room volume, surface areas, and the absorption coefficients of selected materials. Air absorption is included. It also provides a basic compliance check against AS/NZS 2107:2016 Tmf targets and estimates treatment needs.

---

Feel free to contribute or report issues!

bruno.marion@sydney.edu.au
