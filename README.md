# Yazid-Askar-Dissertation-Codes
 A Python-based bottom-up model for my MSc project, "Energy Electronics in the UK"

# Energy Electronics in the UK: A Bottom-Up Simulation Model

A Python-based bottom-up model to close the "granularity gap" in UK residential energy data by estimating appliance-level electricity consumption and CO₂ emissions. This repository contains the code developed for the MSc Individual Project at King's College London by Yazid Askar.

# Project Goal

The United Kingdom's 2050 net-zero ambition is challenged by a critical "granularity gap" in its residential energy data. Official statistics lack the device-specific detail required for effective appliance regulation. This project was designed to close this evidence gap.

# How It Works

This project employs a bottom-up engineering methodology, implemented in Python, to quantify the operational electricity consumption and carbon emissions of the 26 most common household appliances in the UK.

The core model calculates national energy demand by aggregating the consumption of individual appliances using three key inputs:
* **Device Stock:** The number of units for each appliance installed in UK homes.
* **Power Ratings:** The active and standby power draw for each device.
* **Usage Patterns:** The average daily active usage time.

## Repository Structure

This repository contains several Python scripts:
* `FComputing.py`, `FKitchen.py`, `fgame.py`, `personal.py`: These are the four category-level scripts designed to run analyses on those specific groups of appliances.
* `final.py`: This is the main script that combines the data from all categories to calculate the aggregate results for all 26 appliances.

## Running the Model

The model is designed to be interactive, allowing you to test different scenarios by modifying the input parameters.

1.  **Start with the Category-Level Scripts:** Open one of the four category-specific Python files (e.g., `FKitchen.py`).

2.  **Modify Input Parameters:** At the top of each file, you will find the input parameters (device stock, power ratings, usage times) explicitly defined. You can change these values to see how they affect the results for that category.

    ```python
    # --- Input Parameters for Kitchen Appliances ---
    # (Modify these values to test different scenarios)

    kettle_stock = 27.0  # million units
    kettle_power = 3000  # Watts
    kettle_usage = 12    # minutes/day
    ```

3.  **Run the Final Code:** The `final.py` script is pre-loaded with the default parameters from the dissertation. You can run this file directly to reproduce the report's main findings.

    **Please Note:** The individual category scripts and the final script are not dynamically linked. If you make changes to the input parameters in a category file, you will need to manually update those parameters in the `final.py` script to see their effect on the national total.

## Adapting and Extending the Model

This codebase is designed to be a flexible and extensible tool.

* **Create Your Own Categories:** You can create your own category-level analysis by using one of the existing scripts (e.g., `FKitchen.py`) as a template. Simply copy the file, rename it, and modify the input parameter variables at the top to model a completely new set of devices.

* **Update or Change Benchmarks:** The model's validation functions can be updated. As new official data is released (e.g., future versions of ECUK), you can input the new figures to re-validate the model. You can also adapt the code to benchmark the results against an entirely different data source.

## Citation

This code is part of the dissertation "Energy Electronics in the UK" submitted in partial fulfilment of the requirements of the MSc Project Module at King's College London.

> Askar, Y. (2025). *Energy Electronics in the UK*. MSc Project Report, Engineering Department, King's College London. Supervised by Dr. Mohit Arora.
