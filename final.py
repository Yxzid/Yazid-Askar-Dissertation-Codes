import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.size': 10})
mpl.rcParams['font.family'] = 'DejaVu Sans'

# ── 1. INPUT ──────────────────────────────
# Kitchen appliances
kitchen_data = [
    ("Fridge/Freezer", 150, 480, 15, 21.03), ("Kettle", 3000, 12, 0.00, 27.00),
    ("Dishwasher", 800, 51, 0.50, 14.2), ("Air Fryer", 1500, 25, 0.5, 16.50),
    ("Electric Hob", 1800, 20, 1, 14.8), ("Microwave", 1000, 11, 2, 25.60),
    ("Coffee Machine", 1400, 3, 0.77, 16.20), ("Rice Cooker", 700, 30, 0.00, 4.50),
    ("Toaster", 900, 9, 0.00, 21.90), ("Washing Machine", 700, 34, 1.00, 27.50),
    ("Electric Oven", 550, 35, 2, 20.9)
]
kitchen_df = pd.DataFrame(kitchen_data, columns=["Device", "Pmid", "T_active",
                                                 "P_standby", "Units_mil"])
kitchen_df["Category"] = "Kitchen"

# Office devices
office_data = [
    ("Wifi Router", 10.88, 1440, 0, 26.98), ("Desktop Computer", 100, 138, 0.5, 3.84),
    ("Laptop", 42.0, 219, 0.5, 31.862), ("Monitor", 21.4, 138, 0.3, 19.2),
    ("Projector", 225, 30, 0.3, 0.6), ("Printer", 26.64, 0.15, 1.4, 8.11)
]
office_df = pd.DataFrame(office_data, columns=["Device", "Pmid", "T_active",
                                               "P_standby", "Units_mil"])
office_df["Category"] = "Office"

# Personal devices
personal_data = [
    ("Smartphones", 5.0, 165.5, 0.04, 64.93),
    ("Feature Phone", 1.75, 112.8, 0.075, 0.4101),
    ("Tablets", 12, 171.8, 0.05, 34.96),
    ("Smart Speaker", 2.4, 36, 1.3, 9.37)
]
personal_df = pd.DataFrame(personal_data, columns=["Device", "Pmid", "T_active",
                                                   "P_standby", "Units_mil"])
personal_df["Category"] = "Personal"

# Entertainment devices
entertainment_data = [
    ("Gaming Console (Handheld)", 9.8, 101.8, 0.08, 2.44),
    ("Gaming Console (Home)", 214.3, 150, 0.31, 9.77),
    ("TV (LCD)", 50.4, 270, 0.5, 52.3), ("TV (OLED)", 81, 270, 0.5, 1.05),
    ("Set-Top Box", 20.1, 196, 0.4, 26.049)
]
entertainment_df = pd.DataFrame(entertainment_data, columns=["Device", "Pmid",
                                "T_active", "P_standby", "Units_mil"])
entertainment_df["Category"] = "Entertainment"

# Combine all devices
combined_df = pd.concat([kitchen_df, office_df, personal_df, entertainment_df],
                        ignore_index=True)
combined_df["T_standby"] = 1440 - combined_df["T_active"]

# ======== ENERGY & EMISSIONS CALC =================================
CARBON = 0.22535  # kgCO2/kWh

# Energy calculations
combined_df["kWh_hh_active"]  = (combined_df["Pmid"] / 1000) * \
                                (combined_df["T_active"] / 60) * 365
combined_df["kWh_hh_standby"] = (combined_df["P_standby"] / 1000) * \
                                (combined_df["T_standby"] / 60) * 365
combined_df["kWh_hh"]         = combined_df["kWh_hh_active"] + \
                                combined_df["kWh_hh_standby"]

combined_df["kWh_hh_active_min"] = (combined_df["Pmid"]*0.9 / 1000) * \
                                   (combined_df["T_active"]/60) * 365
combined_df["kWh_hh_active_max"] = (combined_df["Pmid"]*1.1 / 1000) * \
                                   (combined_df["T_active"]/60) * 365

combined_df["GWh_nat_active"]  = combined_df["kWh_hh_active"]  * \
                                 combined_df["Units_mil"]
combined_df["GWh_nat_standby"] = combined_df["kWh_hh_standby"] * \
                                 combined_df["Units_mil"]
combined_df["GWh_nat"]         = combined_df["GWh_nat_active"] + \
                                 combined_df["GWh_nat_standby"]
combined_df["GWh_nat_active_min"] = combined_df["kWh_hh_active_min"] * \
                                    combined_df["Units_mil"]
combined_df["GWh_nat_active_max"] = combined_df["kWh_hh_active_max"] * \
                                    combined_df["Units_mil"]

# Emissions calculations
combined_df["kt_nat"] = combined_df["GWh_nat"] * CARBON
combined_df["kt_nat_active"] = combined_df["GWh_nat_active"] * CARBON
combined_df["kt_nat_standby"] = combined_df["GWh_nat_standby"] * CARBON
combined_df["kt_nat_active_min"] = combined_df["GWh_nat_active_min"] * CARBON
combined_df["kt_nat_active_max"] = combined_df["GWh_nat_active_max"] * CARBON

total_energy_gwh   = combined_df["GWh_nat"].sum()
total_emissions_kt = total_energy_gwh * CARBON

# ======== STACKED BAR PLOT FUNCTION ===============================
def plot_stacked_energy(df, title, ylabel, nat=False, top_n=26):
    """Stacked bars with error bars; labels clear error tops."""
    d = df.sort_values("GWh_nat" if nat else "kWh_hh", ascending=False)
    if top_n:
        d = d.head(top_n)

    if nat:
        active = d["GWh_nat_active"];    standby = d["GWh_nat_standby"]
        active_min = d["GWh_nat_active_min"]; active_max = d["GWh_nat_active_max"]
    else:
        active = d["kWh_hh_active"];     standby = d["kWh_hh_standby"]
        active_min = d["kWh_hh_active_min"]; active_max = d["kWh_hh_active_max"]

    total       = active + standby
    y_err_lower = active - active_min
    y_err_upper = active_max - active

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(d.Device, active,   label="Active",   color="#1f77b4")
    ax.bar(d.Device, standby, bottom=active, label="Stand-by", color="#aec6cf")

    ax.errorbar(d.Device, total, yerr=[y_err_lower, y_err_upper],
                fmt='none', ecolor='k', capsize=4,
                label="±10 % Active Power")

    # ▲ pad y-axis 10 % above tallest error bar
    total_plus_err = total + y_err_upper
    ax.set_ylim(0, total_plus_err.max()*1.10)

    # ▲ label above error-bar tip + 2 % padding
    for i, (tot, err) in enumerate(zip(total, y_err_upper)):
        ax.text(i,
                tot + err + total_plus_err.max()*0.02,
                f"{tot:,.0f}" if nat else f"{tot:,.1f}",
                ha="center", va="bottom", fontsize=9,
                zorder=3, clip_on=False)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(len(d.Device)))
    ax.set_xticklabels(d.Device, rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ======== STACKED EMISSIONS PLOT FUNCTION =========================
def plot_stacked_emissions(df, title, ylabel, top_n=26):
    """Stacked emissions bars with error bars and clear labels."""
    d = df.sort_values("kt_nat", ascending=False).head(top_n)
    
    active = d["kt_nat_active"]
    standby = d["kt_nat_standby"]
    total = active + standby
    
    # Error bars calculation (emissions)
    y_err_lower = active - d["kt_nat_active_min"]
    y_err_upper = d["kt_nat_active_max"] - active

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(d.Device, active, label="Active", color="#d62728")        # Brick red
    ax.bar(d.Device, standby, bottom=active, label="Stand-by", color="#f7a4a4")  # Light red
    
    ax.errorbar(d.Device, total, yerr=[y_err_lower, y_err_upper],
                fmt='none', ecolor='k', capsize=4,
                label="±10 % Active Power")
    
    # Adjust y-axis limits
    total_plus_err = total + y_err_upper
    ax.set_ylim(0, total_plus_err.max() * 1.10)
    
    # Add labels above error bars
    for i, (tot, err) in enumerate(zip(total, y_err_upper)):
        ax.text(i,
                tot + err + total_plus_err.max()*0.02,
                f"{tot:,.0f}",
                ha="center", va="bottom", fontsize=9,
                zorder=3, clip_on=False)
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(len(d.Device)))
    ax.set_xticklabels(d.Device, rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ======== PLOTS ========================================
# Energy plots
plot_stacked_energy(combined_df,
                    "UK National Appliance Electricity Demand (2025)",
                    "GWh/year", nat=True)

plot_stacked_energy(combined_df,
                    "Household Appliance Electricity Consumption (2025)",
                    "kWh/year", nat=False)

# Emissions plot
plot_stacked_emissions(
    combined_df,
    "UK National Appliance CO₂ Emissions (2025)",
    "kt CO₂e/year"
)

# ======== PLOTS: CATEGORY & DEVICE DONUTS, KPI CARDS ==============

# ----------  donut helper ---------------------------------
def donut_chart(series, title, label_fmt):
    """
    series     : pd.Series   (index = labels, values = numbers)
    title      : str         chart title
    label_fmt  : fn(label, value, pct) -> str  text for legend
    """
    total = series.sum()
    pct   = series / total * 100
    legend_labels = [label_fmt(lbl, val, p) for lbl, val, p
                     in zip(series.index, series.values, pct)]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges = ax.pie(series.values, startangle=90)[0]

    # donut hole
    ax.add_artist(plt.Circle((0, 0), 0.45, fc="white"))
    ax.axis("equal")
    ax.set_title(title, pad=20, fontsize=14)

    ax.legend(wedges, legend_labels,
              loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=10, frameon=False)
    fig.tight_layout()
    plt.show()


# ---------- CATEGORY-LEVEL DONUTS --------------------------------
cat_energy = combined_df.groupby("Category")["GWh_nat"].sum()
cat_emis   = combined_df.groupby("Category")["kt_nat"].sum()

# a) Energy share (%)
donut_chart(cat_energy,
            "UK Energy Consumption by Category",
            lambda lbl, v, p: f"{lbl} – {p:.1f}%")

# b) Emissions share (%)
donut_chart(cat_emis,
            "UK CO₂ Emissions by Category",
            lambda lbl, v, p: f"{lbl} – {p:.1f}%")

# c) Energy absolute (GWh)
donut_chart(cat_energy,
            "Annual Energy Consumption by Category (GWh)",
            lambda lbl, v, p: f"{lbl} – {v:,.0f} GWh")

# d) Emissions absolute (kt)
donut_chart(cat_emis,
            "Annual CO₂ Emissions by Category (kt CO₂e)",
            lambda lbl, v, p: f"{lbl} – {v:,.0f} kt")


# ---------- DEVICE-LEVEL DONUTS  --------------
def donut_devices(cat):
    df = (combined_df[combined_df["Category"] == cat]
          .sort_values("GWh_nat", ascending=False))

    # group tail into “Others” if more than 5 devices
    if len(df) > 5:
        df = pd.concat([
            df.head(4),
            pd.DataFrame({"Device": ["Others"],
                          "GWh_nat": [df["GWh_nat"].iloc[4:].sum()]})
        ])

    donut_chart(df.set_index("Device")["GWh_nat"],
                f"{cat} Devices",
                lambda lbl, v, p: f"{lbl} – {p:.1f}%")

for cat in combined_df["Category"].unique():
    donut_devices(cat)


# --- helper: KPI rectangle --------------------------------------------------
def kpi_card(title, number, unit, fill):
    """Draw a rounded-rectangle KPI card filled with `fill` colour."""
    fig, ax = plt.subplots(figsize=(5, 2.7))
    ax.axis("off")

    # full-axes rounded rectangle
    rect = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=fill,
        zorder=0
    )
    ax.add_patch(rect)


    ax.text(0.5, 0.65, f"{number:,.0f}",
            ha="center", va="center",
            fontsize=36, fontweight="bold", color="#000000",
            transform=ax.transAxes)

    ax.text(0.5, 0.28, unit,
            ha="center", va="center",
            fontsize=13, color="#000000",
            transform=ax.transAxes)

    fig.suptitle(title, y=0.98, fontsize=15, color="#000000")
    fig.tight_layout()
    plt.show()

# --- KPI: total energy & emissions ---
# Energy card  (blue)
kpi_card("UK Residential Electronics Energy",
         total_energy_gwh, "GWh",
         fill="#d7e8ff")        # pastel blue

# Emissions card (red)
kpi_card("UK Residential Electronics Emissions",
         total_emissions_kt, "kt CO₂e",
         fill="#ffe3e3")        # pastel red

# ======== SENSITIVITY ANALYSIS (ENERGY + CO₂)  ====================

base_E = combined_df["GWh_nat"].sum()         # 56 600 GWh
base_C = base_E * CARBON                      # 12 760 kt

rows, params = [], ["Pmid", "T_active", "P_standby", "Units_mil"]

for _, r in combined_df.iterrows():
    dev_E0 = r["GWh_nat"]
    for p in params:
        for f in (1.1, 0.9):                  # +10 %, −10 %
            new_r        = r.copy()
            new_r[p]    *= f
            if p == "T_active":
                new_r["T_standby"] = 1440 - new_r["T_active"]

            act   = (new_r["Pmid"]/1000) * (new_r["T_active"]/60) * 365
            stb   = (new_r["P_standby"]/1000) * (new_r["T_standby"]/60) * 365
            new_E = (act + stb) * new_r["Units_mil"]

            ΔE = (base_E - dev_E0 + new_E) - base_E               # GWh
            ΔC = ΔE * CARBON                                      # kt

            rows.append({
                "Device":    r["Device"],
                "Category":  r["Category"],
                "Parameter": p,
                "ΔE_GWh":    abs(ΔE),
                "ΔE_%":      abs(ΔE) / base_E * 100,
                "ΔC_kt":     abs(ΔC),
                "ΔC_%":      abs(ΔC) / base_C * 100
            })

# maximum swing (±10 %) for each device-parameter
sens = (pd.DataFrame(rows)
        .sort_values("ΔE_GWh", ascending=False)
        .drop_duplicates(subset=["Device", "Parameter"]))

top10   = sens.nlargest(10, "ΔE_GWh")
top10_C = top10.set_index(["Device", "Parameter"]).loc[
             sens.set_index(["Device", "Parameter"]).nlargest(10, "ΔC_kt").index
         ].reset_index()

# --- bar-plot helper ------------------------------------------------
def barplot(df, value_col, pct_col, title, xlabel, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(14, 8))

    ylabels = df["Device"] + " (" + df["Parameter"] + ")"
    bars    = ax.barh(ylabels, df[value_col], color=color)

    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(axis="x", alpha=0.3)

    # pad axis so labels fit
    x_max = df[value_col].max()
    ax.set_xlim(0, x_max * 1.15)        # 15 % head-room

    for bar, pct in zip(bars, df[pct_col]):
        w = bar.get_width()
        ax.text(w + x_max*0.02,                 # 2 % inside the padded area
                bar.get_y() + bar.get_height()/2,
                f"{w:,.0f}  ({pct:.2f} %)",
                ha="left", va="center", fontsize=9)

    fig.tight_layout()
    plt.show()

# --- plot energy & emissions ---------------------------------------
barplot(top10,   "ΔE_GWh", "ΔE_%",
        "Top-10 Most Sensitive Parameters – Energy (±10 %)",
        "Maximum Change (GWh)")

barplot(top10_C, "ΔC_kt",  "ΔC_%",
        "Top-10 Most Sensitive Parameters – CO₂ (±10 %)",
        "Maximum Change (kt CO₂e)")

# ======== TERMINAL OUTPUT ====================================================
print("="*70)
print(f"UK TOTAL ENERGY CONSUMPTION: {base_E:,.1f} GWh")
print(f"UK TOTAL EMISSIONS:          {base_C:,.1f} kt CO2e")
print("="*70)

print("\nTOP 10 MOST IMPACTFUL PARAMETERS (ENERGY):")
print(top10[["Device","Category","Parameter","ΔE_GWh","ΔE_%"]]
      .to_string(index=False, formatters={"ΔE_GWh": "{:,.1f}".format,
                                          "ΔE_%":   "{:.2f}".format}))

print("\nTOP 10 MOST IMPACTFUL PARAMETERS (CO₂):")
print(top10_C[["Device", "Category", "Parameter", "ΔC_kt", "ΔC_%"]]
      .to_string(index=False,
                 formatters={"ΔC_kt": "{:,.1f}".format,
                             "ΔC_%":  "{:.2f}".format}))

print("\nCATEGORY ENERGY DISTRIBUTION:")
print(cat_energy.to_string())
print()