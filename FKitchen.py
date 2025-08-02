import pandas as pd, numpy as np, matplotlib.pyplot as plt
plt.style.use("ggplot"); plt.rcParams.update({'font.size':10})

# ── 1. INPUT ──────────────────────────────
data = [
 # Device, Pmid(W),  T_active(min), P_stby(W), Units_mil
 ("Fridge/Freezer",   150, 480, 15, 21.03),             
 ("Kettle",         3000,   12, 0.00, 27.00),             
 ("Dishwasher",      800,  51, 0.50, 14.2),             
 ("Air Fryer",       1500,   25, 0.5, 16.50),             
 ("Electric Hob",    1800, 20, 1, 14.8),           
 ("Microwave",       1000, 11, 2, 25.60),             
 ("Coffee Machine", 1400,   3, 0.77, 16.20),
 ("Rice Cooker",     700,   30, 0.00,  4.50),
 ("Toaster",        900,   9, 0.00, 21.90),
 ("Washing Machine", 700,  34, 1.00, 27.50),             
 ("Electric Oven",  550,   35, 2, 20.9),             
]

df = pd.DataFrame(data, columns=["Device","Pmid","T_active","P_standby","Units_mil"])

# ── 2. ±10 % bands ──────────────────────────────────────────────────
df["Pmin"], df["Pmax"] = df.Pmid*0.9, df.Pmid*1.1
df["T_standby"]        = 1440 - df.T_active

# ── 3. Deterministic mid-case ───────────────────────────────────────
CARBON = 0.22535
df["kWh_hh_active_mid"] = df.Pmid/1000 * df.T_active/60 * 365
df["kWh_hh_standby"]    = df.P_standby/1000 * df.T_standby/60 * 365
df["kWh_hh"]            = df.eval("kWh_hh_active_mid + kWh_hh_standby")
df["GWh_nat"]           = df.kWh_hh * df.Units_mil
df["kgCO2_hh"]          = df.kWh_hh * CARBON
df["kt_nat"]            = df.GWh_nat * CARBON
df["kWh_hh_active_min"] = df.Pmin/1000 * df.T_active/60 * 365
df["kWh_hh_active_max"] = df.Pmax/1000 * df.T_active/60 * 365

# ── 4. Monte-Carlo ±10 % ───────────────────────────────────────────
N, rng = 10_000, np.random.default_rng(42)
rows=[]
for _,r in df.iterrows():
    P = rng.triangular(r.Pmin,r.Pmid,r.Pmax,N)/1000
    kwh = P*(r.T_active/60)*365 + r.P_standby/1000*(r.T_standby/60)*365
    rows.append(kwh)
mc=np.vstack(rows); total_nat=(mc*df.Units_mil.values[:,None]).sum(0)
df["P5"],df["P50"],df["P95"]=[np.percentile(mc,q,axis=1) for q in (5,50,95)]

# ── 5. Helper plot functions ────────────────────────────────────────
def stacked(d, ttl, yl, nat=False):
    d=d.copy(); key="GWh_nat" if nat else "kWh_hh"
    if nat:
        d["active"]=d.kWh_hh_active_mid*d.Units_mil
        d["stand"]=d.kWh_hh_standby   *d.Units_mil
        d["active_low"]=d.kWh_hh_active_min*d.Units_mil
        d["active_high"]=d.kWh_hh_active_max*d.Units_mil
    else:
        d["active"]=d.kWh_hh_active_mid
        d["stand"]=d.kWh_hh_standby
        d["active_low"]=d.kWh_hh_active_min
        d["active_high"]=d.kWh_hh_active_max
    d=d.sort_values(key,ascending=False)
    fig,ax=plt.subplots(figsize=(12,6))
    ax.bar(d.Device,d.active,color="#1f77b4",label="Active")
    ax.bar(d.Device,d.stand,bottom=d.active,color="#aec6cf",label="Stand-by")
    tot=d.active+d.stand
    
    # Calculate error bar positions
    y_err_lower = d.active - d.active_low
    y_err_upper = d.active_high - d.active
    ax.errorbar(d.Device,tot,
                yerr=[y_err_lower, y_err_upper],
                fmt='none',ecolor='k',capsize=4)
    
    # Calculate label position above error bars
    max_error = max(y_err_upper.max(), y_err_lower.max())
    label_height = tot + y_err_upper + max_error * 0.15
    
    # Add value labels above error bars
    for i, (dev, height) in enumerate(zip(d.Device, label_height)):
        value = tot.iloc[i]
        ax.text(i, height, 
                f'{value:,.0f}' if nat else f'{value:,.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    ax.set(title=ttl,ylabel=yl)
    ax.set_xticks(range(len(d.Device)))
    ax.set_xticklabels(d.Device,rotation=45,ha="right")
    ax.legend(); plt.tight_layout(); plt.show()

def carbon(d,col,ttl,yl,color,nat=False):
    v = d[col] if nat else d[col]
    d2=d.sort_values(col,ascending=False)
    fig,ax=plt.subplots(figsize=(12,6))
    bars=ax.bar(d2.Device,v.loc[d2.index],color=color)
    for b in bars:
        h=b.get_height()
        # Format based on magnitude
        fmt = f'{h:,.0f}' if h > 10 else f'{h:,.1f}'
        ax.text(b.get_x()+b.get_width()/2, h*1.01, 
                fmt, ha='center', va='bottom', fontsize=9)
    ax.set(title=ttl,ylabel=yl)
    ax.set_xticks(range(len(d2.Device)))
    ax.set_xticklabels(d2.Device,rotation=45,ha="right")
    plt.tight_layout()
    plt.show()

# ── 6. Plot suite ───────────────────────────────────────────────────
stacked(df,"Household kitchen electricity","kWh / hh·yr")
stacked(df,"UK kitchen electricity","GWh / yr",nat=True)
carbon(df,"kgCO2_hh","Household kitchen CO₂e","kg / hh·yr","#d62728")
carbon(df,"kt_nat","UK kitchen CO₂e","kt / yr","red",nat=True)

#  ──── MONTE CARLO PLOTS ───────────────────────────────────────────────────
plt.figure(figsize=(10,6))
plt.hist(total_nat/1000, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
plt.axvline(np.percentile(total_nat/1000, 5), color='red', linestyle='--', label='5th percentile')
plt.axvline(np.percentile(total_nat/1000, 95), color='blue', linestyle='--', label='95th percentile')
plt.title('Monte Carlo: Total UK Kitchen Energy Consumption')
plt.xlabel('Total National Energy (TWh/year)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.2)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
df_sorted = df.sort_values('P50', ascending=False)
devices = df_sorted.Device.values
for i, device in enumerate(devices):
    plt.errorbar(i, df_sorted['P50'].iloc[i], 
                 yerr=[[df_sorted['P50'].iloc[i] - df_sorted['P5'].iloc[i]], 
                       [df_sorted['P95'].iloc[i] - df_sorted['P50'].iloc[i]]],
                 fmt='o', color='black', capsize=5)
plt.xticks(range(len(devices)), devices, rotation=45, ha='right')
plt.ylabel('Household Energy (kWh/year)')
plt.title('Monte Carlo: Appliance Energy Consumption Ranges')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── 7. ECUK validation ─────────────────────────────────────────────
ecuk = {
    "Fridge/Freezer": 6019,
    "Kettle": 4843,
    "Dishwasher": 3502,
    "Washing Machine": 6773,
    "Microwave": 2507,
    "Electric Oven": 2008,
    "Electric Hob": 2657
}
val = df[df.Device.isin(ecuk)].copy()
val["ECUK"] = val.Device.map(ecuk)
val["Δ"] = 100 * (val.GWh_nat - val.ECUK) / val.ECUK

fig, ax = plt.subplots(figsize=(12,7))
x = np.arange(len(val))
w = 0.35
bars_model = ax.bar(x - w/2, val.GWh_nat, w, label="Model", color="#1f77b4")
bars_ecuk = ax.bar(x + w/2, val.ECUK, w, label="ECUK", color="#ff7f0e")

# Add value labels on top of bars
for bar in bars_model:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

for bar in bars_ecuk:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

# Calculate maximum bar height for positioning
max_bar = max(val.GWh_nat.max(), val.ECUK.max())
top_margin = max_bar * 0.25  # 25% headroom

# Add percentage difference labels with adjusted position
for j in range(len(val)):
    row = val.iloc[j]
    m, e, d = row.GWh_nat, row.ECUK, row.Δ
    col = 'green' if abs(d) < 10 else 'orange' if abs(d) < 25 else 'red'
    
    # Position above both bars with padding
    y_pos = max(m, e) + (top_margin * 0.15)
    ax.text(j, y_pos, f'{d:+.0f}%', 
            ha='center', bbox=dict(facecolor=col, alpha=0.8, pad=0.3))

# Set y-axis limits with headroom
ax.set_ylim(0, max_bar + top_margin)

ax.set_ylabel("National electricity (GWh / yr)")
ax.set_title("Model vs ECUK – kitchen appliances")
ax.set_xticks(x)
ax.set_xticklabels(val.Device, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()

print("\nValidation (GWh / yr)")
print(val[["Device", "GWh_nat", "ECUK", "Δ"]].to_string(
    index=False,
    formatters={
        "GWh_nat": lambda x: f'{x:,.0f}',
        "ECUK": lambda x: f'{x:,.0f}',
        "Δ": lambda x: f'{x:+.1f}%'
    }
))

# ── 8. Summary Table ───────────────────────────────────────────────
print("\nDevice Energy and Emissions Summary:")
print("="*65)
print(f"{'Device':<20} {'kWh/house/yr':>12} {'GWh/UK/yr':>12} {'kt CO₂e/yr':>12} {'kg CO₂e/house/yr':>18}")
print("-"*65)
for _, row in df.iterrows():
    print(f"{row.Device:<20} "
          f"{row.kWh_hh:>12.1f} "
          f"{row.GWh_nat:>12.1f} "
          f"{row.kt_nat:>12.1f} "
          f"{row.kgCO2_hh:>18.1f}")
print("="*65)