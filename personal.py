import pandas as pd, numpy as np, matplotlib.pyplot as plt
plt.style.use("ggplot"); plt.rcParams.update({'font.size':10})

# ── 1. INPUT  ──────────────────────────────────────
data = [
 # Device, Pmid(W), T_active(min), P_stby(W), Units_mil
 ("Smartphones",     5.0,  165.5,  0.04, 64.93),
 ("Feature Phone",    1.75,   112.8,  0.075, 0.4101),
 ("Tablets",          12,  171.8,  0.05,  34.96),
 ("Smart Speaker",    2.4,   36,  1.3,   9.37)
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
stacked(df,"Household Personal device electricity","kWh / hh·yr")
stacked(df,"UK Personal device electricity","GWh / yr",nat=True)
carbon(df,"kgCO2_hh","Household Personal device CO₂e","kg / hh·yr","#d62728")
carbon(df,"kt_nat","UK Personal device CO₂e","kt / yr","red",nat=True)

#  ──── MONTE CARLO PLOTS ───────────────────────────────────────────────────
plt.figure(figsize=(10,6))
plt.hist(total_nat/1000, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
plt.axvline(np.percentile(total_nat/1000, 5), color='red', linestyle='--', label='5th percentile')
plt.axvline(np.percentile(total_nat/1000, 95), color='blue', linestyle='--', label='95th percentile')
plt.title('Monte Carlo: Total UK Personal Device Energy Consumption')
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
plt.title('Monte Carlo: Personal Device Energy Consumption Ranges')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── 7. ECUK validation ─────────────────────────────────────────────
print("\nNote: ECUK doesn't provide official energy consumption figures for these mobile devices")
print("Skipping validation step")

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