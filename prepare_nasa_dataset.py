"""
Data Preparation Script: NASA Li-ion Battery Prognostics Benchmark Dataset
Extracts cycle-level BMS physical sensor features from raw discharge cycles
and assigns standard battery degradation health states.
"""

import os
import numpy as np
import pandas as pd

def extract_nasa_features(metadata_path, data_dir, output_csv, battery_ids=None, ambient_temp=24):
    print("=" * 70)
    print("NASA Li-ion Battery Benchmark Feature Extraction")
    print("=" * 70)
    
    meta_df = pd.read_csv(metadata_path)
    discharges = meta_df[meta_df['type'] == 'discharge'].copy()
    discharges['Capacity'] = pd.to_numeric(discharges['Capacity'], errors='coerce')
    discharges = discharges.dropna(subset=['Capacity'])
    discharges = discharges[discharges['Capacity'] > 0]
    
    if ambient_temp is not None:
        discharges = discharges[discharges['ambient_temperature'] == ambient_temp]
        
    if battery_ids is not None:
        discharges = discharges[discharges['battery_id'].isin(battery_ids)]
        
    print(f"Total cycles to process: {len(discharges)}")
    print(f"Batteries included: {discharges['battery_id'].unique().tolist()}")
    
    records = []
    skipped = 0
    
    for idx, row in discharges.iterrows():
        fpath = os.path.join(data_dir, row['filename'])
        if not os.path.exists(fpath):
            skipped += 1
            continue
            
        c_df = pd.read_csv(fpath)
        if len(c_df) < 5:
            skipped += 1
            continue
            
        v = c_df['Voltage_measured'].values
        curr = c_df['Current_measured'].values
        temp = c_df['Temperature_measured'].values
        t = c_df['Time'].values
        v_load = c_df['Voltage_load'].values
        c_load = c_df['Current_load'].values
        
        duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
        if duration <= 0:
            duration = 1.0
            
        # Numerical integration for delivered energy: sum(V * |I| * dt) / 3600
        dt = np.diff(t, prepend=t[0])
        dt = np.clip(dt, 0, 100) # clip any timestamp anomalies
        power = v * np.abs(curr)
        energy_wh = float(np.sum(power * dt) / 3600.0)
        
        cap = float(row['Capacity'])
        
        # NASA Ames standard EOL thresholding:
        # Nominal: 2.0 Ah
        # Critical EOL (<1.4 Ah, ~70% capacity remaining)
        # Warning (1.4 Ah <= Cap < 1.6 Ah, 70-80% capacity)
        # OK (>= 1.6 Ah, fresh to healthy operational state)
        if cap < 1.4:
            state = 'Critical'
            label = 0
        elif cap < 1.6:
            state = 'Warning'
            label = 2
        else:
            state = 'OK'
            label = 1
            
        records.append({
            'battery_id': row['battery_id'],
            'cycle_file': row['filename'],
            'Capacity_Ah': round(cap, 4),
            'Health_State': state,
            'Target_Label': label,
            'Max_Temp_C': round(float(np.max(temp)), 3),
            'Avg_Temp_C': round(float(np.mean(temp)), 3),
            'Temp_Rise_C': round(float(np.max(temp) - np.min(temp)), 3),
            'Min_Voltage_V': round(float(np.min(v)), 4),
            'Avg_Voltage_V': round(float(np.mean(v)), 4),
            'Final_Voltage_V': round(float(v[-1]), 4),
            'Voltage_Drop_Rate_Vps': round(float((v[0] - v[-1]) / duration), 6),
            'Avg_Current_A': round(float(np.mean(curr)), 4),
            'Max_Current_A': round(float(np.max(np.abs(curr))), 4),
            'Discharge_Duration_s': round(duration, 1),
            'Energy_Delivered_Wh': round(energy_wh, 4),
            'Max_Load_Current_A': round(float(np.max(c_load)), 4),
            'Min_Load_Voltage_V': round(float(np.min(v_load)), 4),
            'Avg_Load_Voltage_V': round(float(np.mean(v_load)), 4)
        })
        
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    
    print(f"\nExtraction complete! Saved {len(out_df)} cycle records to: {output_csv}")
    print("\nClass distribution:")
    print(out_df['Health_State'].value_counts())
    print("\nClass distribution (%):")
    print((out_df['Health_State'].value_counts(normalize=True) * 100).round(2))
    print("=" * 70)
    return out_df

if __name__ == '__main__':
    metadata_path = os.path.expanduser('~/Downloads/cleaned_dataset/metadata.csv')
    data_dir = os.path.expanduser('~/Downloads/cleaned_dataset/data')
    
    # 1. Standard Benchmark: Classic 4 cells (B0005, B0006, B0007, B0018)
    extract_nasa_features(
        metadata_path=metadata_path,
        data_dir=data_dir,
        output_csv='nasa_battery_benchmark_classic4.csv',
        battery_ids=['B0005', 'B0006', 'B0007', 'B0018'],
        ambient_temp=24
    )
    
    # 2. Comprehensive Benchmark: All 24°C cells (1,375 cycles)
    extract_nasa_features(
        metadata_path=metadata_path,
        data_dir=data_dir,
        output_csv='nasa_battery_benchmark_all24c.csv',
        battery_ids=None,
        ambient_temp=24
    )
