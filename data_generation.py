import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(101)

def generate_clinical_dataset(n_rows=10000):
    data = []
    
    # Domain Definitions
    pathogens = ['Rotavirus', 'Norovirus', 'Shigella', 'E. coli (ETEC)', 'Cholera', 'Unknown']
    # Dehydration is ordinal: None < Some < Severe
    dehydration_levels = ['None', 'Some', 'Severe']
    travel_opts = ['None', 'Domestic', 'International']
    
    for _ in range(n_rows):
        # --- 1. Basic Demographics ---
        age_months = np.random.randint(1, 144) # 1 month to 12 years
        sex = np.random.choice(['Male', 'Female'])
        
        # Weight correlates with age (approximate)
        weight_kg = max(2.5, np.random.normal(3.5 + (age_months * 0.25), 2.0))
        
        # --- 2. Feature Engineering: Nutritional Status (MUAC) ---
        # MUAC (cm) correlates with weight/age status. 
        # <11.5cm = Severe Acute Malnutrition (SAM)
        muac_cm = np.random.normal(14.5, 1.8)
        muac_cm = max(9.0, min(22.0, muac_cm))
        
        # --- 3. Feature Engineering: Age Groups ---
        if age_months < 12: age_group = 'Infant'
        elif age_months < 60: age_group = 'Toddler'
        else: age_group = 'Child'

        # --- 4. Clinical Presentation ---
        duration_days = np.random.randint(1, 10)
        # Correlate severity with duration and malnutrition
        severity_prob = [0.5, 0.3, 0.2] if muac_cm > 12.5 else [0.2, 0.4, 0.4]
        dehydration = np.random.choice(dehydration_levels, p=severity_prob)
        
        # --- 5. History & Context ---
        travel_history = np.random.choice(travel_opts, p=[0.8, 0.15, 0.05])
        previous_infections = np.random.poisson(1)
        
        # Local Antimicrobial Resistance (AMR) Index (0.0 = Low resistance, 1.0 = High)
        local_amr_index = round(np.random.uniform(0.1, 0.7), 2)
        
        # --- 6. Microbiology ---
        pathogen = np.random.choice(pathogens, p=[0.35, 0.15, 0.15, 0.20, 0.05, 0.10])
        is_bacterial = pathogen in ['Shigella', 'E. coli (ETEC)', 'Cholera']
        
        # Antibiotic Resistance of the specific bug
        is_resistant_strain = np.random.choice([0, 1], p=[0.7, 0.3]) # 1 = Resistant
        
        # --- 7. Target Variable: Benefit Probability ---
        # Base calculation logic
        benefit = 0.1 # Baseline (low benefit)
        
        if is_bacterial:
            benefit += 0.6 # Big jump for bacteria
            if is_resistant_strain == 1:
                benefit -= 0.4 # Resistance kills efficacy
            elif local_amr_index > 0.5:
                benefit -= 0.1 # Environmental resistance factor
        else:
            # Viral causes
            benefit -= 0.1 # Antibiotics not recommended
            
        # Clinical Severity Adjustments
        if dehydration == 'Severe': benefit += 0.15 # Higher urgency
        if muac_cm < 11.5: benefit += 0.1 # High risk patient needs aggressive care
        
        # Cap between 0 and 1 and add noise
        benefit = max(0.0, min(1.0, benefit + np.random.normal(0, 0.02)))

        data.append({
            'Age_Months': age_months,
            'Age_Group': age_group,
            'Sex': sex,
            'Weight_kg': round(weight_kg, 2),
            'MUAC_cm': round(muac_cm, 1),
            'Dehydration_Severity': dehydration,
            'Duration_Days': duration_days,
            'Travel_History': travel_history,
            'Previous_Infections': previous_infections,
            'Local_AMR_Index': local_amr_index,
            'Pathogen': pathogen,
            'Is_Resistant_Strain': is_resistant_strain,
            'Treatment_Benefit_Probability': round(benefit, 4)
        })

    df = pd.DataFrame(data)
    
    # --- INTRODUCE MISSING VALUES (To simulate real world data) ---
    # Randomly remove 10% of Dehydration_Severity and MUAC
    mask1 = np.random.rand(len(df)) < 0.1
    df.loc[mask1, 'Dehydration_Severity'] = np.nan
    
    mask2 = np.random.rand(len(df)) < 0.05
    df.loc[mask2, 'MUAC_cm'] = np.nan
    
    return df

# Generate Raw Data
raw_df = generate_clinical_dataset(8000)
print("Raw Data Generated. Missing values introduced.")
print(raw_df.isnull().sum())
raw_df.to_csv('azithromycin_pediatric_data.csv', index=False)