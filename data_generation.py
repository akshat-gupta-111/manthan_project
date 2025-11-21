import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

def generate_synthetic_diarrhea_data(n_rows=8000):
    data = []
    
    pathogens = ['Rotavirus', 'Norovirus', 'E. coli (ETEC)', 'Shigella', 'Cholera', 'Unknown']
    dehydration_levels = ['None', 'Some', 'Severe']
    travel_options = ['No Travel', 'Domestic', 'International']
    
    for _ in range(n_rows):
        # --- 1. Patient Characteristics ---
        age_months = np.random.randint(1, 12 * 12) # 1 month to 12 years
        sex = np.random.choice(['Male', 'Female'])
        
        # Weight correlates with age (approximate growth curve with noise)
        base_weight = 3.5 + (age_months * 0.25) 
        weight_kg = round(np.random.normal(base_weight, 2.0), 2)
        weight_kg = max(2.5, weight_kg) # Ensure no negative/tiny weights

        # --- 2. Age Groups (Derived) ---
        if age_months < 12:
            age_group = 'Infant'
        elif age_months < 36:
            age_group = 'Toddler'
        else:
            age_group = 'Child'

        # --- 3. Nutritional Status (MUAC) ---
        # Mid-Upper Arm Circumference in cm. < 11.5 is severe acute malnutrition
        muac_cm = round(np.random.normal(14.5, 1.5), 1)
        muac_cm = max(9.0, min(20.0, muac_cm)) # Clip reasonable range

        # --- 4. Clinical Presentation ---
        duration_days = np.random.randint(1, 14)
        severity = np.random.choice(dehydration_levels, p=[0.2, 0.5, 0.3])
        fever = np.random.choice([0, 1], p=[0.4, 0.6]) # 1 = Yes
        
        # --- 5. History ---
        travel_history = np.random.choice(travel_options, p=[0.7, 0.2, 0.1])
        previous_infections = np.random.poisson(1) # Count of recent infections
        
        # --- 6. Microbiology & Resistance ---
        # Bacteria = High Azithro utility; Virus = Low utility
        pathogen = np.random.choice(pathogens, p=[0.3, 0.2, 0.25, 0.15, 0.05, 0.05])
        
        is_bacterial = pathogen in ['E. coli (ETEC)', 'Shigella', 'Cholera']
        
        # Is the specific bug resistant?
        pathogen_resistance = np.random.choice(['Sensitive', 'Resistant'], p=[0.7, 0.3])
        
        # Local community resistance index (0.0 to 1.0)
        local_amr_index = round(np.random.uniform(0.1, 0.6), 2)

        # --- 7. Treatment Given ---
        treatment_given = np.random.choice(['Azithromycin', 'Placebo/ORS Only'], p=[0.5, 0.5])

        # --- 8. Calculating Benefit Probability (The Target Variable) ---
        # This logic creates the "Signal" for the ML model to learn.
        
        # Start with a base probability of benefit
        benefit_prob = 0.1 
        
        # Logic: Azithromycin helps bacteria, but not if resistant, and not viruses.
        if treatment_given == 'Azithromycin':
            if is_bacterial:
                benefit_prob += 0.6
                if pathogen_resistance == 'Resistant':
                    benefit_prob -= 0.4  # Resistance drastically reduces benefit
            else:
                # Viral case: Antibiotics might actually harm (microbiome) or do nothing
                benefit_prob -= 0.05 
        
        # Adjust for Local Resistance (General environmental factor)
        benefit_prob -= (local_amr_index * 0.2)
        
        # Adjust for Malnutrition (Lower MUAC reduces recovery chance)
        if muac_cm < 11.5:
            benefit_prob -= 0.15
        elif muac_cm < 12.5:
            benefit_prob -= 0.05
            
        # Adjust for Severity (Severe cases might benefit more from aggressive treatment)
        if severity == 'Severe' and is_bacterial:
            benefit_prob += 0.1
            
        # Add some random noise (biological variability)
        benefit_prob += np.random.normal(0, 0.05)
        
        # Clip probability between 0 and 1
        benefit_prob = max(0.0, min(1.0, benefit_prob))
        
        row = {
            'Age_Months': age_months,
            'Age_Group': age_group,
            'Sex': sex,
            'Weight_kg': weight_kg,
            'MUAC_cm': muac_cm,
            'Nutritional_Status': 'Malnourished' if muac_cm < 12.5 else 'Normal',
            'Duration_Days': duration_days,
            'Dehydration_Severity': severity,
            'Fever': fever,
            'Travel_History': travel_history,
            'Previous_Infections_Count': previous_infections,
            'Pathogen_Type': pathogen,
            'Is_Bacterial': 1 if is_bacterial else 0,
            'Pathogen_Resistance': pathogen_resistance,
            'Local_AMR_Index': local_amr_index,
            'Treatment_Given': treatment_given,
            'Benefit_Probability': round(benefit_prob, 4) # TARGET VARIABLE
        }
        data.append(row)

    return pd.DataFrame(data)

# Generate the data
df = generate_synthetic_diarrhea_data(8000)

# Show first few rows
print(df.head())

# Optional: Save to CSV
df.to_csv('azithromycin_pediatric_data.csv', index=False)