import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
n_samples = 5000

def generate_dataset():
    print("Generating synthetic clinical data...")
    
    # 1. Basic Patient Characteristics
    ids = range(1, n_samples + 1)
    # Age: 6 months to 60 months (5 years)
    age_months = np.random.randint(6, 61, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Weight: Roughly correlated with age + some random variance
    weight_kg = (age_months * 0.3) + np.random.normal(3, 1, n_samples)
    weight_kg = weight_kg.clip(min=4) # Ensure no unrealistic weights
    
    # Nutritional Status (MUAC - Mid-Upper Arm Circumference in cm)
    # Introduce correlation: Lower weight/age ratio often implies lower MUAC
    muac = np.random.normal(14, 1.5, n_samples)
    # Introduce missing values (simulating field data gaps)
    muac[np.random.rand(n_samples) < 0.1] = np.nan 
    
    # 2. Clinical Presentation
    # Dehydration: WHO Scale
    dehydration = np.random.choice(['None', 'Some', 'Severe'], n_samples, p=[0.6, 0.3, 0.1])
    duration_days = np.random.choice([1, 2, 3, 4, 5], n_samples)
    stool_frequency = np.random.poisson(5, n_samples).clip(min=3) # Stools per day
    
    # 3. History & Risk Factors
    travel_history = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    prev_infections = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    # 4. Microbiology (The underlying cause)
    # Rotavirus (Viral - Azithro won't work), ETEC/Shigella/Cholera (Bacterial - Azithro might work)
    pathogens = ['Rotavirus', 'ETEC', 'Shigella', 'Vibrio', 'Unknown']
    pathogen_col = np.random.choice(pathogens, n_samples, p=[0.4, 0.2, 0.1, 0.05, 0.25])
    
    # Resistance Profile (Only relevant for bacteria)
    # 0 = Sensitive, 1 = Resistant
    resistance = np.zeros(n_samples)
    for i in range(n_samples):
        if pathogen_col[i] in ['ETEC', 'Shigella', 'Vibrio']:
            resistance[i] = np.random.choice([0, 1], p=[0.7, 0.3]) # 30% resistance rate
    
    # 5. Treatment Assignment (Randomized Control Trial style)
    treatment = np.random.choice(['Azithromycin', 'Placebo'], n_samples)
    
    # 6. Outcome Generation (Simulating Biology)
    # Target: Hours to resolution
    outcome_hours = []
    
    for i in range(n_samples):
        base_recovery = 72 # Avg 3 days
        
        # Impact of dehydration and malnutrition
        if dehydration[i] == 'Severe': base_recovery += 24
        if np.isnan(muac[i]) or muac[i] < 12.5: base_recovery += 12
        
        # Impact of Treatment
        treatment_effect = 0
        
        if treatment[i] == 'Azithromycin':
            if pathogen_col[i] in ['Rotavirus', 'Unknown']:
                treatment_effect = 0 # No effect on virus
            elif pathogen_col[i] in ['ETEC', 'Shigella', 'Vibrio']:
                if resistance[i] == 0:
                    treatment_effect = -36 # Big reduction if sensitive bacteria
                else:
                    treatment_effect = 0 # Resistant bacteria
        
        # Add randomness
        final_hours = base_recovery + treatment_effect + np.random.normal(0, 10)
        outcome_hours.append(max(24, final_hours)) # Minimum 24 hours

    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': ids,
        'age_months': age_months,
        'gender': gender,
        'weight_kg': weight_kg.round(2),
        'muac_cm': muac,
        'dehydration_grade': dehydration,
        'duration_pre_enrollment_days': duration_days,
        'travel_history': travel_history,
        'previous_infections_count': prev_infections,
        'pathogen_identified': pathogen_col,
        'azithro_resistance_detected': resistance, # 1=Resistant, 0=Sensitive
        'treatment_group': treatment,
        'hours_to_resolution': np.round(outcome_hours, 1)
    })
    
    # Save to CSV for the next script
    df.to_csv('raw_diarrhea_data.csv', index=False)
    print("Data generated and saved to 'raw_diarrhea_data.csv'")
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv('data_final.csv', index=False)