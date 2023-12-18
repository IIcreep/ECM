import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

def ecm(params, freq, T=25):
    # Unpack parameters, including reference resistance value and activation energy
    R1_0, Ea1, R2_0, Ea2, Rs, L, Q1, alpha1, Q2, alpha2 = params

    T_K = T + 273.15
    k = 8.617333262145e-5
    R1 = R1_0 * np.exp(Ea1 / (k * T_K))
    R2 = R2_0 * np.exp(Ea2 / (k * T_K))

    omega = 2 * np.pi * freq
    Z_CPE1 = 1 / (1j * omega)**alpha1 / Q1
    Z_CPE2 = 1 / (1j * omega)**alpha2 / Q2
    Z_parallel1 = (R1 * Z_CPE1) / (R1 + Z_CPE1)
    Z_parallel2 = (R2 * Z_CPE2) / (R2 + Z_CPE2)
    Z_pred = Rs + 1j * omega * L + Z_parallel1 + Z_parallel2

    return Z_pred.real, Z_pred.imag

# Define loss function
def complex_loss(params, freq, ReZ_data, ImZ_data, T=25):
    # Unpack parameters, including reference resistance value and activation energy
    R1_0, Ea1, R2_0, Ea2, Rs, L, Q1, alpha1, Q2, alpha2 = params
    
    T_K = T + 273.15
    k = 8.617333262145e-5
    R1 = R1_0 * np.exp(Ea1 / (k * T_K))
    R2 = R2_0 * np.exp(Ea2 / (k * T_K))

    omega = 2 * np.pi * freq
    Z_CPE1 = 1 / (1j * omega) ** alpha1 / Q1
    Z_CPE2 = 1 / (1j * omega) ** alpha2 / Q2
    Z_parallel1 = (R1 * Z_CPE1) / (R1 + Z_CPE1)
    Z_parallel2 = (R2 * Z_CPE2) / (R2 + Z_CPE2)
    Z_pred = Rs + 1j * omega * L + Z_parallel1 + Z_parallel2

    loss_val = np.sum(((ReZ_data - Z_pred.real) / ReZ_data) ** 2) + np.sum(((ImZ_data - Z_pred.imag) / ImZ_data) ** 2)
    return loss_val

if __name__ == "__main__":
    df = pd.read_csv("/Users/hu/Desktop/Battery/ECMData/EIS data/EIS_state_I_25C06.csv")
    unique_cycle_numbers = df['cyclenumber'].unique()
    split_dfs = {}
    for number in unique_cycle_numbers:
        split_dfs[number] = df[df['cyclenumber'] == number]
        df_existing = pd.read_csv('/Users/hu/Desktop/Battery/parameter/ECM_I_25C06.csv')
        freq = split_dfs[number]['freq/Hz'].tolist()
        freq = np.array(freq)
        ReZ = split_dfs[number]['Re(Z)/Ohm'].tolist()
        ReZ = np.array(ReZ)
        ImZ = split_dfs[number]['-Im(Z)/Ohm'].tolist()
        ImZ = [-x for x in ImZ]
        ImZ = np.array(ImZ)
        Phase = split_dfs[number]['Phase(Z)/deg'].tolist()
        Phase = np.array(Phase)

        param_bounds = [(0.01, 1), # R1_0
                        (0.001, 0.1),    # Ea1
                        (0.01, 2),     # R2_0
                        (0.001, 0.1),    # Ea2
                        (1e-3, 1),  # Rs
                        (1e-9, 1e-3), # L
                        (10, 100),  # Q1
                        (0, 3.14),    # alpha1
                        (1e-3, 1),  # Q2
                        (0, 3.14)]    # alpha2

        result = differential_evolution(complex_loss, param_bounds, args=(freq, ReZ, ImZ), strategy='currenttobest1exp', popsize=40, tol=1e-3, mutation=(0.5, 1), recombination=1)
        optimal_params = result.x
        print("Optimal parameters are:", optimal_params)

        df_new_row = pd.DataFrame([result.x], columns=['R1_0', 'Ea1', 'R2_0', 'Ea2', 'Rs', 'L', 'Q1', 'alpha1', 'Q2', 'alpha2'])
        df_updated = pd.concat([df_existing, df_new_row], ignore_index=True)
        df_updated.to_csv('/Users/hu/Desktop/Battery/parameter/ECM_I_25C06.csv', index=False)

    # Calculate R2 and RMSE
    ecm_data = pd.read_csv('/Users/hu/Desktop/Battery/parameter/ECM_I_25C08.csv')
    eis_data = pd.read_csv('/Users/hu/Desktop/Battery/ECMData/EIS data/EIS_state_I_25C08.csv')

    def calculate_r2_rmse(ReZ_pred, ImZ_pred, ReZ_data, ImZ_data):
        ss_res = np.sum((ReZ_data - ReZ_pred) ** 2) + np.sum((ImZ_data - ImZ_pred) ** 2)
        ss_tot = np.sum((ReZ_data - np.mean(ReZ_data)) ** 2) + np.sum((ImZ_data - np.mean(ImZ_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(ss_res / len(ReZ_data))
        return r2, rmse

    # Extract ECM parameters
    unique_cycle_numbers = eis_data['cyclenumber'].unique()
    split_dfs = {}
    sumr2,sumrmse = 0,0
    for i,number in enumerate(unique_cycle_numbers):
        split_dfs[number] = eis_data[eis_data['cyclenumber'] == number]
        freq = split_dfs[number]['freq/Hz'].tolist()
        freq = np.array(freq)
        ReZ = split_dfs[number]['Re(Z)/Ohm'].tolist()
        ReZ = np.array(ReZ)
        ImZ = split_dfs[number]['-Im(Z)/Ohm'].tolist()
        ImZ = [-x for x in ImZ]
        ImZ = np.array(ImZ)
        Phase = split_dfs[number]['Phase(Z)/deg'].tolist()
        Phase = np.array(Phase)
        ecm_params = ecm_data.iloc[i]

        ReZ_pred, ImZ_pred = ecm(ecm_params, freq)

        r2, rmse = calculate_r2_rmse(ReZ_pred, ImZ_pred, ReZ, ImZ)
        sumr2,sumrmse = sumr2 + r2, sumrmse + rmse
        
    print(sumr2/len(unique_cycle_numbers))
    print(sumrmse/len(unique_cycle_numbers))
