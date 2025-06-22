import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Load the training data
def load_data(filename='data/isoflops_curves.json'):
    """Load training run data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def quadratic_func(n, a, b, c):
    """Quadratic function for fitting IsoFLOP curves"""
    return a * (np.log10(n))**2 + b * np.log10(n) + c

def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)

def extract_isoflop_data(data):
    """
    Group training runs by compute budget and sort by parameters
    """
    budget_groups = defaultdict(list)
    for run in data:
        budget = run['compute_budget']
        budget_groups[budget].append(run)
    
    # Sort each group by parameters
    for budget in budget_groups:
        budget_groups[budget].sort(key=lambda x: x['parameters'])
    
    return budget_groups

def fit_quadratic_curves(budget_groups):
    """
    Fit quadratic curves to each IsoFLOP curve and find minima
    """
    fitted_curves = {}
    optimal_points = []
    
    for budget, runs in budget_groups.items():
        if len(runs) < 5:  # Need enough points for quadratic fit
            continue
            
        parameters = np.array([run['parameters'] for run in runs])
        losses = np.array([run['final_loss'] for run in runs])
        
        try:
            # Fit quadratic in log space
            popt, _ = curve_fit(quadratic_func, parameters, losses, maxfev=5000)
            
            # Find minimum of quadratic
            a, b, c = popt
            if a > 0:  # Only valid if parabola opens upward
                # Minimum occurs at n_opt where derivative = 0
                # d/d(log N) [a*(log N)^2 + b*(log N) + c] = 2a*log N + b = 0
                log_n_opt = -b / (2 * a)
                n_opt = 10**log_n_opt
                loss_opt = quadratic_func(n_opt, a, b, c)
                
                fitted_curves[budget] = {
                    'parameters': parameters,
                    'losses': losses,
                    'coeffs': popt,
                    'n_opt': n_opt,
                    'loss_opt': loss_opt
                }
                
                optimal_points.append({
                    'compute_budget': budget,
                    'parameters': n_opt,
                    'final_loss': loss_opt
                })
        except:
            # If quadratic fit fails, use minimum loss point
            min_idx = np.argmin(losses)
            optimal_points.append({
                'compute_budget': budget,
                'parameters': parameters[min_idx],
                'final_loss': losses[min_idx]
            })
    
    return fitted_curves, optimal_points

def calculate_dataset_size(compute_budget, parameters):
    """Calculate dataset size using C ≈ 6ND"""
    return compute_budget / (6 * parameters)

def plot_isoflop_curves(budget_groups, fitted_curves):
    """Plot 1: C-N-Loss curves showing multiple quadratic lines"""
    plt.figure(figsize=(14, 8))
    
    # Get unique compute budgets and sort them
    budgets = sorted(budget_groups.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(budgets)))
    
    for i, budget in enumerate(budgets):
        runs = budget_groups[budget]
        parameters = np.array([run['parameters'] for run in runs])
        losses = np.array([run['final_loss'] for run in runs])
        
        # Plot data points
        plt.scatter(parameters, losses, alpha=0.6, color=colors[i], 
                   label=f'C = {budget:.1e}', s=30)
        
        # Plot fitted quadratic curve if available
        if budget in fitted_curves:
            curve_data = fitted_curves[budget]
            coeffs = curve_data['coeffs']
            
            # Generate smooth curve
            n_smooth = np.logspace(np.log10(parameters.min()), 
                                  np.log10(parameters.max()), 200)
            loss_smooth = quadratic_func(n_smooth, *coeffs)
            
            plt.plot(n_smooth, loss_smooth, '--', color=colors[i], 
                    linewidth=2, alpha=0.8)
            
            # Mark minimum point
            plt.scatter(curve_data['n_opt'], curve_data['loss_opt'], 
                       color='red', s=100, marker='*', 
                       edgecolor='black', linewidth=1, zorder=5)
    
    plt.xlabel('Model Size N (Parameters)', fontsize=12)
    plt.ylabel('Final Training Loss', fontsize=12)
    plt.title('IsoFLOP Curves: Training Loss vs Model Size for Different Compute Budgets', 
              fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('output/isoflop_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 圖1已保存為: output/isoflop_curves.png")
    plt.show()

def plot_scaling_laws(optimal_points, num_budgets):
    """Plot 2: Fitted scaling laws (should be straight lines in log-log space)"""
    # Prepare data
    compute_budgets = np.array([p['compute_budget'] for p in optimal_points])
    optimal_params = np.array([p['parameters'] for p in optimal_points])
    optimal_datasets = np.array([calculate_dataset_size(c, n) 
                                for c, n in zip(compute_budgets, optimal_params)])
    
    # Fit scaling laws
    def fit_scaling_law(x_data, y_data):
        log_x = np.log10(x_data)
        log_y = np.log10(y_data)
        coeffs = np.polyfit(log_x, log_y, 1)
        b = coeffs[0]  # slope
        log_a = coeffs[1]  # intercept
        a = 10**log_a
        return a, b
    
    a_n, b_n = fit_scaling_law(compute_budgets, optimal_params)
    a_d, b_d = fit_scaling_law(compute_budgets, optimal_datasets)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Model size scaling
    ax1.loglog(compute_budgets, optimal_params, 'bo', markersize=8, 
               label='⟨C_i, N_opt(C_i)⟩ points')
    
    # Fit line
    x_smooth = np.logspace(np.log10(compute_budgets.min()), 
                          np.log10(1e24), 1000)
    y_smooth = power_law(x_smooth, a_n, b_n)
    ax1.loglog(x_smooth, y_smooth, 'r-', linewidth=2, 
               label=f'N_opt = {a_n:.2e} × C^{b_n:.3f}')
    
    # Predictions
    predictions_c = [1e23, 1e24]
    predictions_n = [power_law(c, a_n, b_n) for c in predictions_c]
    ax1.loglog(predictions_c, predictions_n, 'rs', markersize=10, 
               label='Extrapolations')
    
    ax1.set_xlabel('Compute Budget C (FLOPs)', fontsize=12)
    ax1.set_ylabel('Optimal Model Size N_opt (Parameters)', fontsize=12)
    ax1.set_title('Model Size Scaling Law', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Dataset size scaling
    ax2.loglog(compute_budgets, optimal_datasets, 'go', markersize=8, 
               label='⟨C_i, D_opt(C_i)⟩ points')
    
    # Fit line
    y_smooth_d = power_law(x_smooth, a_d, b_d)
    ax2.loglog(x_smooth, y_smooth_d, 'r-', linewidth=2, 
               label=f'D_opt = {a_d:.2e} × C^{b_d:.3f}')
    
    # Predictions
    predictions_d = [power_law(c, a_d, b_d) for c in predictions_c]
    ax2.loglog(predictions_c, predictions_d, 'rs', markersize=10, 
               label='Extrapolations')
    
    ax2.set_xlabel('Compute Budget C (FLOPs)', fontsize=12)
    ax2.set_ylabel('Optimal Dataset Size D_opt (Tokens)', fontsize=12)
    ax2.set_title('Dataset Size Scaling Law', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('output/scaling_laws.png', dpi=300, bbox_inches='tight')
    print("✓ 圖2已保存為: output/scaling_laws.png")
    plt.show()
    
    # Print results
    n_opt_1e23 = power_law(1e23, a_n, b_n)
    n_opt_1e24 = power_law(1e24, a_n, b_n)
    d_opt_1e23 = power_law(1e23, a_d, b_d)
    d_opt_1e24 = power_law(1e24, a_d, b_d)
    
    print("\n" + "="*60)
    print("DELIVERABLE ANSWERS")
    print("="*60)
    print(f"\n1. Model Size Scaling Law:")
    print(f"   For a budget of 10^23 FLOPs, the predicted optimal model size is {n_opt_1e23:.2e} parameters, and for 10^24 FLOPs it is {n_opt_1e24:.2e} parameters.")
    
    print(f"\n2. Dataset Size Scaling Law:")
    print(f"   For a budget of 10^23 FLOPs, the predicted optimal dataset size is {d_opt_1e23:.2e} tokens, and for 10^24 FLOPs it is {d_opt_1e24:.2e} tokens.")
    
    # Save results to file
    results_text = f"""IsoFLOPs Analysis Results
=========================

Model Size Scaling Law: N_opt = {a_n:.2e} × C^{b_n:.3f}
Dataset Size Scaling Law: D_opt = {a_d:.2e} × C^{b_d:.3f}

Predictions:
- For 10^23 FLOPs: {n_opt_1e23:.2e} parameters, {d_opt_1e23:.2e} tokens
- For 10^24 FLOPs: {n_opt_1e24:.2e} parameters, {d_opt_1e24:.2e} tokens

Generated {len(optimal_points)} optimal points from {num_budgets} compute budgets.
"""
    
    with open('output/results.txt', 'w') as f:
        f.write(results_text)
    print("✓ 結果已保存為: output/results.txt")

def main():
    # Load data
    print("Loading training data...")
    data = load_data()
    print(f"Loaded {len(data)} training runs")
    
    # Group by compute budget
    print("Grouping data by compute budget...")
    budget_groups = extract_isoflop_data(data)
    print(f"Found {len(budget_groups)} unique compute budgets")
    
    # Fit quadratic curves to each IsoFLOP
    print("Fitting quadratic curves to IsoFLOP data...")
    fitted_curves, optimal_points = fit_quadratic_curves(budget_groups)
    print(f"Successfully fitted {len(fitted_curves)} quadratic curves")
    print(f"Extracted {len(optimal_points)} optimal points")
    
    # Plot 1: IsoFLOP curves with quadratic fits
    print("\nGenerating Plot 1: IsoFLOP Curves (C-N-Loss with quadratic fits)...")
    plot_isoflop_curves(budget_groups, fitted_curves)
    
    # Plot 2: Scaling laws
    print("Generating Plot 2: Scaling Laws (straight lines in log-log space)...")
    plot_scaling_laws(optimal_points, len(budget_groups))
    
    return fitted_curves, optimal_points

if __name__ == "__main__":
    fitted_curves, optimal_points = main()
    
    # Script generates two key visualizations:
    # 1. output/isoflop_curves.png - IsoFLOP curves showing training loss vs model size 
    #    for different compute budgets (multiple quadratic curves with minima marked)
    # 2. output/scaling_laws.png - Scaling laws showing optimal model size and dataset size 
    #    vs compute budget (straight lines in log-log space with extrapolations)
    # 3. output/results.txt - Text file with numerical results and predictions