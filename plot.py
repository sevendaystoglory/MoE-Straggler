import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
import torch

def generate_expert_plots(model, tokenizer, input_text, run_idx, all_expert_assignments):
    """Collect expert assignments from a model after running inference"""
    # Create directory for plots if it doesn't exist
    os.makedirs("expert_plots", exist_ok=True)
    
    # Run inference to populate expert assignments
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model(**inputs)
    
    # Collect assignments from all layers for this run
    run_assignments = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'expert_assignments'):
            run_assignments[layer_idx] = {
                expert_id: token_list.copy() 
                for expert_id, token_list in layer.block_sparse_moe.expert_assignments.items()
            }
    
    # Store assignments for this run
    all_expert_assignments[run_idx] = run_assignments

def generate_cumulative_plots(model, all_expert_assignments):
    """Generate the two requested plots with cumulative data from all runs"""
    if not all_expert_assignments:
        print("No expert assignments collected.")
        return
    
    # 1. Cumulative total tokens in each expert
    plot_cumulative_expert_usage(model, all_expert_assignments)
    
    # 2. Heatmap across layers
    plot_layer_expert_heatmap(model, all_expert_assignments)
    
    print(f"Plots have been saved to the 'expert_plots' directory.")

def plot_cumulative_expert_usage(model, all_expert_assignments):
    """Plot cumulative total tokens handled by each expert across all runs"""
    try:
        num_experts = model.config.num_local_experts
    except AttributeError:
        # Find the maximum expert ID across all assignments
        num_experts = 0
        for run_assignments in all_expert_assignments.values():
            for layer_assignments in run_assignments.values():
                if layer_assignments:
                    num_experts = max(num_experts, max(layer_assignments.keys()) + 1)
    
    # Count total tokens per expert across all runs and layers
    expert_counts = defaultdict(int)
    
    for run_assignments in all_expert_assignments.values():
        for layer_assignments in run_assignments.values():
            for expert_id, token_list in layer_assignments.items():
                expert_counts[expert_id] += len(token_list)
    
    # Plot total token counts per expert
    plt.figure(figsize=(12, 6))
    experts = sorted(expert_counts.keys())
    token_counts = [expert_counts[expert] for expert in experts]
    
    bars = plt.bar(experts, token_counts, color='darkblue', alpha=0.7)
    
    # Add count labels on top of bars
    for bar, count in zip(bars, token_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Cumulative Total Tokens Handled by Each Expert (All Runs, All Layers)', fontsize=14)
    plt.xlabel('Expert ID', fontsize=12)
    plt.ylabel('Total Number of Tokens', fontsize=12)
    plt.xticks(experts)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('expert_plots/cumulative_expert_usage.png', dpi=300)
    plt.close()

def plot_layer_expert_heatmap(model, all_expert_assignments):
    """Create a heatmap showing expert usage across layers"""
    # Determine number of layers and experts
    num_layers = max(max(run_assignments.keys()) for run_assignments in all_expert_assignments.values()) + 1
    
    try:
        num_experts = model.config.num_local_experts
    except AttributeError:
        # Find the maximum expert ID across all assignments
        num_experts = 0
        for run_assignments in all_expert_assignments.values():
            for layer_assignments in run_assignments.values():
                if layer_assignments:
                    num_experts = max(num_experts, max(layer_assignments.keys()) + 1)
    
    # Initialize a matrix to count tokens per expert per layer
    layer_expert_counts = np.zeros((num_layers, num_experts))
    
    # Aggregate token counts per expert per layer across all runs
    for run_assignments in all_expert_assignments.values():
        for layer_idx, layer_assignments in run_assignments.items():
            for expert_id, token_list in layer_assignments.items():
                layer_expert_counts[layer_idx, expert_id] += len(token_list)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(layer_expert_counts, annot=True, fmt='.0f', cmap='YlGnBu')
    
    plt.title('Token Distribution Across Experts and Layers (All Runs)', fontsize=14)
    plt.xlabel('Expert ID', fontsize=12)
    plt.ylabel('Layer ID', fontsize=12)
    
    # Add a colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Number of Tokens', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('expert_plots/layer_expert_heatmap.png', dpi=300)
    plt.close() 