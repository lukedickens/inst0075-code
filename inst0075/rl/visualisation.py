import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from inst0075.rl.environment.states_and_actions import dominant_actions


def plot_mdp_matrices(mdp):
    fig, axes = plt.subplots(mdp.num_actions, 2, figsize=(12, 5 * mdp.num_actions))
    
    for i, action in enumerate(mdp.action_names):
        # Transition Heatmap
        sns.heatmap(mdp.T[i], annot=True, fmt=".1f", cmap="Blues", 
                    xticklabels=mdp.state_names, yticklabels=mdp.state_names, 
                    ax=axes[i, 0], cbar=False)
        axes[i, 0].set_title(f"Transitions (T) - Action: {action}")
        axes[i, 0].set_xlabel("To State")
        axes[i, 0].set_ylabel("From State")

        # Reward Heatmap
        sns.heatmap(mdp.R[i], annot=True, fmt=".1f", cmap="PiYG", center=0,
                    xticklabels=mdp.state_names, yticklabels=mdp.state_names, 
                    ax=axes[i, 1])
        axes[i, 1].set_title(f"Rewards (R) - Action: {action}")
        axes[i, 1].set_xlabel("To State")
        axes[i, 1].set_ylabel("From State")

    plt.tight_layout()
    plt.show()
    
def plot_policy_matrix(
        policy, mdp, scale=0.5, exclude_absorbing=True, states_as_rows=True,
        title=None):
    state_names = mdp.state_names
    action_names = mdp.action_names
    if exclude_absorbing:
        filter_ = ~np.array(mdp.absorbing)
        state_names = np.array(state_names)[filter_]
        policy = policy[filter_,:]

    num_states = state_names.size
    num_actions = mdp.num_actions    
    action_span = (1+num_actions)*scale
    state_span = (2+num_states)*scale
    
    if states_as_rows:
        plt.figure(figsize=(action_span, state_span)) 
        xticklabels = action_names
        yticklabels = state_names
        xlabel = "Action"
        ylabel = "State"
        matrix = policy
    else:
        # actions as rows
        plt.figure(figsize=(state_span, action_span))
        xticklabels = state_names
        yticklabels = action_names
        xlabel = "State"
        ylabel = "Action"
        matrix = policy.T

    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=xticklabels, yticklabels=yticklabels, cbar=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    else:
        plt.title(f"Policy")

    plt.tight_layout()
    plt.show()
    
def plot_value_function(V, mdp, scale=0.5, title=None):
    num_states = mdp.num_states
    plt.figure(figsize=((2+num_states)*scale, 3*scale))
    
    sns.heatmap(V.reshape((1,-1)), annot=True, fmt=".2f", cmap="RdYlGn", 
                xticklabels=mdp.state_names, cbar=False)
    if title:
        plt.title(title)
    else:
        plt.title("Values")
    plt.xlabel("State")

    plt.tight_layout()
    plt.show()

def plot_q_function(
        Q, mdp, scale=0.5, title=None, 
        exclude_absorbing=True, states_as_rows=True):
    state_names = mdp.state_names
    action_names = mdp.action_names
    
    if exclude_absorbing:
        filter_ = ~np.array(mdp.absorbing)
        state_names = np.array(state_names)[filter_]
        Q = Q[filter_,:]

    num_states = state_names.size
    num_actions = mdp.num_actions    
    action_span = (1+num_actions)*scale
    state_span = (2+num_states)*scale
    
    if states_as_rows:
        plt.figure(figsize=(action_span, state_span)) 
        xticklabels = action_names
        yticklabels = state_names
        xlabel = "Action"
        ylabel = "State"
        matrix = Q
    else:
        # actions as rows
        plt.figure(figsize=(state_span, action_span))
        xticklabels = state_names
        yticklabels = action_names
        xlabel = "State"
        ylabel = "Action"
        matrix = Q.T

    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", 
                xticklabels=xticklabels, yticklabels=yticklabels, cbar=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    else:
        plt.title(f"Q-Values")

    plt.tight_layout()
    plt.show()

def report_mdp_summary(mdp, round_to=3):
    # Summarise the basic MDP properties
    mdp_data = {
        "Property": ["States", "Actions", "State Names", "Action Names", "Initial State Distribution"],
        "Value": [
            mdp.num_states,
            mdp.num_actions,
            ", ".join(mdp.state_names),
            ", ".join(mdp.action_names),
            np.round(mdp.initial, round_to)
        ]
    }
    return pd.DataFrame(mdp_data)
    
    
def line_plot_value_convergence(
        num_episodes_seq, all_v_estimates, V_trusted, state_names,
        target_name="DP", experiment_name="MC"):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Value Estimates per State ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(state_names)))
    
    # Plot MC estimates (dots)
    for i, state in enumerate(state_names):
        ax1.plot(num_episodes_seq, all_v_estimates[:, i], 
                 label=f'{experiment_name}: {state}', color=colors[i], marker='o', markersize=4)

    # Plot Baselines (dashed lines)
    for i, state in enumerate(state_names):
        ax1.axhline(
            y=V_trusted[i], color=colors[i], linestyle='--', alpha=0.6,
            label=f'{target_name}: {state}')

    ax1.set_title(f"State Value Estimates: {experiment_name} vs {target_name}")
    ax1.set_xlabel("Number of Episodes")
    ax1.set_ylabel("Value $V(s)$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small', ncol=2)

    # --- Plot 2: Learning Curve (RMSE) ---
    # Formula for RMSE: $\sqrt{\frac{1}{N} \sum (\hat{V} - V)^2}$
    errors = all_v_estimates - V_trusted
    rmse = np.sqrt(np.mean(errors**2, axis=1))

    ax2.plot(num_episodes_seq, rmse, color='crimson', lw=2, marker='s')
    ax2.set_title(f"Convergence Error (RMSE relative to {target_name})")
    ax2.set_xlabel("Number of Episodes")
    ax2.set_ylabel("RMSE")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
# deprecated
def plot_dual_action_policy_stages(
        fig, num_axes, axis_number, policy, state_names, action_names,
        title):
    # given a figure object plots the next policy in the next axis
    num_states = len(state_names)
    num_actions = len(action_names)
    ax = fig.add_subplot(num_axes, 1, axis_number+1)
    dom_actions = dominant_actions(policy)
    ax.plot(np.arange(num_states), dom_actions)
    ax.set_title(title)
    ax.set_xticks(np.arange(num_states))
    ax.set_xticklabels(state_names)
    ax.set_yticks(np.arange(num_actions))
    ax.set_yticklabels(action_names)
    return ax


    
