# Full script: Multiscale Bayesian particle filter with time scaling and RMSE/plotting output
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(42)

# Global parameters
D = 4
T1 = 51
T2 = 21
T3 = 11
Nx1 = 3
Nx2 = 3
Nx3 = 3
K_env = np.array([12.0, 8.0, 10.0])

B = np.zeros((D, D))  # Initialize as all zeros
# Assign interactions based on environment type
B[0, 2] = 1  # High trait value, favorable with Low trait value, favorable
B[2, 0] = 1
B[1, 3] = 1  # High trait value, harsh with Low trait value, harsh
B[3, 1] = 1

noise_S = np.random.normal(0, 0.0001, (D, T1))
noise_R = np.random.normal(0, 0.0001, (D, T1))
noise_P = np.random.normal(0, 0.0001, (D, T1))

# Scaling factors for environmental influence
beta_2_x2 = np.array([0.85, 0.9, 0.88, 0.72])

K0 = np.zeros(D)
K0 = np.array(
    [
        40,  # Individual 0: long-lived, large-bodied
        20,  # Individual 1: short-lived, early maturity
        30,  # Individual 2: intermediate
        15,  # Individual 3: small-bodied, harsh habitat
    ]
)

K = np.ones((D, T1))
for d in range(D):
    K[d, 0] = K0[d]

    # Intrinsic growth rates
r = np.zeros(D)
r = np.array([0.07, 0.04, 0.06, 0.03])

# Reproductive efficiency constants
a = np.zeros(D)
a[0] = 0.7  # High trait value, favorable
a[1] = 0.4  # High trait value, harsh
a[2] = 0.5  # Low trait value, favorable
a[3] = 0.3  # Low trait value, harsh

A2_t2 = np.array(
    [
        [0.7, 0.0, 0.3],
        [0.0, 1.0, -0.4],
        [0.3, -0.4, 1.0],
    ]
)
# Environmental influence coefficients
c_E = np.zeros(D)
c_E[0] = 0.6  # High trait value, favorable
c_E[1] = 0.5  # High trait value, harsh
c_E[2] = 0.6  # Low trait value, favorable
c_E[3] = 0.5  # Low trait value, harsh


epsilon = 0.5
inv_epsilon = 1 / epsilon

obs_noise_x1 = 0.001
obs_noise_x2 = 0.01
obs_noise_x3 = 0.01

N_particles = 50

# Initial trait baselines
mu_0 = np.array(
    [
        [14.0, 10.0, 7.0],  # Individual 1: long-lived
        [8.0, 6.5, 3.5],  # Individual 2: short-lived
        [10.0, 7.0, 6.0],  # Individual 3: intermediate
        [6.0, 5.0, 9.5],  # Individual 4: small-bodied, harsher conditions
    ]
)

# Environmental parameters
alpha_d = np.array([1.0, 1.0, 1.0, 1.0])
Sigma = np.eye(Nx3)
r_x3 = np.array([0.4, 0.8, 0.6])
K_x3 = np.array([25.0, 18.0, 12.0])

# Weights
weights_x1 = np.random.rand(D, T2, T1)
weights_x1 /= weights_x1.sum(axis=(1, 2), keepdims=True)

weights_x2 = np.random.rand(D, T2)
weights_x2 /= weights_x2.sum(axis=1, keepdims=True)
# Set output folder
output_folder = os.path.expanduser("~/Desktop/Multiscale_PF")
os.makedirs(output_folder, exist_ok=True)

weights_forx1_inx2 = np.random.rand(D, T2, T1)
weights_forx1_inx2 /= weights_forx1_inx2.sum(
    axis=2, keepdims=True
)  # normalize over T1 only, per t2


def save_rmse_plot_over_time(true_data, est_data, time_axis, title, ylabel, filename):
    """Save RMSE plot over specified time axis."""
    rmse = np.sqrt(np.mean((true_data - est_data) ** 2, axis=0))
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rmse)), rmse)
    plt.title(f"RMSE over {time_axis} - {title}")
    plt.xlabel(f"{time_axis} steps")
    plt.ylabel(f"RMSE {ylabel}")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def plot_and_save_state_vs_estimate(
    true_vals, est_vals, time_points, title, ylabel, filename
):
    """Plot and save true vs estimated values over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, true_vals, label="True", linewidth=2)
    plt.plot(time_points, est_vals, label="Estimated", linestyle="--")
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def compute_phi(
    x1, x2, weights_x1, weights_x2, alpha_d, Sigma, t3, assigned_d, component
):
    """
    Individuals close to the mean contribute more (Gaussian),
    with extra penalty if below the mean.
    Large individuals consume more (sigmoid).
    Trait values are averaged from x1 and (optionally) x2 combined.
    """

    D, T3, T2, T1, Nx1 = x1.shape
    _, _, _, Nx2 = x2.shape

    include_x2 = False  # Set to False to exclude x2 contribution

    # Compute population-level combined values (x1 [+ x2]) at t3
    all_combined_vals = []
    for d_all in range(D):
        x1_avg = np.mean(x1[d_all, t3 - 1], axis=(0, 1))  # shape (Nx1,)
        if include_x2:
            x2_avg = np.mean(x2[d_all, t3 - 1], axis=0)  # shape (Nx2,)
            combined = x1_avg[0] + x2_avg[0]
        else:
            combined = x1_avg[0]
        all_combined_vals.append(combined)

    all_combined_vals = np.array(all_combined_vals)
    pop_mean = np.mean(all_combined_vals)
    pop_std = np.std(all_combined_vals)
    if pop_std == 0:
        pop_std = 1.0

    sharpness = 1.5 / pop_std  # for sigmoid

    contribution_score = 0
    consumption_score = 0

    for d in assigned_d:
        x1_avg = np.mean(x1[d, t3 - 1], axis=(0, 1))
        if include_x2:
            x2_avg = np.mean(x2[d, t3 - 1], axis=0)
            combined_size = x1_avg[0] + x2_avg[0]
        else:
            combined_size = x1_avg[0]

        deviation = combined_size - pop_mean

        # Asymmetric Gaussian contribution
        if deviation < 0:
            contrib = np.exp(-0.5 * (deviation / (0.5 * pop_std)) ** 2)
        else:
            contrib = np.exp(-0.5 * (deviation / (1.5 * pop_std)) ** 2)

        # Sigmoid consumption: bigger individuals consume more
        consump = 1 / (1 + np.exp(-sharpness * deviation))

        contribution_score += alpha_d[d] * contrib
        consumption_score += alpha_d[d] * consump

        print(f"[t3 = {t3-1} | comp = {component}] Individual {d}")
        print(f"  Combined size val : {combined_size:.4f}")
        print(f"  Deviation from μ  : {deviation:.4f}")
        print(f"  Contribution (Asymmetric Gaussian) : {contrib:.4f}")
        print(f"  Consumption (Sigmoid)              : {consump:.4f}")
        print("")

    return contribution_score, consumption_score

    # 🔍 Debug info
    # print(f"[t3 = {t3} | comp = {component}] Individual {d}")
    # print(f"  Trait vector      : {trait_vec}")
    # print(f"  Deviation from μ  : {deviation}")
    # print(f"  Mahalanobis dist² : {dist:.4f}")


# State transitions
def f_x3(x3_prev, x1, x2, t3):
    if t3 >= T3 // 2:
        new_environment = np.array([12.0, 8.0, 30.0])
    else:
        new_environment = np.array([27.0, 15.0, 10.0])

    # Compute phi values (now contribution and consumption)
    contrib_1, consump_1 = compute_phi(
        x1, x2, weights_x1, weights_x2, alpha_d, Sigma, t3, assigned_d=[0], component=0
    )
    contrib_2, consump_2 = compute_phi(
        x1, x2, weights_x1, weights_x2, alpha_d, Sigma, t3, assigned_d=[2], component=1
    )
    contrib_3, consump_3 = compute_phi(
        x1,
        x2,
        weights_x1,
        weights_x2,
        alpha_d,
        Sigma,
        t3,
        assigned_d=[1, 3],
        component=2,
    )

    contribution = np.array([contrib_1, contrib_2, contrib_3])
    consumption = np.array([consump_1, consump_2, consump_3])

    # Regeneration (e.g., logistic resource growth)
    regeneration_contribution = r_x3 * x3_prev * (1 - x3_prev / K_x3)

    # Updated environmental state
    return epsilon * (
        new_environment
        + regeneration_contribution
        + (5.0 * contribution)
        - (5.0 * consumption)
    )


def f_x2(x2_prev, mu_t2, x3_effect, A2_t2, alpha_x2=0.5, beta=0.8):
    return x2_prev + alpha_x2 * (mu_t2 - A2_t2 @ x2_prev) + x3_effect


def f_x1(
    x1_prev,
    x2,
    x3_component,
    interaction_term,
    d,
    K0,
    r,
    c_E,
    noise_S,
    noise_R,
    noise_P,
):
    s_prev, r_prev, p_prev = x1_prev
    interaction_modifier = 0.1 * interaction_term
    growth_capacity = 1 - x1_prev[0] / (
        K0[d] - interaction_modifier
    )  # can remove the -interactiom modifier
    K_env = np.array([12.0, 8.0, 10.0])  # thresholds per component

    if d == 0:
        k_env = K_env[0]
    elif d == 2:
        k_env = K_env[1]
    elif d in [1, 3]:
        k_env = K_env[2]

    # Michaelis-Menten environmental effect
    env_effect = x3_component / (x3_component + k_env)

    size = (
        s_prev
        + inv_epsilon
        * (r[d] * (s_prev + 0.6 * x2[0]) * growth_capacity * c_E[d] * env_effect)
    ) + np.sqrt(inv_epsilon) * noise_S

    # Reproductive effort
    R_max, k, gamma = 1.0, 0.05, 0.03
    t_star = 0.75 * (K0[d])
    env_effect = 0.05
    size_effect = 1 / (1 + np.exp(-k * (s_prev - t_star)))
    hereditary_effect = gamma * x2[1]
    environment_effect = env_effect * (x3_component / (x3_component + 1))
    reproductive_effort_change = (
        R_max * size_effect * hereditary_effect * environment_effect * x1_prev[2]
    )
    effort = (
        0.9 * x1_prev[1]  # 0.9 for stabilitiy
        + inv_epsilon * reproductive_effort_change
    )
    effort = np.clip(effort + (np.sqrt(inv_epsilon) * noise_R), 0, 1)

    # Survival
    beta_size, beta_traits, beta_environment, beta_repro = 1, 1, 1, 20
    s_star = 0.75 * (K0[d])
    k_size = 0.1
    size_eff = 1 / (1 + np.exp(-k_size * (s_prev - s_star)))
    protection_score = (
        beta_environment * x3_component
        + beta_traits * x2[2]
        + beta_size * s_prev
        - beta_repro * x1_prev[1]
    )
    # survival = p_prev * inv_epsilon * np.exp(-1 / protection_score)
    survival = p_prev * np.exp(-inv_epsilon / protection_score)

    survival = np.clip(survival + np.sqrt(inv_epsilon) * noise_P, 0, 1)

    return np.array([size, effort, survival])


def f_x1_nonoise(
    x1_prev,
    x2,
    x3_component,
    interaction_term,
    d,
    K0,
    r,
    c_E,
):
    s_prev, r_prev, p_prev = x1_prev
    interaction_modifier = 0.1 * interaction_term
    growth_capacity = 1 - x1_prev[0] / (
        K0[d] - interaction_modifier
    )  # can remove the -interactiom modifier
    K_env = np.array([12.0, 8.0, 10.0])  # thresholds per component

    if d == 0:
        k_env = K_env[0]
    elif d == 2:
        k_env = K_env[1]
    elif d in [1, 3]:
        k_env = K_env[2]

    # Michaelis-Menten environmental effect
    env_effect = x3_component / (x3_component + k_env)

    size = s_prev + inv_epsilon * (
        r[d] * (s_prev + 0.6 * x2[0]) * growth_capacity * c_E[d] * env_effect
    )

    # Reproductive effort
    R_max, k, gamma = 1.0, 0.05, 0.03
    t_star = 0.75 * (K0[d])
    env_effect = 0.05
    size_effect = 1 / (1 + np.exp(-k * (s_prev - t_star)))
    hereditary_effect = gamma * x2[1]
    environment_effect = env_effect * (x3_component / (x3_component + 1))
    reproductive_effort_change = (
        R_max * size_effect * hereditary_effect * environment_effect * x1_prev[2]
    )
    effort = (
        0.9 * x1_prev[1]  # 0.9 for stabilitiy
        + inv_epsilon * reproductive_effort_change
    )
    effort = np.clip(effort, 0, 1)

    # Survival
    beta_size, beta_traits, beta_environment, beta_repro = 1, 1, 1, 20
    s_star = 0.75 * (K0[d])
    k_size = 0.1
    size_eff = 1 / (1 + np.exp(-k_size * (s_prev - s_star)))
    protection_score = (
        beta_environment * x3_component
        + beta_traits * x2[2]
        + beta_size * s_prev
        - beta_repro * x1_prev[1]
    )
    # print(f"[d = {d}] Protection Score: {protection_score:.4f}")

    # survival = p_prev * inv_epsilon * np.exp(-1 / protection_score)
    survival = p_prev * np.exp(-inv_epsilon / protection_score)

    survival = np.clip(survival, 0, 1)

    return np.array([size, effort, survival])


# Generate true states and measurements
def generate_true_data():
    # Initialize states
    x1 = np.zeros((D, T3, T2, T1, Nx1))
    x2 = np.zeros((D, T3, T2, Nx2))
    x3 = np.zeros((T3, Nx3))  # Environmental state over time
    x1_nonoise = np.zeros((D, T3, T2, T1, Nx1))
    x2_nonoise = np.zeros((D, T3, T2, Nx2))
    x3_nonoise = np.zeros((T3, Nx3))  # Environmental state over time
    y1 = np.zeros_like(x1)
    y2 = np.zeros_like(x2)
    y3 = np.zeros_like(x3)

    # Constants (same as before)
    K0 = np.array([40, 20, 30, 15])
    r = np.array([0.07, 0.04, 0.06, 0.03])
    c_E = np.zeros(D)
    c_E[0] = 0.6  # High trait value, favorable
    c_E[1] = 0.5  # High trait value, harsh
    c_E[2] = 0.6  # Low trait value, favorable
    c_E[3] = 0.5  # Low trait value, harsh

    beta_2_x2 = np.array([0.85, 0.9, 0.88, 0.72])
    A2_t2 = np.array(
        [
            [0.7, 0.0, 0.3],  # component 1 pulls mostly on itself
            [0.0, 1.0, -0.4],
            [0.3, -0.4, 1.0],
        ]
    )
    w2 = np.random.normal(0, 0.05, (D, T2, Nx2))
    w3 = np.random.normal(0, 0.01, (T3, Nx3))
    noise_S = np.random.normal(0, 0.001, (D, T1))
    noise_R = np.random.normal(0, 0.001, (D, T1))
    noise_P = np.random.normal(0, 0.001, (D, T1))

    # Initialize x1 and x2 (same as before)
    x1[0, :, :, 0, 0] = 10
    x1[1, :, :, 0, 0] = 5
    x1[2, :, :, 0, 0] = 10
    x1[3, :, :, 0, 0] = 5
    x1[:, :, :, 0, 1] = 0.1
    x1[:, :, :, 0, 2] = 1.0
    x2[:, :, 0, :] = np.expand_dims(mu_0, axis=1) + np.random.normal(
        0, 0.1, size=(D, T3, 3)
    )

    x1_nonoise[0, :, :, 0, 0] = 10
    x1_nonoise[1, :, :, 0, 0] = 5
    x1_nonoise[2, :, :, 0, 0] = 10
    x1_nonoise[3, :, :, 0, 0] = 5
    x1_nonoise[:, :, :, 0, 1] = 0.1
    x1_nonoise[:, :, :, 0, 2] = 1.0
    x2_nonoise[:, :, 0, :] = np.expand_dims(mu_0, axis=1) + np.random.normal(
        0, 0.1, size=(D, T3, 3)
    )

    # Initialize x3[0] (initial environmental conditions)
    x3[0, :] = np.array([27.0, 15.0, 10.0])
    x3_nonoise[0, :] = np.array([27.0, 15.0, 10.0])
    contrib_scores = np.zeros((T3, 3))  # 3 components
    consump_scores = np.zeros((T3, 3))

    # Loop through T3 and update dynamics, including environmental transition
    for t3 in range(1, T3):

        if t3 >= T3 // 2:
            new_environment = np.array([12.0, 8.0, 30.0])
        else:
            new_environment = x3[0, :]

        # ---- Compute contribution & consumption ----
        contrib_1, consump_1 = compute_phi(
            x1,
            x2,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[0],
            component=0,
        )
        contrib_2, consump_2 = compute_phi(
            x1,
            x2,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[2],
            component=1,
        )
        contrib_3, consump_3 = compute_phi(
            x1,
            x2,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[1, 3],
            component=2,
        )

        contribution = np.array([contrib_1, contrib_2, contrib_3])
        consumption = np.array([consump_1, consump_2, consump_3])
        regeneration_contribution = r_x3 * x3[t3 - 1, :] * (1 - x3[t3 - 1, :] / K_x3)

        # ---- Noiseless version for x3_nonoise ----
        contrib_1_nonoise, consump_1_nonoise = compute_phi(
            x1_nonoise,
            x2_nonoise,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[0],
            component=0,
        )
        contrib_2_nonoise, consump_2_nonoise = compute_phi(
            x1_nonoise,
            x2_nonoise,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[2],
            component=1,
        )
        contrib_3_nonoise, consump_3_nonoise = compute_phi(
            x1_nonoise,
            x2_nonoise,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[1, 3],
            component=2,
        )

        contribution_nonoise = np.array(
            [contrib_1_nonoise, contrib_2_nonoise, contrib_3_nonoise]
        )
        consumption_nonoise = np.array(
            [consump_1_nonoise, consump_2_nonoise, consump_3_nonoise]
        )

        contrib_scores[t3, 0], consump_scores[t3, 0] = compute_phi(
            x1,
            x2,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[0],
            component=0,
        )
        contrib_scores[t3, 1], consump_scores[t3, 1] = compute_phi(
            x1,
            x2,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[2],
            component=1,
        )
        contrib_scores[t3, 2], consump_scores[t3, 2] = compute_phi(
            x1,
            x2,
            weights_x1,
            weights_x2,
            alpha_d,
            Sigma,
            t3,
            assigned_d=[1, 3],
            component=2,
        )
        regeneration_contribution_nonoise = (
            r_x3 * x3_nonoise[t3 - 1, :] * (1 - x3_nonoise[t3 - 1, :] / K_x3)
        )

        # ---- Update environmental states ----
        x3[t3, :] = (
            epsilon
            * (
                new_environment
                + regeneration_contribution
                + (5.0 * contribution)
                - (5.0 * consumption)
            )
            + epsilon * w3[t3 - 1, :]
        )

        x3_nonoise[t3, :] = epsilon * (
            new_environment
            + regeneration_contribution_nonoise
            + (5.0 * contribution_nonoise)
            - (5.0 * consumption_nonoise)
        )

        # print(
        #    f"t3 = {t3}, Component 1 = {x3[t3, 0]}, Component 2 = {x3[t3, 1]}, Component 3 = {x3[t3, 2]}"
        # )

        # Further calculations for x1, x2 dynamics (same as before)
        for t2 in range(1, T2):
            for d in range(D):
                trait_scaling = np.array(
                    [1.0, 40.0, 40.0]
                )  # scale dims 2 and 3 to match dim 1

                # Self-effect (mean over T1 without weights)
                weighted_sum_x1 = np.mean(x1[d, t3, t2 - 1, :, :], axis=0)
                # print(
                #   "Raw values of x1[d, t3, t2-1, :, 0]:", x1[d, t3, t2 - 1, :, 0]
                # )  # Size dim
                # print("Mean before scaling:", np.mean(x1[d, t3, t2 - 1, :, 0]))

                # Interaction effect (mean over T1 for each interacting individual, then sum)
                interaction_term_x2 = np.sum(
                    [
                        B[d, d_prime] * np.mean(x1[d_prime, t3, t2 - 1, :, :], axis=0)
                        for d_prime in range(D)
                        if d_prime != d
                    ],
                    axis=0,
                )

                # Apply scaling to each trait dimension
                weighted_sum_x1_scaled = trait_scaling * weighted_sum_x1
                interaction_term_x2_scaled = trait_scaling * interaction_term_x2

                # Self-effect (mean over T1 without weights)
                weighted_sum_x1_nonoise = np.mean(
                    x1_nonoise[d, t3, t2 - 1, :, :], axis=0
                )

                # Interaction effect (mean over T1 for each interacting individual, then sum)
                interaction_term_x2_nonoise = np.sum(
                    [
                        B[d, d_prime]
                        * np.mean(x1_nonoise[d_prime, t3, t2 - 1, :, :], axis=0)
                        for d_prime in range(D)
                        if d_prime != d
                    ],
                    axis=0,
                )

                # Apply trait scaling
                weighted_sum_x1_scaled_nonoise = trait_scaling * weighted_sum_x1_nonoise
                interaction_term_x2_scaled_nonoise = (
                    trait_scaling * interaction_term_x2_nonoise
                )

                # Combine with baseline and scale
                combined_term = 0.5 * (
                    weighted_sum_x1_scaled + interaction_term_x2_scaled
                )
                # print(f"[t2 = {t2} | d = {d}]")
                # print(f"  weighted_sum_x1     : {weighted_sum_x1}")
                # print(f"  interaction_term_x2 : {interaction_term_x2}")
                # print(f"  combined_term       : {combined_term}")
                # print("")

                mu_t2 = mu_0[d] + 0.8 * combined_term
                if d == 0:
                    x3_effect = beta_2_x2[d] * x3[t3, 0]
                elif d == 2:
                    x3_effect = beta_2_x2[d] * x3[t3, 1]
                elif d in [1, 3]:
                    x3_effect = beta_2_x2[d] * x3[t3, 2]

                combined_term_nonoise = 0.5 * (
                    weighted_sum_x1_scaled_nonoise + interaction_term_x2_scaled_nonoise
                )
                mu_t2 = mu_0[d] + 0.8 * combined_term_nonoise
                if d == 0:
                    x3_effect_nonoise = beta_2_x2[d] * x3_nonoise[t3, 0]
                elif d == 2:
                    x3_effect_nonoise = beta_2_x2[d] * x3_nonoise[t3, 1]
                elif d in [1, 3]:
                    x3_effect_nonoise = beta_2_x2[d] * x3_nonoise[t3, 2]

                # x3_effect = beta_2_x2[d] * x3[t3, d % Nx3]
                x2[d, t3, t2] = (
                    f_x2(x2[d, t3, t2 - 1], mu_t2, x3_effect, A2_t2) + w2[d, t2 - 1]
                )

                x2_nonoise[d, t3, t2] = f_x2(
                    x2[d, t3, t2 - 1], mu_t2, x3_effect_nonoise, A2_t2
                )

            for t1 in range(1, T1):
                for d in range(D):
                    if d == 0:  # Individuals 0 and 2 use x3[t3, 0]
                        x3_component = x3[t3, 0]
                        x3_component_nonoise = x3_nonoise[t3, 0]
                    elif d == 2:  # Specific value for individual 2
                        x3_component = x3[t3, 1]
                        x3_component_nonoise = x3_nonoise[t3, 1]
                    elif d == 1:  # Individual 1 uses x3[t3, 1]
                        x3_component = x3[t3, 2]
                        x3_component_nonoise = x3_nonoise[t3, 2]
                    else:  # Default for other individuals (e.g., Individual 3 uses x3[t3, 2])
                        x3_component = x3[t3, 2]
                        x3_component_nonoise = x3_nonoise[t3, 2]
                    interaction_term = np.sum(B[d, :] * x1[:, t3, t2, t1 - 1, 0])
                    interaction_term_nonoise = np.sum(
                        B[d, :] * x1_nonoise[:, t3, t2, t1 - 1, 0]
                    )

                    x1[d, t3, t2, t1] = f_x1(
                        x1[d, t3, t2, t1 - 1],
                        x2[d, t3, t2],
                        x3_component,
                        interaction_term,
                        d,
                        K0,
                        r,
                        c_E,
                        noise_S[d, t1 - 1],
                        noise_R[d, t1 - 1],
                        noise_P[d, t1 - 1],
                    )
                    x1_nonoise[d, t3, t2, t1] = f_x1_nonoise(
                        x1_nonoise[d, t3, t2, t1 - 1],
                        x2_nonoise[d, t3, t2],
                        x3_component_nonoise,
                        interaction_term_nonoise,
                        d,
                        K0,
                        r,
                        c_E,
                    )

    # Add observational noise
    y1 = x1 + np.random.normal(0, obs_noise_x1, x1.shape)
    y2 = x2 + np.random.normal(0, obs_noise_x2, x2.shape)
    y3 = x3 + np.random.normal(0, obs_noise_x3, x3.shape)

    return (
        x1,
        x2,
        x3,
        x1_nonoise,
        x2_nonoise,
        x3_nonoise,
        y1,
        y2,
        y3,
        contrib_scores,
        consump_scores,
    )


from pathlib import Path

# ========== CONTINUATION: Utility Functions for Particle Filter and Visualization ========== #


def initialize_particles(num_particles, shape, init_mean=0.0, init_std=1.0):
    """Initialize particles from a Gaussian distribution."""
    return np.random.normal(init_mean, init_std, size=(num_particles,) + shape)


def measurement_model(x_true, noise_std=0.1):
    """Generate noisy measurements from the true state."""
    return x_true + np.random.normal(0, noise_std, size=x_true.shape)


def plot_states_vs_estimates(
    true_states, estimated_states, title, ylabel, time_label, filename=None
):
    """Plot true vs. estimated states with optional file saving."""
    fig, ax = plt.subplots(figsize=(10, 6))
    T = true_states.shape[0]
    ax.plot(range(T), true_states, label="True", linewidth=2)
    ax.plot(range(T), estimated_states, label="Estimated", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(time_label)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
    plt.show()


def resample_particles(weights, particles):
    """Resample particles based on normalized weights."""
    N = particles.shape[0]
    indices = np.random.choice(N, size=N, p=weights)
    return particles[indices]


def normalize_weights(log_weights):
    """Numerically stable weight normalization."""
    max_log_w = np.max(log_weights)
    weights = np.exp(log_weights - max_log_w)
    weights /= np.sum(weights)
    return weights


def compute_rmse(true, estimate):
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((true - estimate) ** 2))


# ========== NEXT: Particle Filtering Algorithm ========== #


def multiscale_particle_filter(
    y1,
    y2,
    y3,
    K0,
    r,
    c_E,
    A2_t2,
    beta_2_x2,
    mu_0,
    weights_x1,
    weights_x2,
    noise_std_x1=0.001,
    noise_std_x2=0.001,
    noise_std_x3=0.001,
    N_particles=50,
):
    # Dimensions
    D, T3, T2, T1, Nx1 = y1.shape
    Nx2 = y2.shape[-1]
    Nx3 = y3.shape[-1]

    # Initialize arrays for particle means (estimated states)
    x1_est = np.zeros((D, T3, T2, T1, Nx1))
    x2_est = np.zeros((D, T3, T2, Nx2))
    x3_est = np.zeros((T3, Nx3))

    # Initialize particles
    x1_part = np.random.normal(0, 1, (N_particles, D, T3, T2, T1, Nx1))
    x2_part = np.random.normal(0, 1, (N_particles, D, T3, T2, Nx2))
    x3_part = np.random.normal(0, 1, (N_particles, T3, Nx3))

    # Initial particles set from first observations
    x1_part[:, :, :, :, 0, :] = y1[:, :, :, 0, :] + np.random.normal(
        0, noise_std_x1, (N_particles, D, T3, T2, Nx1)
    )
    x2_part[:, :, :, 0, :] = y2[:, :, 0, :] + np.random.normal(
        0, noise_std_x2, (N_particles, D, T3, Nx2)
    )
    x3_part[:, 0, :] = y3[0, :] + np.random.normal(0, noise_std_x3, (N_particles, Nx3))

    # Loop over time
    for t3 in range(1, T3):
        # Apply the environmental change after T3 // 2
        if t3 >= T3 // 2:  # Switch environment at half the timeline
            new_environment = np.array([12.0, 8.0, 30.0])  # New environment
        else:
            new_environment = np.array(
                [27.0, 15.0, 10.0]
            )  # Previous environment for earlier steps

        weights_x3_part = np.zeros(N_particles)
        for n in range(N_particles):
            x3_pred = f_x3(
                x3_part[n, t3 - 1], x1_part[n], x2_part[n], t3
            ) + epsilon * np.random.normal(0, noise_std_x3, Nx3)

            x3_part[n, t3] = x3_pred
            weights_x3_part[n] = (
                -0.5 * np.sum((y3[t3] - x3_pred) ** 2) / (noise_std_x3**2)
            )

        # Normalize and resample particles for x3
        w_x3_norm = normalize_weights(weights_x3_part)
        x3_part[:, t3] = resample_particles(w_x3_norm, x3_part[:, t3])
        x3_est[t3] = np.average(x3_part[:, t3], axis=0, weights=w_x3_norm)

        for t2 in range(1, T2):
            for d in range(D):
                trait_scaling = np.array([1.0, 40.0, 40.0])
                weights_x2_part = np.zeros(N_particles)
                for n in range(N_particles):
                    # weighted_sum_x1 = T1 * np.sum(
                    #   weights_forx1_inx2[d, t2, :, None]
                    #  * x1_part[n, d, t3, t2, :, :],
                    # axis=0,
                    # )
                    mean_x1_over_t1 = np.mean(x1_part[n, d, t3, t2 - 1, :, :], axis=0)

                    # Compute interaction term from other individuals' x1
                    # interaction_term_x2 = np.sum(
                    #    [
                    #       B[d, d_prime]
                    #      * T1
                    #     * np.sum(
                    #        weights_forx1_inx2[d_prime, t2, :, None]
                    #       * x1_part[n, d_prime, t3, t2, :, :],
                    #      axis=0,
                    # )
                    # for d_prime in range(D)
                    # if d_prime != d
                    # ],
                    # axis=0,
                    # )
                    interaction_term_x2 = np.sum(
                        [
                            B[d, d_prime]
                            * np.mean(
                                x1_part[n, d_prime, t3, t2 - 1, :, :],
                                axis=0,
                            )
                            for d_prime in range(D)
                            if d_prime != d
                        ],
                        axis=0,
                    )

                    # Combine individual and interaction effects
                    weighted_sum_x1_scaled = trait_scaling * mean_x1_over_t1
                    interaction_term_x2_scaled = trait_scaling * interaction_term_x2

                    combined_term = 0.5 * (
                        weighted_sum_x1_scaled + interaction_term_x2_scaled
                    )

                    # Add baseline and scale
                    mu_t2 = mu_0[d] + 0.8 * combined_term
                    # x3_effect = beta_2_x2[d] * x3_part[n, t3, d % Nx3]
                    if d == 0:
                        x3_effect = beta_2_x2[d] * x3_part[n, t3, 0]
                    elif d == 2:
                        x3_effect = beta_2_x2[d] * x3_part[n, t3, 1]
                    elif d in [1, 3]:
                        x3_effect = beta_2_x2[d] * x3_part[n, t3, 2]

                    x2_pred = f_x2(
                        x2_part[n, d, t3, t2 - 1], mu_t2, x3_effect, A2_t2
                    ) + np.random.normal(0, noise_std_x2, Nx2)

                    x2_part[n, d, t3, t2] = x2_pred
                    weights_x2_part[n] = (
                        -0.5
                        * np.sum((y2[d, t3, t2] - x2_pred) ** 2)
                        / (noise_std_x2**2)
                    )

                w_x2_norm = normalize_weights(weights_x2_part)
                x2_part[:, d, t3, t2] = resample_particles(
                    w_x2_norm, x2_part[:, d, t3, t2]
                )
                x2_est[d, t3, t2] = np.average(
                    x2_part[:, d, t3, t2], axis=0, weights=w_x2_norm
                )

            for t1 in range(1, T1):
                for d in range(D):
                    weights_x1_part = np.zeros(N_particles)
                    for n in range(N_particles):
                        if d == 0:  # Individuals 0 and 2 use x3[t3, 0]
                            x3_component = x3_part[n, t3, 0]
                        elif d == 2:  # Specific value for individual 2
                            x3_component = x3_part[n, t3, 1]
                        elif d == 1:  # Individual 1 uses x3[t3, 1]
                            x3_component = x3_part[n, t3, 2]
                        else:  # Default for other individuals (e.g., Individual 3 uses x3[t3, 2])
                            x3_component = x3_part[n, t3, 2]
                        interaction_term = np.sum(
                            B[d, :] * x1_part[n, :, t3, t2, t1 - 1, 0]
                        )
                        x1_pred = f_x1(
                            x1_part[n, d, t3, t2, t1 - 1],
                            x2_part[n, d, t3, t2],
                            x3_component,
                            interaction_term,
                            d,
                            K0,
                            r,
                            c_E,
                            noise_S[d, t1 - 1],
                            noise_R[d, t1 - 1],
                            noise_P[d, t1 - 1],
                        )

                        x1_part[n, d, t3, t2, t1] = x1_pred
                        weights_x1_part[n] = (
                            -0.5
                            * np.sum((y1[d, t3, t2, t1] - x1_pred) ** 2)
                            / (noise_std_x1**2)
                        )

                    w_x1_norm = normalize_weights(weights_x1_part)
                    x1_part[:, d, t3, t2, t1] = resample_particles(
                        w_x1_norm, x1_part[:, d, t3, t2, t1]
                    )
                    x1_est[d, t3, t2, t1] = np.average(
                        x1_part[:, d, t3, t2, t1], axis=0, weights=w_x1_norm
                    )

    return x1_est, x2_est, x3_est


# This block defines reusable functions for particle filtering and visualization.
# In the next message, I'll provide a concrete example using these utilities with your
# model’s generated data (e.g., for a specific variable in x1, x2, or x3). Let me know
# which variable (e.g., x1 size, x2 trait, or x3 environment) you'd like to filter first.
# === RUN THE FULL SIMULATION AND FILTER ===
(
    x1_true,
    x2_true,
    x3_true,
    x1_nonoise,
    x2_nonoise,
    x3_nonoise,
    y1,
    y2,
    y3,
    contrib_scores,
    consump_scores,
) = generate_true_data()
x1_est, x2_est, x3_est = multiscale_particle_filter(
    y1, y2, y3, K0, r, c_E, A2_t2, beta_2_x2, mu_0, weights_x1, weights_x2
)


# Function to create GIFs for x1
def create_x1_gifs(x1_nonoise, x1_est, output_dir):
    labels = ["Size (S)", "Reproductive Effort (R)", "Survival Probability (P)"]
    for dim in range(x1_nonoise.shape[-1]):
        for t3_idx in range(1, x1_nonoise.shape[1]):
            fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
            fig.subplots_adjust(right=0.75)  # extra space for legend

            def update(t2_idx):
                ax.clear()
                if t2_idx == 0:
                    return
                for d in range(x1_nonoise.shape[0]):
                    ax.plot(
                        range(x1_nonoise.shape[3]),
                        x1_nonoise[d, t3_idx, t2_idx, :, dim],
                        label=f"Individual {d+1} True",
                        linewidth=2,
                    )
                    ax.plot(
                        range(x1_est.shape[3]),
                        x1_est[d, t3_idx, t2_idx, :, dim],
                        linestyle="--",
                        label=f"Individual {d+1} Est",
                    )
                ax.set_title(f"Dynamics of {labels[dim]} for t3={t3_idx}, t2={t2_idx}")
                ax.set_xlabel("Time $t_1$")
                ax.set_ylabel(labels[dim])
                ax.grid(True)
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

            anim = FuncAnimation(
                fig,
                update,
                frames=range(1, x1_nonoise.shape[2]),
                repeat=False,
            )
            gif_path = os.path.join(
                output_dir, f"x1_dynamics_dim{dim+1}_t3{t3_idx}.gif"
            )
            anim.save(gif_path, writer=PillowWriter(fps=2))
            plt.close(fig)


def create_x2_gifs(x2_nonoise, x2_est, output_dir):
    labels = [f"Hereditary Factor {i + 1}" for i in range(x2_nonoise.shape[-1])]
    for dim in range(x2_nonoise.shape[-1]):
        # Flatten both arrays across all individuals and time points for this dimension
        all_vals = np.concatenate(
            [x2_nonoise[..., dim].flatten(), x2_est[..., dim].flatten()]
        )
        min_val, max_val = np.min(all_vals), np.max(all_vals)
        if max_val - min_val == 0:
            scale = lambda x: x * 0  # All values are the same
        else:
            scale = lambda x: (x - min_val) / (max_val - min_val) * 30

        fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
        fig.subplots_adjust(right=0.75)

        def update(t3_idx):
            ax.clear()
            for d in range(x2_nonoise.shape[0]):
                ax.plot(
                    range(x2_nonoise.shape[2]),
                    scale(x2_nonoise[d, t3_idx, :, dim]),
                    label=f"Individual {d+1} True",
                    linewidth=2,
                )
                ax.plot(
                    range(x2_est.shape[2]),
                    scale(x2_est[d, t3_idx, :, dim]),
                    linestyle="--",
                    label=f"Individual {d+1} Est",
                )
            ax.set_title(f"Dynamics of {labels[dim]} for $t_3$ = {t3_idx}")
            ax.set_xlabel("Time $t_2$")
            ax.set_ylabel(labels[dim] + " (scaled to [0, 30])")
            ax.set_ylim(0, 35)
            ax.set_xticks(range(x2_nonoise.shape[2]))
            ax.grid(True)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        anim = FuncAnimation(
            fig, update, frames=range(1, x2_nonoise.shape[1]), repeat=False
        )
        gif_path = os.path.join(output_dir, f"x2_dynamics_dim{dim+1}.gif")
        anim.save(gif_path, writer=PillowWriter(fps=2))
        plt.close(fig)


def create_x3_plots(x3_nonoise, x3_est, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 14), constrained_layout=True)
    fig.subplots_adjust(right=0.75)

    for k in range(x3_nonoise.shape[-1]):
        ax = axes[k]
        ax.plot(range(x3_nonoise.shape[0]), x3_nonoise[:, k], label="True")
        ax.plot(range(x3_est.shape[0]), x3_est[:, k], linestyle="--", label="Estimated")
        ax.set_title(f"x3 Component {k+1}")
        ax.set_xlabel("Time $t_3$")
        ax.set_ylabel(f"x3 Component {k+1}")
        ax.grid(True)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        # Add vertical dashed line at t_3 = 5 (regime switch)
        ax.axvline(x=5, color="black", linestyle="--", label="Environmental Switch")

    plot_path = os.path.join(output_dir, "x3_components_plot.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Static plot for x3 saved to {plot_path}")


def compute_rmse_x1(x1_nonoise, x1_est):
    D, T3, T2, T1, Nx1 = x1_nonoise.shape
    rmse = np.zeros((D, Nx1, T3, T2))
    for d in range(D):
        for dim in range(Nx1):
            for t3 in range(1, T3):
                for t2 in range(1, T2):
                    error = x1_nonoise[d, t3, t2, 1:, dim] - x1_est[d, t3, t2, 1:, dim]
                    mse = np.mean(error**2)
                    rmse[d, dim, t3, t2] = np.sqrt(mse)
    return rmse


def compute_rmse_x2(x2_nonoise, x2_est):
    D, T3, T2, Nx2 = x2_nonoise.shape
    rmse = np.zeros((D, Nx2, T3))
    for d in range(D):
        for dim in range(Nx2):
            for t3 in range(1, T3):
                error = x2_nonoise[d, t3, 1:, dim] - x2_est[d, t3, 1:, dim]
                mse = np.mean(error**2)
                rmse[d, dim, t3] = np.sqrt(mse)
    return rmse


def compute_rmse_x3(x3_nonoise, x3_est):
    T3, Nx3 = x3_nonoise.shape
    rmse = np.zeros((Nx3, T3))
    for dim in range(Nx3):
        for t3 in range(1, T3):
            error = x3_nonoise[t3, dim] - x3_est[t3, dim]
            rmse[dim, t3] = np.sqrt(error**2)
    return rmse


def create_rmse_x1_gifs(rmse_x1, output_dir):
    D, Nx1, T3, T2 = rmse_x1.shape
    for dim in range(Nx1):
        for t3_idx in range(1, T3):
            fig, ax = plt.subplots(figsize=(10, 6))

            def update(t2_idx):
                ax.clear()
                if t2_idx == 0:
                    return
                for d in range(D):
                    ax.plot(
                        range(1, T2),
                        rmse_x1[d, dim, t3_idx, 1:],
                        label=f"Individual {d+1}",
                        marker="o",
                    )
                ax.set_title(f"RMSE of x1 Dimension {dim+1} | t3={t3_idx}")
                ax.set_xlabel("Time $t_2$")
                ax.set_ylabel("RMSE")
                ax.set_ylim(0, np.max(rmse_x1[:, dim]) + 0.1)
                ax.legend()
                ax.grid(True)

            anim = FuncAnimation(fig, update, frames=range(1, T2), repeat=False)
            gif_path = os.path.join(output_dir, f"rmse_x1_dim{dim+1}_t3{t3_idx}.gif")
            anim.save(gif_path, writer=PillowWriter(fps=2))
            plt.close(fig)


def create_rmse_x2_gifs(rmse_x2, output_dir):
    D, Nx2, T3 = rmse_x2.shape
    for dim in range(Nx2):
        fig, ax = plt.subplots(figsize=(10, 6))

        def update(t3_idx):
            ax.clear()
            for d in range(D):
                ax.plot(
                    range(1, T3),
                    rmse_x2[d, dim, 1:],
                    label=f"Individual {d+1}",
                    marker="o",
                )
            ax.set_title(f"RMSE of x2 Dimension {dim+1} over Time $t_3$")
            ax.set_xlabel("Time $t_3$")
            ax.set_ylabel("RMSE")
            ax.set_ylim(0, np.max(rmse_x2[:, dim]) + 0.1)
            ax.legend()
            ax.grid(True)

        anim = FuncAnimation(fig, update, frames=range(1, T3), repeat=False)
        gif_path = os.path.join(output_dir, f"rmse_x2_dim{dim+1}_over_t3.gif")
        anim.save(gif_path, writer=PillowWriter(fps=2))
        plt.close(fig)


def create_rmse_x3_plot(rmse_x3, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    for dim in range(rmse_x3.shape[0]):
        ax = axes[dim]
        ax.plot(range(1, rmse_x3.shape[1]), rmse_x3[dim, 1:], label=f"x3[{dim+1}]")
        ax.set_title(f"RMSE of x3 Component {dim+1}")
        ax.set_xlabel("Time $t_3$")
        ax.set_ylabel("RMSE")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "rmse_x3_static.png")
    plt.savefig(path)
    plt.close()
    print(f"RMSE plot for x3 saved to: {path}")


def plot_contrib_consump_per_individual(contrib_scores, consump_scores, output_path):
    """
    Plots contribution and consumption over t3 for each individual (d = 0 to 3),
    with improved aesthetics.
    """
    T3 = contrib_scores.shape[0]
    d_to_component = {0: 0, 1: 2, 2: 1, 3: 2}

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    for d in range(4):
        comp = d_to_component[d]

        scaled_contrib = 5.0 * contrib_scores[:, comp]
        scaled_consump = 5.0 * consump_scores[:, comp]

        axes[d].plot(
            range(T3), scaled_contrib, label="Contribution", color="blue", linewidth=2
        )
        axes[d].plot(
            range(T3), scaled_consump, label="Consumption", color="red", linewidth=2
        )

        display_individual = d + 1
        display_component = comp + 1

        axes[d].set_title(
            f"Individual {display_individual} (Component {display_component})",
            fontsize=12,
        )
        axes[d].set_ylabel("Score", fontsize=10)
        axes[d].grid(True, linestyle="--", alpha=0.5)

        # Lighter, thinner switch line
        axes[d].axvline(x=5, color="black", linestyle="--", linewidth=1)

    axes[-1].set_xlabel("Time $t_3$", fontsize=12)

    # Shared legend outside
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space at top for legend
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Individual contribution/consumption plot saved to: {output_path}")


def evaluate_and_plot_all(
    x1, x2, x3, x1_est, x2_est, x3_est, contrib_scores, consump_scores, output_folder
):
    # Create true vs estimated GIFs
    create_x1_gifs(x1, x1_est, output_folder)
    create_x2_gifs(x2, x2_est, output_folder)
    create_x3_plots(x3, x3_est, output_folder)

    # Compute RMSEs
    rmse_x1 = compute_rmse_x1(x1, x1_est)
    rmse_x2 = compute_rmse_x2(x2, x2_est)
    rmse_x3 = compute_rmse_x3(x3, x3_est)

    # Create RMSE visualizations
    create_rmse_x1_gifs(rmse_x1, output_folder)
    create_rmse_x2_gifs(rmse_x2, output_folder)
    create_rmse_x3_plot(rmse_x3, output_folder)
    plot_contrib_consump_per_individual(
        contrib_scores,
        consump_scores,
        os.path.join(output_folder, "phi_per_individual.png"),
    )

    print(f"All visualizations saved to {output_folder}")


# Main script starts here
if __name__ == "__main__":
    # Create output folder
    save_dir = os.path.expanduser("~/Desktop/Multiscale_PF")
    os.makedirs(save_dir, exist_ok=True)

    # Generate true data
    (
        x1_true,
        x2_true,
        x3_true,
        x1_nonoise,
        x2_nonoise,
        x3_nonoise,
        y1,
        y2,
        y3,
        contrib_scores,
        consump_scores,
    ) = generate_true_data()
    # np.save("x2_nonoise_dev_env.npy", x2_nonoise)  # for dev+env
    # or

    # Run particle filter
    x1_est, x2_est, x3_est = multiscale_particle_filter(
        y1,
        y2,
        y3,
        K0,
        r,
        c_E,
        A2_t2,
        beta_2_x2,
        mu_0,
        weights_x1,
        weights_x2,
        noise_std_x1=obs_noise_x1,
        noise_std_x2=obs_noise_x2,
        noise_std_x3=obs_noise_x3,
        N_particles=N_particles,
    )

    # Create GIFs for true vs estimated states for x1, x2
    evaluate_and_plot_all(
        x1_nonoise,
        x2_nonoise,
        x3_nonoise,
        x1_est,
        x2_est,
        x3_est,
        contrib_scores,
        consump_scores,
        save_dir,
    )
