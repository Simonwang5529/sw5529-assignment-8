import numpy as np
import matplotlib
matplotlib.use('Agg')  # Switch backend to 'Agg' to generate images without a GUI
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8],
                                  [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1) and apply the adjusted shift
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    X2 += np.array([distance, -distance])  # Shift along y = -x
    y2 = np.ones(n_samples)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y
def calculate_slope(beta1, beta2):
    if beta2 == 0:
        return float('inf')  # Return 'infinity' if beta2 is zero to avoid division by zero
    else:
        return -beta1 / beta2
        
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def calculate_logistic_loss(X, y, model):
    y_pred = model.predict_proba(X)
    epsilon = 1e-15
    return -np.mean(y * np.log(y_pred[:, 1] + epsilon) + (1 - y) * np.log(1 - y_pred[:, 1] + epsilon))

def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    sample_data = {}
    
    # Calculate the number of rows needed for the dataset plots
    n_cols = 2
    n_rows = (step_num + n_cols - 1) // n_cols

    # Create figure for dataset visualization
    plt.figure(figsize=(20, 5 * n_rows))

    for i, distance in enumerate(shift_distances):
        X, y = generate_ellipsoid_clusters(distance=distance)
        
        # Fit logistic regression and record parameters
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        
        # Calculate slope using the improved function
        slope = calculate_slope(beta1, beta2)
        slope_list.append(slope)
        
        # Calculate intercept with safety check
        intercept = -beta0 / beta2 if abs(beta2) >= 1e-10 else float('inf')
        intercept_list.append(intercept)

        # Plot dataset (using 1-based indexing for subplot)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class 1')
        
        # Calculate loss
        loss = calculate_logistic_loss(X, y, model)
        loss_list.append(loss)

        # Setup for contour plotting
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept],
                'k-', label='Decision Boundary')

        # Plot confidence contours
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7:
                # Get vertices using the updated method
                class_1_vertices = np.concatenate([path.vertices for path in class_1_contour.get_paths()])
                class_0_vertices = np.concatenate([path.vertices for path in class_0_contour.get_paths()])
                distances = cdist(class_1_vertices, class_0_vertices, metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)

        plt.title(f"Shift Distance = {distance:.2f}", fontsize=24)
        plt.xlabel("x1")
        plt.ylabel("x2")

        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        margin_text = f"Margin Width: {min_distance:.2f}"
        plt.text(x_min + 0.1, y_max - 1.0, equation_text, fontsize=24, color="black", ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.text(x_min + 0.1, y_max - 1.5, margin_text, fontsize=24, color="black", ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        if i == 0:
            plt.legend(loc='lower right', fontsize=20)

        sample_data[distance] = (X, y, model, beta0, beta1, beta2, min_distance)
    
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")
    plt.close()

    # Create new figure for parameter plots
    plt.figure(figsize=(18, 12))

    # Parameter plots in 2x4 grid
    plt.subplot(2, 4, 1)
    plt.plot(shift_distances, beta0_list, 'b-', marker='o')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    plt.subplot(2, 4, 2)
    plt.plot(shift_distances, beta1_list, 'r-', marker='o')
    plt.title("Shift Distance vs Beta1")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    plt.subplot(2, 4, 3)
    plt.plot(shift_distances, beta2_list, 'g-', marker='o')
    plt.title("Shift Distance vs Beta2")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    plt.subplot(2, 4, 4)
    plt.subplot(2, 4, 4)
    valid_slopes = [(d, s) for d, s in zip(shift_distances, slope_list) if abs(s) != float('inf')]
    if valid_slopes:
        distances, slopes = zip(*valid_slopes)
        plt.plot(distances, slopes, 'm-', marker='o')
    plt.title("Shift Distance vs Slope")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope (-Beta1/Beta2)")
    plt.grid(True)

    plt.subplot(2, 4, 5)
    plt.plot(shift_distances, [b0/b2 for b0, b2 in zip(beta0_list, beta2_list)], 'c-', marker='o')
    plt.title("Shift Distance vs Intercept Ratio")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0 / Beta2")

    plt.subplot(2, 4, 6)
    plt.plot(shift_distances, loss_list, 'y-', marker='o')
    plt.title("Shift Distance vs Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    plt.subplot(2, 4, 7)
    plt.plot(shift_distances, margin_widths, 'k-', marker='o')
    plt.title("Shift Distance vs Margin")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    plt.close()

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
