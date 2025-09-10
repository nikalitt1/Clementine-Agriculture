import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ------------------------
# Load CSV
# ------------------------
csv_filename = "/home/nika/RF_Data_Arm3_cleaned.csv"
df = pd.read_csv(csv_filename, header=0)
df.columns = ["angle_x", "angle_y", "pixel_x", "pixel_y", "depth", "pitch", "roll"]

# ------------------------
# Features and targets
# ------------------------
y = df[["angle_x", "angle_y"]].values
X = df[["pixel_x", "pixel_y", "depth", "pitch", "roll"]].values

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Decision Tree Regressor
# ------------------------
tree = DecisionTreeRegressor(max_depth=22, random_state=42)
tree.fit(X_train, y_train)

tree_model_path = "/home/nika/decision_tree_flat_fine2.pkl"
with open(tree_model_path, 'wb') as f:
    pickle.dump(tree, f)

print(f"Decision tree trained and saved to {tree_model_path}")

val_preds_tree = tree.predict(X_val)
mae_angle0_tree = mean_absolute_error(y_val[:, 0], val_preds_tree[:, 0])
mae_angle1_tree = mean_absolute_error(y_val[:, 1], val_preds_tree[:, 1])
avg_mae_tree = (mae_angle0_tree + mae_angle1_tree) / 2

print(f"\nDecision Tree Validation MAE:")
print(f"  angle_x: {mae_angle0_tree:.2f} degrees")
print(f"  angle_y: {mae_angle1_tree:.2f} degrees")
print(f"  Average MAE: {avg_mae_tree:.2f} degrees")

print("\nSample Predictions vs Actual Values for Decision Tree:")
for i in range(10):
    predicted = val_preds_tree[i]
    actual = y_val[i]
    print(f"[{i}] Predicted: angle_x = {predicted[0]:.2f}, angle_y = {predicted[1]:.2f}  |  Actual: angle_x = {actual[0]:.2f}, angle_y = {actual[1]:.2f}")

# ------------------------
# Random Forest Regressor
# ------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_model_path = "/home/nika/RF_Metal_Arm3.pkl"
with open(rf_model_path, 'wb') as f:
    pickle.dump(rf, f)

print(f"Random Forest trained and saved to {rf_model_path}")

val_preds_rf = rf.predict(X_val)
mae_angle0_rf = mean_absolute_error(y_val[:, 0], val_preds_rf[:, 0])
mae_angle1_rf = mean_absolute_error(y_val[:, 1], val_preds_rf[:, 1])
avg_mae_rf = (mae_angle0_rf + mae_angle1_rf) / 2

print(f"\nRandom Forest Validation MAE:")
print(f"  angle_x: {mae_angle0_rf:.2f} degrees")
print(f"  angle_y: {mae_angle1_rf:.2f} degrees")
print(f"  Average MAE: {avg_mae_rf:.2f} degrees")

print("\nSample Predictions vs Actual Values for Random Forest:")
for i in range(10):
    predicted = val_preds_rf[i]
    actual = y_val[i]
    print(f"[{i}] Predicted: angle_x = {predicted[0]:.2f}, angle_y = {predicted[1]:.2f}  |  Actual: angle_x = {actual[0]:.2f}, angle_y = {actual[1]:.2f}")
