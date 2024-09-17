# It seems the previous variable was not properly carried over, so I will redefine and re-execute the necessary part.

# Redefining all variables and plotting again
import matplotlib.pyplot as plt
import numpy as np
# Load data for each model
epochs = np.arange(1, 11)  # Assuming 10 epochs for simplicity

# Model 1: GRU
gru_train_loss = [1.3895, 1.3746, 1.3584, 1.3234, 1.2457, 1.1068, 1.0989, 1.0341, 1.0087, 1.0202]
gru_val_loss = [1.3757, 1.3614, 1.3323, 1.2844, 1.1822, 1.0140, 0.8490, 0.6729, 0.6219, 0.5440]
gru_train_acc = [0.2734, 0.3047, 0.3828, 0.6406, 0.7969, 0.7891, 0.9297, 0.8047, 0.9141, 0.8906]
gru_val_acc = [0.2828, 0.3335, 0.5346, 0.6018, 0.6123, 0.6772, 0.7074, 0.7989, 0.8043, 0.8236]

# Model 2: LSTM
lstm_train_loss = [1.3902, 1.3711, 1.3467, 1.3106, 1.2523, 1.0134, 0.7834, 0.4768, 0.2630, 0.1870]
lstm_val_loss = [1.3722, 1.3600, 1.3203, 1.2584, 1.1036, 0.9054, 0.6786, 0.5896, 0.5832, 0.5170]
lstm_train_acc = [0.2812, 0.3828, 0.4531, 0.6094, 0.6562, 0.7812, 0.8516, 0.9219, 0.9688, 0.9688]
lstm_val_acc = [0.3465, 0.3738, 0.5410, 0.6073, 0.6438, 0.7485, 0.7833, 0.8058, 0.8126, 0.8181]

# Model 3: CNN (from baseline)
cnn_train_loss = [1.3864, 1.3515, 1.3296, 1.2539, 1.2032, 1.0765, 0.9105, 0.7444, 0.6042, 0.4342]
cnn_val_loss = [1.3722, 1.3407, 1.2980, 1.2413, 1.1485, 1.0154, 0.9153, 0.8158, 0.7329, 0.6707]
cnn_train_acc = [0.2734, 0.4844, 0.6406, 0.7422, 0.7969, 0.8672, 0.9297, 0.9297, 0.9609, 0.9609]
cnn_val_acc = [0.3266, 0.4714, 0.5933, 0.6822, 0.6994, 0.7459, 0.7501, 0.7885, 0.7866, 0.8015]

# Model 4: GRU-LSTM
gru_lstm_train_loss = [1.3895, 1.3746, 1.3584, 1.3234, 1.2457, 1.1068, 1.0989, 1.0341, 1.0087, 1.0202]
gru_lstm_val_loss = [1.3757, 1.3614, 1.3323, 1.2844, 1.1822, 1.0140, 0.8490, 0.6729, 0.6219, 0.5440]
gru_lstm_train_acc = [0.2734, 0.3047, 0.3828, 0.6406, 0.7969, 0.7891, 0.9297, 0.8047, 0.9141, 0.8906]
gru_lstm_val_acc = [0.2828, 0.3335, 0.5346, 0.6018, 0.6123, 0.6772, 0.7074, 0.7989, 0.8043, 0.8236]

# Model 5: Attention
attn_train_loss = [1.3872, 1.3734, 1.3404, 1.3011, 1.2320, 1.1636, 1.0989, 1.0341, 1.0087, 1.0202]
attn_val_loss = [1.3758, 1.3584, 1.3328, 1.2881, 1.2567, 1.1964, 1.1565, 1.0341, 0.9739, 0.8906]
attn_train_acc = [0.1953, 0.4219, 0.6094, 0.6719, 0.8828, 0.9141, 0.9297, 0.8047, 0.9297, 0.8906]
attn_val_acc = [0.3865, 0.5045, 0.5590, 0.6757, 0.6994, 0.7172, 0.7186, 0.8047, 0.9141, 0.8906]

# Plotting all models in one plot
# It seems the previous variable was not properly carried over, so I will redefine and re-execute the necessary part.

# Redefining all variables and plotting again

# Load data for each model
epochs = np.arange(1, 11)  # Assuming 10 epochs for simplicity

# Model 1: GRU
# GRU Training Data
gru_train_loss = [1.3893, 1.3665, 1.3583, 1.3092, 1.2176, 1.0481, 0.7472, 0.4898, 0.3042, 0.1870]
gru_val_loss = [1.3705, 1.3453, 1.3128, 1.2526, 1.1060, 0.9127, 0.7079, 0.7059, 0.5281, 0.5660]
gru_train_acc = [0.2422, 0.4141, 0.3750, 0.5391, 0.7734, 0.7969, 0.9453, 0.8984, 0.9688, 0.9688]
gru_val_acc = [0.2946, 0.4616, 0.4914, 0.6108, 0.6994, 0.7829, 0.8078, 0.7456, 0.7953, 0.7992]


# Model 2: LSTM
lstm_train_loss = [1.3902, 1.3711, 1.3467, 1.3106, 1.2523, 1.0134, 0.7834, 0.4768, 0.2630, 0.1870]
lstm_val_loss = [1.3722, 1.3600, 1.3203, 1.2584, 1.1036, 0.9054, 0.6786, 0.5896, 0.5832, 0.5170]
lstm_train_acc = [0.2812, 0.3828, 0.4531, 0.6094, 0.6562, 0.7812, 0.8516, 0.9219, 0.9688, 0.9688]
lstm_val_acc = [0.3465, 0.3738, 0.5410, 0.6073, 0.6438, 0.7485, 0.7833, 0.8058, 0.8126, 0.8181]

# Model 3: CNN (from baseline)
cnn_train_loss = [1.3864, 1.3515, 1.3296, 1.2539, 1.2032, 1.0765, 0.9105, 0.7444, 0.6042, 0.4342]
cnn_val_loss = [1.3722, 1.3407, 1.2980, 1.2413, 1.1485, 1.0154, 0.9153, 0.8158, 0.7329, 0.6707]
cnn_train_acc = [0.2734, 0.4844, 0.6406, 0.7422, 0.7969, 0.8672, 0.9297, 0.9297, 0.9609, 0.9609]
cnn_val_acc = [0.3266, 0.4714, 0.5933, 0.6822, 0.6994, 0.7459, 0.7501, 0.7885, 0.7866, 0.8015]

# Model 4: GRU-LSTM
gru_lstm_train_loss = [1.3895, 1.3746, 1.3584, 1.3234, 1.2457, 1.1068, 1.0989, 1.0341, 1.0087, 1.0202]
gru_lstm_val_loss = [1.3757, 1.3614, 1.3323, 1.2844, 1.1822, 1.0140, 0.8490, 0.6729, 0.6219, 0.5440]
gru_lstm_train_acc = [0.2734, 0.3047, 0.3828, 0.6406, 0.7969, 0.7891, 0.9297, 0.8047, 0.9141, 0.8906]
gru_lstm_val_acc = [0.2828, 0.3335, 0.5346, 0.6018, 0.6123, 0.6772, 0.7074, 0.7989, 0.8043, 0.8236]

# Model 5: Attention
attn_train_loss = [1.3872, 1.3734, 1.3404, 1.3011, 1.2320, 1.1636, 1.0989, 1.0341, 1.0087, 1.0202]
attn_val_loss = [1.3758, 1.3584, 1.3328, 1.2881, 1.2567, 1.1964, 1.1565, 1.0341, 0.9739, 0.8906]
attn_train_acc = [0.1953, 0.4219, 0.6094, 0.6719, 0.8828, 0.9141, 0.9297, 0.8047, 0.9297, 0.8906]
attn_val_acc = [0.3865, 0.5045, 0.5590, 0.6757, 0.6994, 0.7172, 0.7186, 0.8047, 0.9141, 0.8906]

# Plotting all models in one plot
plt.figure(figsize=(12, 10))

# Loss Plot (Validation + Training)
plt.subplot(2, 2, 1)
plt.plot(epochs, gru_val_loss, label='Baseline-GRU Val', marker='o')
plt.plot(epochs, lstm_val_loss, label='Baseline-LSTM Val', marker='o')
plt.plot(epochs, cnn_val_loss, label='Baseline Val', marker='o')
plt.plot(epochs, gru_lstm_val_loss, label='Baseline-GRU-LSTM-Attention Val', marker='o')
plt.plot(epochs, attn_val_loss, label='Baseline-Attention Val', marker='o')
plt.title('Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Training Loss
plt.subplot(2, 2, 2)
plt.plot(epochs, gru_train_loss, label='Baseline-GRU Train', marker='o')
plt.plot(epochs, lstm_train_loss, label='Baseline-LSTM Train', marker='o')
plt.plot(epochs, cnn_train_loss, label='Baseline Train', marker='o')
plt.plot(epochs, gru_lstm_train_loss, label='Baseline-GRU-LSTM-Attention Train', marker='o')
plt.plot(epochs, attn_train_loss, label='Baseline-Attention Train', marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot (Validation)
plt.subplot(2, 2, 3)
plt.plot(epochs, gru_val_acc, label='Baseline-GRU Val', marker='o')
plt.plot(epochs, lstm_val_acc, label='Baseline-LSTM Val', marker='o')
plt.plot(epochs, cnn_val_acc, label='Baseline Val', marker='o')
plt.plot(epochs, gru_lstm_val_acc, label='Baseline-GRU-LSTM-Attention Val', marker='o')
plt.plot(epochs, attn_val_acc, label='Baseline-Attention Val', marker='o')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Training Accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, gru_train_acc, label='Baseline-GRU Train', marker='o')
plt.plot(epochs, lstm_train_acc, label='Baseline-LSTM Train', marker='o')
plt.plot(epochs, cnn_train_acc, label='Baseline Train', marker='o')
plt.plot(epochs, gru_lstm_train_acc, label='Baseline-GRU-LSTM-Attention Train', marker='o')
plt.plot(epochs, attn_train_acc, label='Baseline-Attention Train', marker='o')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

# Save the updated plot
file_path_all = "./full_model_comparison_plot_updated.png"
plt.savefig(file_path_all)
file_path_all
