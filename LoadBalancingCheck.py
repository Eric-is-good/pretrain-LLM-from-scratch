import json
import matplotlib.pyplot as plt
import numpy as np

# Set Chinese font. This is a common font that supports Chinese characters.
# If this font is not available in the environment, matplotlib will fall back to a default font.
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # To display the minus sign correctly

data = {
    "step": 1000,
    "moe_layer_2": [552, 1014, 476, 349, 517, 610, 959, 723, 885, 641, 1195, 670, 675, 739, 798, 518, 17, 640, 3, 338, 81, 697, 425, 587, 502, 77, 403, 273, 1084, 399, 0, 273, 237, 1203, 257, 476, 440, 5, 105, 151, 1015, 785, 708, 933, 251, 359, 96, 435],
    "moe_layer_3": [618, 1171, 757, 769, 617, 980, 761, 767, 627, 1189, 697, 808, 364, 743, 582, 764, 504, 47, 360, 592, 0, 275, 495, 394, 724, 574, 28, 472, 366, 173, 2, 346, 6, 1052, 19, 353, 29, 314, 3, 1369, 322, 391, 190, 781, 171, 686, 448, 876],
    "moe_layer_4": [717, 749, 540, 984, 629, 783, 329, 642, 662, 285, 578, 721, 660, 795, 414, 652, 40, 299, 781, 365, 0, 556, 739, 63, 54, 136, 0, 44, 735, 629, 396, 0, 345, 55, 1907, 637, 628, 8, 1049, 618, 1749, 398, 532, 465, 90, 328, 105, 685],
    "moe_layer_5": [767, 292, 617, 438, 841, 1072, 522, 783, 774, 1075, 827, 788, 683, 534, 777, 437, 290, 580, 1040, 485, 332, 36, 169, 1089, 476, 251, 20, 45, 55, 683, 32, 382, 717, 36, 21, 352, 767, 371, 1088, 239, 346, 1033, 343, 758, 117, 666, 0, 530],
    "moe_layer_6": [573, 734, 544, 590, 832, 793, 622, 1336, 746, 412, 490, 422, 1445, 753, 378, 916, 6, 5, 0, 58, 214, 840, 925, 1070, 566, 905, 576, 519, 253, 11, 643, 1089, 102, 226, 726, 71, 225, 99, 619, 538, 450, 90, 397, 329, 343, 4, 698, 393],
    "moe_layer_7": [602, 517, 907, 1012, 754, 570, 805, 556, 1327, 616, 398, 505, 529, 453, 805, 654, 265, 1040, 1164, 727, 2, 8, 42, 803, 1131, 398, 94, 818, 384, 0, 459, 144, 266, 627, 37, 390, 198, 731, 653, 305, 173, 694, 514, 105, 391, 507, 440, 56],
    "moe_layer_8": [480, 603, 637, 878, 542, 538, 801, 409, 338, 850, 684, 499, 970, 778, 495, 622, 61, 770, 366, 339, 1018, 321, 402, 1553, 395, 1682, 1, 364, 90, 579, 369, 484, 78, 296, 210, 410, 167, 392, 786, 25, 549, 1, 864, 4, 374, 534, 512, 456],
    "moe_layer_9": [637, 692, 469, 1003, 678, 1138, 627, 256, 351, 507, 309, 566, 457, 513, 787, 767, 530, 38, 525, 541, 145, 631, 99, 151, 549, 720, 840, 700, 519, 313, 680, 6, 373, 295, 502, 1062, 5, 82, 1054, 604, 178, 339, 370, 832, 15, 1203, 535, 383],
    "moe_layer_10": [1265, 785, 645, 968, 1113, 1004, 1001, 724, 1070, 991, 1021, 843, 602, 689, 1794, 689, 512, 43, 152, 101, 54, 107, 144, 321, 455, 268, 10, 352, 556, 464, 364, 149, 212, 422, 1, 219, 637, 733, 771, 305, 305, 592, 12, 57, 295, 207, 548, 4],
    "moe_layer_11": [952, 683, 1029, 661, 1042, 434, 932, 590, 762, 754, 453, 594, 752, 585, 760, 1062, 944, 33, 580, 540, 77, 19, 311, 64, 11, 1864, 683, 292, 191, 507, 4, 626, 76, 860, 55, 125, 670, 688, 516, 35, 185, 646, 164, 405, 246, 296, 11, 807]
}

layer_keys = sorted([key for key in data.keys() if key.startswith('moe_layer')])
num_layers = len(layer_keys)

# Create subplots, aiming for a 2-column layout
num_cols = 2
num_rows = (num_layers + num_cols - 1) // num_cols  # Calculate rows needed
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

for i, layer_key in enumerate(layer_keys):
    ax = axes[i]
    layer_data = data[layer_key]
    ax.hist(layer_data, bins=20, edgecolor='black')
    ax.set_title(layer_key)
    ax.set_xlabel("频数")
    ax.set_ylabel("出现次数")

# Hide any unused subplots
for i in range(num_layers, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("moe_layer_histograms.png")
print("Image saved to moe_layer_histograms.png")
