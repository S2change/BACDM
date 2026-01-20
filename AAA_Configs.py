normalization_mean = (0.485, 0.456, 0.406, 0.485, 0.456, 0.406)
# (0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485)
normalization_std = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
# (0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229)

channel_nums = 6
selected_nums = [0, 1, 2, 3, 4, 5]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


seednumber = 42
LearningRate = 0.01
EPOCH = 100
batch_size = 16 # 28
num_workers = 16

Train_im_pathA = "G:/BACDM/data/before/" # 2019and2020_before
Train_im_pathB = "G:/BACDM/data/after/" # 2019and2020_after
Train_lb_path = "G:/BACDM/data/label/" # 2019and2020_label
Train_weight_path = "G:/BACDM/logs/" # 权重保存的路径

Train_pretrained_path = None


Test_im_pathA = r"C:\Users\domwe\Documents\thesis_work\BACDM\test_data\before"
Test_im_pathB = r"C:\Users\domwe\Documents\thesis_work\BACDM\test_data\after"
Test_det_path = r"C:\Users\domwe\Documents\thesis_work\BACDM\test_data\predictions"
Test_weight_path = r"C:\Users\domwe\Documents\thesis_work\logs\B12118A432.pth"
