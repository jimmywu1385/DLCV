import os
root = os.path.join("dlcv-final-problem1-talking-to-me", "student_data", "student_data", "crop_train")
for i in os.listdir(root):
    if len(os.listdir(os.path.join(root, i))) == 0:
        os.rmdir(os.path.join(root, i))