from file_process import class_image_label_score
FULL_PREDICT_FILE = ""
CROP_PREDICT_FILE = ""

full_predict_info = class_image_label_score.get_final_predict_labels(FULL_PREDICT_FILE)
crop_predict_info = class_image_label_score.get_final_predict_labels(CROP_PREDICT_FILE)
def merge_predict_info(full_predict_info, crop_predict_info):
    for full_image_labels in full_predict_info:
        for crop_image_labels in crop_predict_info:
            if full_image_labels.image == crop_image_labels.image:
                merge_labels()
def merge_labels(full_image_labels, crop_image_labels):
    merged_labels = []
    merged_labels.append(full_image_labels[0])
    merged_labels.append(crop_image_labels[0])
