import numpy as np
def merge_label_score_by_max_score(full_label_scores, cropped_label_scores):
    merged_label_scores = full_label_scores
    for crop_label_score in cropped_label_scores:
        contains = False
        for i in range(len(merged_label_scores)):
            if merged_label_scores[i].label == crop_label_score.label:
                contains = True
                if crop_label_score.score > merged_label_scores[i].score:
                    merged_label_scores[i] = crop_label_score
                break
        if not contains:
            merged_label_scores.append(crop_label_score)
    merged_label_scores = sorted(merged_label_scores, key=sort_by_score, reverse=True)
    # print("merged_label_scores:",merged_label_scores)
    return merged_label_scores
def merge_label_score_by_proportion(full_label_scores, cropped_label_scores, full_score_proportion=0.5, crop_score_propertion=0.5):
    merged_label_scores = full_label_scores
    for crop_label_score in cropped_label_scores:
        contains = False
        for i in range(len(merged_label_scores)):
            if merged_label_scores[i].label == crop_label_score.label:
                contains = True
                merged_label_scores[i].score = full_score_proportion*merged_label_scores[i].score + crop_score_propertion*crop_label_score.score
                break
        if not contains:
            merged_label_scores.append(crop_label_score)
    merged_label_scores = sorted(merged_label_scores, key=sort_by_score, reverse=True)
    # print("merged_label_scores:",merged_label_scores)
    return merged_label_scores
def merge_label_score_by_max_score_with_first_two_and_same_label(full_label_scores, cropped_label_scores, default_first_two_score_full=0.9, default_first_two_score_crop=0.8):
    '''
    取预测标签的前两位必选
    :param full_label_scores:
    :param cropped_label_scores:
    :return:
    '''
    full_label_scores[0].score = max(default_first_two_score_full, full_label_scores[0].score)
    full_label_scores[1].score = max(default_first_two_score_full, full_label_scores[1].score) if full_label_scores[1].score > 0.12 else full_label_scores[1].score
    cropped_label_scores[0].score = max(default_first_two_score_crop, cropped_label_scores[0].score)
    cropped_label_scores[1].score = max(default_first_two_score_crop, cropped_label_scores[1].score) if cropped_label_scores[1].score > 0.5 else cropped_label_scores[1].score
    for i in range(min(len(full_label_scores), 2)):
        for j in range(min(len(cropped_label_scores), 2)):
            if full_label_scores[i].label == cropped_label_scores[j].label:
                full_label_scores[i].score = 1.0
                cropped_label_scores[j].score = 1.0
    # full_label_scores[3].score = full_label_scores[3].score*1.5
    # for i in range(min(len(full_label_scores), 5)):
    #     for j in range(min(len(cropped_label_scores), 5)):
    #         if full_label_scores[i].label == cropped_label_scores[j].label:
    #             full_label_scores[i].score = max(0.75, full_label_scores[i].score)
    #             cropped_label_scores[j].score = max(0.75, cropped_label_scores[j].score)
    return merge_label_score_by_max_score(full_label_scores, cropped_label_scores)
def merge_label_score_by_min_score(full_label_scores, cropped_label_scores, full_min_score=[1.0, 0.8], crop_min_score=[1.0, 0.8]):
    '''
    取预测标签的前两位必选
    :param full_label_scores:
    :param cropped_label_scores:
    :return:
    '''
    full_label_scores[0].score = max(full_min_score[0], full_label_scores[0].score)
    full_label_scores[1].score = max(full_min_score[1], full_label_scores[1].score)
    cropped_label_scores[0].score = max(crop_min_score[0], cropped_label_scores[0].score)
    cropped_label_scores[1].score = max(crop_min_score[1], cropped_label_scores[1].score)
    for i in range(min(len(full_label_scores), 5)):
        for j in range(min(len(cropped_label_scores), 5)):
            if full_label_scores[i].label == cropped_label_scores[j].label:
                full_label_scores[i].score = 1.0
                cropped_label_scores[j].score = 1.0
    return merge_label_score_by_max_score(full_label_scores, cropped_label_scores)
conflict_labels = [['37', '184'], #37-jet(喷气式飞机),184-boats(船)
                   ['37', '72'], #37-jet(喷气式飞机),72-water(水)
                   ['15', '122'],   #15-shops(商店)  122-market(市场)
                   ]
def remove_conflict_labels(label_scores):
    for group in conflict_labels:
        conflict_count = 0
        for conflict_label in group:
            for label_score in label_scores:
                if conflict_label == label_score.label:
                    conflict_count += 1
        if conflict_count >= 2:
            for i in range(len(group))[1:]:
                for label_score in label_scores:
                    if group[i] == label_score.label:
                        label_score.score = 0.0


# to_remove_labels = ['45','41','27','135','57','70','212','103','20','156']
to_remove_labels = ['45']
def remove_impossible_labels(full_label_scores, cropped_label_scores):
    '''
    去除[45-close-up(特写镜头),41-landscape(景观),27-hawaii(夏威夷),135-kauai(考艾岛),57-oahu(瓦胡岛),70-silhouette(轮廓),212-scotland(苏格兰),103-log(日志),20-kit(系列),156-detail(细节)]
    :param full_min_score:
    :param cropped_label_scores:
    :return:
    '''
    remove_indexs = []
    for i in range(len(full_label_scores)):
        if full_label_scores[i].label in to_remove_labels:
            remove_indexs.append(i)
    remove_indexs.reverse()
    for i in remove_indexs:
        full_label_scores.pop(i)
    remove_indexs.clear()
    for i in range(len(cropped_label_scores)):
        if cropped_label_scores[i].label in to_remove_labels:
            remove_indexs.append(i)
    remove_indexs.reverse()
    for i in remove_indexs:
        cropped_label_scores.pop(i)
    remove_indexs.clear()
def reset_score_proportion(label_scores, proportion):
    for label_score in label_scores:
        label_score.score = label_score.score*proportion
def sort_by_score(label_score):
    return label_score.score