class ImageLabel(object):
    def __init__(self, image, label_score_strings=None, type=None, label_scores=None):
        '''
        :param image:
        :param label_score_strings:
        :param type: full-全图预测，crop-切图预测
        '''
        self.image = image
        self.type = type
        label_score_array = []
        if label_score_strings != None:
            # print("label_scores:", label_score_strings)
            for label_score in label_score_strings:
                score = float(label_score[label_score.find("(")+1:label_score.find(")")])
                label_score_array.append(LabelScore(label_score[:label_score.find("(")], score))
            self.label_scores = label_score_array
        if label_scores != None:
            self.label_scores = label_scores
class FinalPredictLabels(object):
    def __init__(self, image, labels):
        self.image = image
        self.labels = labels
class LabelScore(object):
    def __init__(self, label, score):
        self.label = label
        self.score = score
    def __repr__(self):
        return self.label+"("+str(self.score)+")"
    def __str__(self):
        return self.label+"("+str(self.score)+")"
def get_predict_labels(predict_file):
    '''
    :param predict_file:
    :return:
    '''
    predict_labels = []
    with open(predict_file) as f:
        while True:
            line = f.readline().strip()
            if line == "": break
            line_array = line.split(" ")
            predict_labels.append(ImageLabel(line_array[0], line_array[1].split(",")))
    return predict_labels
def get_final_predict_labels(predict_file):
    '''
    :param predict_file:
    :return:
    '''
    predict_labels = []
    with open(predict_file) as f:
        while True:
            line = f.readline().strip()
            if line == "": break
            line_array = line.split(" ")
            predict_labels.append(FinalPredictLabels(line_array[0], line_array[1].split(",")))
    return predict_labels
