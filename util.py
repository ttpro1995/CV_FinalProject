
#  (happiness, sadness, surprise, anger, disgust, fear)
happiness = 'HA'
sadness = 'SA'
surprise = 'SU'
anger = 'AN'
disgust = 'DI'
fear = 'FE'
neutral = 'NE'
expression = [neutral,happiness, sadness, surprise, anger, disgust, fear]

def jaffe_labeling(name):
    expression_str = name[3:5]
    return expression.index(expression_str)
