# Import stuff
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# Data Generation
data_list = {'id':[1,2,3,4,5,6,7,8,9,10],
             'name':['Jon','Ada','Steve','Dan','Joe','Marry','Bill','Jim','Milan','Martin'],
             'surname':['Queen','Jobs','Weehaa','Ng','Gates','Buffet','Musk','Ng','Gates','Musk'],
             'age':[10,12,50,47,61,21,45,14,65,15],
             'state':['USA','USA','USA','GB','CZ','GE','USA','CA','USA','USA']
             }

df = pd.DataFrame(data_list)
o = df.dtypes.values[0]

# Encode data so that we can feed it into the NN
label_encoder_logger = {}
for col in ['name','surname','state']:
    label_encoder_logger[col] = MultiLabelBinarizer()
    df['{}_enc'.format(col)] = label_encoder_logger[col].fit_transform(df[col])

output_columns = ['name_enc', 'surname_enc', 'state_enc', 'age']
output_columns = ['name','surname','state','age']
input_column = 'id'

mlb = MultiLabelBinarizer()
target = mlb.fit_transform(df[output_columns])

# Fit the neural net
clf = MLPClassifier()
clf.fit(X=df[[input_column]],y=target)
