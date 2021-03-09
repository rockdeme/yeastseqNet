import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from utils import encode_codons, encode_sequence, loss_plot, true_pred_plot
from network import threeconvnet


datasets = ['Weis', 'Perez', 'Cramer', 'Steinmetz', 'Gresham', 'Coller', 'Young', 'Brown', 'Hughes', 'Struhl',
            'Weighted half-life']
data = pd.read_csv('/home/turos0000/rna/utrs-and-half-lives.csv', sep=';', index_col=0)
features = pd.read_csv('/home/turos0000/rna/yeast-transcriptome-features_complete.csv', sep=';', index_col=0)
features = features.iloc[:, 11:]
data = pd.merge(data, features, left_index=True, right_index=True)

data['Len/3'] = data['Length'] / 3
data['5UTRlen'] = data['5UTR'].str.len()
data['3UTRlen'] = data['3UTR'].str.len()
data = data[data['Len/3'] == data['Len/3'] // 1]

y_column = 'Cramer'

data_dropped = data.dropna(subset=[y_column])
sequence_data = data_dropped[['5UTR', 'ORF', '3UTR']]
sequence_data = sequence_data.dropna()
region_lengths = []

for column in list(sequence_data):
    length = sequence_data[column].map(lambda x: len(x)).max()
    region_lengths.append(length)

utr5_array = np.zeros((len(sequence_data['5UTR']), 4, int(region_lengths[0]))).astype(np.int)
orf_array = np.zeros((len(sequence_data['ORF']), 64, int(region_lengths[1] / 3))).astype(np.int)
utr3_array = np.zeros((len(sequence_data['3UTR']), 4, int(region_lengths[2]))).astype(np.int)
i = 0
for index, sequence in sequence_data.iterrows():
    utr5_array[i, :, :int(len(sequence[0]))] = encode_sequence(sequence[0], enc_type='one-hot')
    orf_array[i, :, :int(len(sequence[1]) / 3)] = encode_codons(sequence[1], enc_type='one-hot')
    utr3_array[i, :, :int(len(sequence[2]))] = encode_sequence(sequence[2], enc_type='one-hot')
    i += 1
utr5_array = np.expand_dims(utr5_array, axis=3)
orf_array = np.expand_dims(orf_array, axis=3)
utr3_array = np.expand_dims(utr3_array, axis=3)
print('5UTR Array Shape: ' + str(utr5_array.shape))
print('ORF Array Shape: ' + str(orf_array.shape))
print('3UTR Array Shape: ' + str(utr3_array.shape))

selected_genes = data_dropped[data_dropped.index.isin(sequence_data.index)]
y = np.log(selected_genes[y_column] + 1 - np.min(selected_genes[y_column]))
sc = StandardScaler()
scaled_features = sc.fit_transform(selected_genes[['Sum tAi', 'gtAi', 'Sum CSC', 'gCSC', 'Sum tRNA abundance',
                                                   'gtRNA abundance', 'Residual']])

utr5_X_train, utr5_X_test, orf_X_train, orf_X_test, utr3_X_train, utr3_X_test, sumtai_X_train, sumtai_X_test, \
gtai_X_train, gtai_X_test, sumcsc_X_train, sumcsc_X_test, gcsc_X_train, gcsc_X_test, sumtab_X_train, sumtab_X_test, \
gtab_X_train, gtab_X_test, residual_X_train, residual_X_test, y_train, y_test = train_test_split(
    utr5_array,
    orf_array,
    utr3_array,
    scaled_features[:, 0],
    scaled_features[:, 1],
    scaled_features[:, 2],
    scaled_features[:, 3],
    scaled_features[:, 4],
    scaled_features[:, 5],
    scaled_features[:, 6],
    y,
    test_size=0.25,
    random_state=120)

model = threeconvnet(utr5_array.shape[2], orf_array.shape[2], utr3_array.shape[2], activation='lrelu',
                     initializer='he_uniform')
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit(x={'orf': orf_X_train,
                       'utr5': utr5_X_train,
                       'utr3': utr3_X_train,
                       'sumtai': sumtai_X_train,
                       'gtai': gtai_X_train,
                       'sumcsc': sumcsc_X_train,
                       'gcsc': gcsc_X_train,
                       'sumtab': sumtab_X_train,
                       'gtab': gtab_X_train,
                       'residual': residual_X_train,
                       },
                    y=y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(
                        {'orf': orf_X_test,
                         'utr5': utr5_X_test,
                         'utr3': utr3_X_test,
                         'sumtai': sumtai_X_test,
                         'gtai': gtai_X_test,
                         'sumcsc': sumcsc_X_test,
                         'gcsc': gcsc_X_test,
                         'sumtab': sumtab_X_test,
                         'gtab': gtab_X_test,
                         'residual': residual_X_test,
                         },
                        y_test),
                    verbose=2,
                    callbacks=[callback])

loss_plot(history)
r_sq = true_pred_plot({'orf': orf_X_train,
                       'utr5': utr5_X_train,
                       'utr3': utr3_X_train,
                       'sumtai': sumtai_X_train,
                       'gtai': gtai_X_train,
                       'sumcsc': sumcsc_X_train,
                       'gcsc': gcsc_X_train,
                       'sumtab': sumtab_X_train,
                       'gtab': gtab_X_train,
                       'residual': residual_X_train,
                       }, y_train,
                      {'orf': orf_X_test,
                       'utr5': utr5_X_test,
                       'utr3': utr3_X_test,
                       'sumtai': sumtai_X_test,
                       'gtai': gtai_X_test,
                       'sumcsc': sumcsc_X_test,
                       'gcsc': gcsc_X_test,
                       'sumtab': sumtab_X_test,
                       'gtab': gtab_X_test,
                       'residual': residual_X_test,
                       }, y_test, model, 0, 6,
                      alphaV=0.4)

print('R squared: ' + str(r_sq))
