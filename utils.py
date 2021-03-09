import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def encode_sequence(sequence, enc_type='one-hot'):
    sequence_list = np.array(list(sequence.upper())).reshape(-1, 1)
    X = [['A'], ['C'], ['T'], ['G']]
    if enc_type == 'one-hot':
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoder.fit(X)
        encoded_sequence_csr = one_hot_encoder.transform(sequence_list)
        encoded_sequence = np.transpose(encoded_sequence_csr.toarray().astype(np.int))
    if enc_type == 'label':
        label_encoder = LabelEncoder()
        label_encoder.fit(X)
        encoded_sequence = label_encoder.transform(sequence_list.ravel())
    return encoded_sequence


def encode_codons(sequence, enc_type='one-hot'):
    sequence_list = np.array([sequence[i:i + 3] for i in range(0, len(sequence), 3)]).reshape(-1, 1)
    if enc_type == 'one-hot':
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        X = [['ATA'], ['ATC'], ['ATT'], ['ATG'], ['ACA'], ['ACC'], ['ACG'], ['ACT'],
             ['AAC'], ['AAT'], ['AAA'], ['AAG'], ['AGC'], ['AGT'], ['AGA'], ['AGG'],
             ['CTA'], ['CTC'], ['CTG'], ['CTT'], ['CCA'], ['CCC'], ['CCG'], ['CCT'],
             ['CAC'], ['CAT'], ['CAA'], ['CAG'], ['CGA'], ['CGC'], ['CGG'], ['CGT'],
             ['GTA'], ['GTC'], ['GTG'], ['GTT'], ['GCA'], ['GCC'], ['GCG'], ['GCT'],
             ['GAC'], ['GAT'], ['GAA'], ['GAG'], ['GGA'], ['GGC'], ['GGG'], ['GGT'],
             ['TCA'], ['TCC'], ['TCG'], ['TCT'], ['TTC'], ['TTT'], ['TTA'], ['TTG'],
             ['TAC'], ['TAT'], ['TAA'], ['TAG'], ['TGC'], ['TGT'], ['TGA'], ['TGG']
             ]
        one_hot_encoder.fit(X)
        encoded_sequence_csr = one_hot_encoder.transform(sequence_list)
        encoded_sequence = np.transpose(encoded_sequence_csr.toarray().astype(np.int))
    if enc_type == 'label':
        label_encoder = LabelEncoder()
        encoded_sequence = label_encoder.fit_transform(sequence_list.ravel())
    return encoded_sequence


def loss_plot(history):
    plt.clf()
    # plt.rcParams["figure.dpi"] = 150
    plt.plot(history.history['loss'], color='grey')
    plt.plot(history.history['val_loss'], color='indigo')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def true_pred_plot(X_train, y_train, X_test, y_test, model, minVal, maxVal, alphaV=1.0):
    plt.clf()
    plt.axes(aspect='equal')
    lims = [minVal, maxVal]
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    # plt.rcParams["figure.dpi"] = 150
    testplot = plt.scatter(y_test, test_predictions, color='indigo', alpha=alphaV, s=10, zorder=2)
    trainplot = plt.scatter(y_train, train_predictions, color='grey', alpha=alphaV, s=10, zorder=1)
    plt.plot(lims, lims, color='black', zorder=0)
    m, b = np.polyfit(y_test, test_predictions[:, 0], 1)
    plt.plot(y_test, m * y_test + b, '-', lw=2, color='black', alpha=0.7)
    r2 = r2_score(y_test, test_predictions[:, 0])
    r2_string = "%.2f" % r2
    plt.xlabel('True Values [half-life (min)]')
    plt.ylabel('Predictions [half-life (min)]')
    plt.title('$R^2$: ' + str(r2_string))
    plt.xlim(lims)
    plt.ylim(lims)
    plt.legend((testplot, trainplot), ('Test set', 'Training set'), loc='upper left')
    plt.show()
    r_squared = r2_score(y_test, test_predictions[:, 0])
    return r_squared