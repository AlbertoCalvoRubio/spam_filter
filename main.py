#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:04:50 2019

@author: Alberto Calvo Rubió
"""


import tp6_2_utils as utils
import load_mails as lm
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from termcolor import colored



######################################################
# Main
######################################################

# Cargar mails
mails, y, mails_test, y_test = lm.load_mails()

# Transformar en bolsas de palabras
X, X_test = utils.bow(mails, y, mails_test, y_test)

# Elegir mejor clasificador mediante kfold crossvalidation
classifiers = ['Multinomial', 'Bernoulli']
classifier_t, f1_score, accuracy, laplace = utils.b_classifier(classifiers, X, y)

print('***********************************')
print('Mejor clasificador: ', colored(classifier_t, 'green'))
print('f1_score: ', f1_score)
print('accuracy: ', accuracy)
print('laplace', laplace)
print('***********************************')

print()
print('--- Evaluacion del mejor clasificador ---')

# Evaluar mejor clasificador con los datos de test

if classifier_t == 'Multinomial':
    b_classifier = MultinomialNB(alpha=laplace)
else:
    b_classifier = BernoulliNB(alpha=laplace)
    
# Entrenar clasificador    
b_classifier.fit(X, y)

print('-- Utilizando predicciones de clases (Umbral=0.5) --')

# Predecir las clases con los datos de test
prediction = b_classifier.predict(X_test)
prediction_proba = b_classifier.predict_proba(X_test)[:,1] # Probabilidades de spam
prediction_custom = (b_classifier.predict_proba(X_test)[:,1] >= 1).astype(bool) # set threshold as 1

# Obtener metricas con prediccion de clases
f1_score = metrics.f1_score(y_test, prediction)
matriz = metrics.confusion_matrix(y_test, prediction)
matriz_normalizada = metrics.confusion_matrix(y_test, prediction, normalize='true')

print("f1_score: ", f1_score)
print("Confusion_matrix:\n", matriz)
print("Normalized confusion matrix:\n", matriz_normalizada)


# Obtener metricas con prediccion de probabilidades(umbral personalizado)
f1_score = metrics.f1_score(y_test, prediction_custom)
matriz = metrics.confusion_matrix(y_test, prediction_custom)
matriz_normalizada = metrics.confusion_matrix(y_test, prediction_custom, normalize='true')

print()
print('-- Utilizando predicciones de propabilidades (Umbral=1)--')
print("f1_score: ", f1_score)
print("Confusion_matrix:\n", matriz)
print("Normalized confusion matrix:\n", matriz_normalizada)


print('--- Dibujando graficas ---')

precision, recall, thresholds = metrics.precision_recall_curve(y_test, prediction_proba)

# Grafica que refleja precision y recall respecto a el umbral
utils.plot_precision_recall_threshold(thresholds, precision, recall)

# Por defecto utiliza predict_proba
metrics.plot_precision_recall_curve(b_classifier, X_test, y_test)

metrics.plot_confusion_matrix(b_classifier, X_test, y_test, normalize='true')


