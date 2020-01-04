#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 19:17:14 2019

@author: Alberto Calvo Rubió
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from termcolor import colored


def bow(mails, y, mails_test, y_test):
    print("---------- Obtaining BOWs from emails ---------")
    vectorizer  = CountVectorizer(ngram_range=(1, 1))  # Initialize BOW structure
    X = vectorizer.fit_transform(mails)                # BOW with word counts
    X_test = vectorizer.transform(mails_test)          

    # Bolsa de palabras normalizada con frecuencia de aparicion 
    vectorizer_attempt = TfidfTransformer(smooth_idf=True) # Previene division 0
    X = vectorizer_attempt.fit_transform(X)
    

    return X, X_test


def kf_cv(classifier_t, X, y):
    
    b_f1_score = 0.0
    y_array = np.array(y)
    
    for laplace in [0.1, 0.5, 1, 1.5, 2, 5, 10]:
        
        f1_score_total = 0.0
        accuracy_total = 0.0
        min_folds = 2
        max_folds = 10
        num_folds = max_folds - min_folds + 1
        
        # Se hace para distintos tamaños de folds
        for fold in range (min_folds, max_folds + 1):
            kf = KFold(n_splits=fold)
            f1_score = 0.0
            accuracy = 0.0
            
            # Se varían los conuntos de entrenamiento y validacion
            for train_index, validation_index in kf.split(X):
                if classifier_t == 'Multinomial':
                    classifier = MultinomialNB(alpha=laplace)
                else:
                    classifier = BernoulliNB(alpha=laplace)
                
                # Datos de entrenamiento y validacion
                train_x = X[train_index]
                train_y = y_array[train_index]
                validation_x = X[validation_index]
                validation_y = y_array[validation_index]
                
                # Entrenar clasificador
                classifier.fit(train_x, train_y)
    
                # Predecir las clases con los de validacion
                prediction = classifier.predict(validation_x)
    
                # Evaluar con datos de validacion
                f1_score = f1_score + metrics.f1_score(validation_y, prediction)
                accuracy = accuracy + metrics.accuracy_score(validation_y, prediction)
            
            # Acumular medias de cada variacion de conjuntos
            f1_score_total = f1_score_total + f1_score/fold
            accuracy_total = accuracy_total + accuracy/fold

        # Media de los distintos tamaños de folds
        f1_score_mean = f1_score_total/num_folds
        accuracy_mean = accuracy_total/num_folds
            
        print('laplace', laplace)
        print('f1_score: ', f1_score_mean)
        print('accuracy: ', accuracy_mean)
        print()
    
        
        # Se elige mejor media f1_score
        if(f1_score_mean >= b_f1_score):
            b_f1_score = f1_score_mean
            b_accuracy = accuracy_mean
            b_laplace = laplace
            
    return b_f1_score, b_accuracy, b_laplace
            
        
def b_classifier(classifiers, X, y):
    b_f1_score = 0.0

    for classifier_t in classifiers:
        print('-- Probando clasificador: ', colored(classifier_t, 'green'), '---')
        f1_score, accuracy, laplace = kf_cv(classifier_t, X, y)
        # Clasificacion por f1_score
        if f1_score >= b_f1_score:
            b_classifier_t = classifier_t
            b_f1_score = f1_score
            b_accuracy = accuracy
            b_laplace = laplace
    
    return b_classifier_t, b_f1_score, b_accuracy, b_laplace

def plot_precision_recall_threshold(thresholds, precision, recall):
    plt.plot(thresholds, precision[:-1], label='precision')
    plt.plot(thresholds, recall[:-1], label='recall')
    plt.title('Precision-recall thresholds')
    plt.xlabel('Threshold')
    plt.legend(loc='lower left')
    plt.show()

    


        