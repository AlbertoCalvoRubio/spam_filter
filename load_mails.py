# Example of loading emails from Enron
# Juan D. Tardos
# 28-Nov-2018

######################################################
# Imports
######################################################


import glob
from sklearn.utils import shuffle


######################################################
# Functions for loading mails
######################################################

def read_folder(folder):
    mails = []
    #print(folder)
    file_list = glob.glob(folder)  # List mails in folder
    #print(file_list)
    num_files = len(file_list)
    for i in range(0, num_files):
        i_path = file_list[i]
        #print(i_path)
        i_file = open(i_path, 'rb')
        i_str = i_file.read()
        i_text = i_str.decode('utf-8', errors='ignore')  # Convert to Unicode
        mails.append(i_text)  # Append to the mail structure
        i_file.close()
    return mails


def load_enron_folders(datasets):
    path = '/home/alberto/Downloads/'
    ham = []
    spam = []
    for j in datasets:
        ham  = ham  + read_folder(path + 'enron' + str(j) + '/ham/*.txt')
        spam = spam + read_folder(path + 'enron' + str(j) + '/spam/*.txt')
    num_ham  = len(ham)
    num_spam = len(spam)
    print("mails:", num_ham+num_spam)
    print("ham  :", num_ham)
    print("spam :", num_spam)

    mails = ham + spam
    labels = [0]*num_ham + [1]*num_spam
    mails, labels = shuffle(mails, labels, random_state=0)
    return mails, labels

def load_mails():
    print("Loading files...")

    print("------Loading train and validation data--------")
    mails, y = load_enron_folders([1,2,3,4,5])

    print("--------------Loading Test data----------------")
    mails_test, y_test = load_enron_folders([6])
    
    return mails, y, mails_test, y_test 

