import numpy as np
import time
from experiments.learners import *
from experiments.tuned_learners import *
from data_experiment.data_touse_new import *
from data_experiment.data_touse_old import *


def prediction_contemporary(Repo, Directory, metrics, repeats, tocsv=False):
    data = data_contemporary(Repo, Directory)
    print(Repo)
    for way in range(6):
        if way == 0:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(KNN(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for KNN:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/contemporary_metric{}.csv".format(metrics), "a+") as output:
                    output.write("%s, " % Repo)
                    output.write(str(np.median(list_output)) + ",")
        if way == 1:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(SVM(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for SVR:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/contemporary_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 2:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/contemporary_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 3:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(RF(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for RFT:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/contemporary_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 4:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART_DE(data, metrics))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART_DE:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/contemporary_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 5:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART_FLASH(data, metrics))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART_FLASH:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/contemporary_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + "\n")
    print("---------------------------------")


def prediction_cocomo(num, metrics, repeats, tocsv=False):
    data = data_cocomo(num)[0]
    print(data_cocomo(num)[1])
    for way in range(6):
        if way == 0:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(KNN(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for KNN:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/cocomo_metric{}.csv".format(metrics), "a+") as output:
                    output.write("%s, " % data_cocomo(num)[1])
                    output.write(str(np.median(list_output)) + ",")
        if way == 1:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(SVM(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for SVR:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/cocomo_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 2:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/cocomo_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 3:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(RF(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for RFT:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/cocomo_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 4:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART_DE(data, metrics))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART_DE:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/cocomo_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 5:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART_FLASH(data, metrics))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART_FLASH:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/cocomo_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + "\n")
    print("---------------------------------")


def prediction_classic(num, metrics, repeats, tocsv=False):
    data = data_classic(num)[0]
    print(data_classic(num)[1])
    for way in range(6):
        if way == 0:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(KNN(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for KNN:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/classic_metric{}.csv".format(metrics), "a+") as output:
                    output.write("%s, " % data_classic(num)[1])
                    output.write(str(np.median(list_output)) + ",")
        if way == 1:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(SVM(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for SVR:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/classic_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 2:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/classic_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 3:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(RF(data)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for RFT:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/classic_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 4:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART_DE(data, metrics))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART_DE:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/classic_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
        if way == 5:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART_FLASH(data, metrics))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            print("median for CART_FLASH:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/classic_metric{}.csv".format(metrics), "a+") as output:
                    output.write(str(np.median(list_output)) + "\n")
    print("---------------------------------")


if __name__ == '__main__':

    repeats = 1

    ## single repo test
    # path = r'../data_experiment/contemporary/'
    # repo = "goby_monthly.csv"
    # with open("../result_experiment/{}_metric{}.csv".format(dataType, metrics), "a+") as output:
    #     output.write("data, KNN, SVR, CART, RF, DE_CART, ROME" + "\n")
    # prediction(repo, path, metrics, repeats, tocsv, dataType)

    # ## cocomo
    # for metrics in [0, 1]:
    #     for num in [0, 1, 2]:
    #         prediction_cocomo(num, metrics, repeats, tocsv=True)
    #
    # ## classic
    # for metrics in [0, 1]:
    #     for num in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    #         prediction_classic(num, metrics, repeats, tocsv=True)

    ## contemporary
    contemporary_pool = []
    path = r'../data_experiment/contemporary/'
    for filename in os.listdir(path):
        if not filename.startswith('.'):
            contemporary_pool.append(os.path.join(filename))
    for metrics in [0, 1]:
        for repo in sorted(contemporary_pool):
            prediction_contemporary(repo, path, metrics, repeats, tocsv=True)
