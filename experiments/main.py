import numpy as np
import time
from experiments.learners import *
from experiments.tuned_learners import *
from data_source.data_touse import *
from testData.contemp_data_touse import *

data = data_github_new()
repeats = 1
methods = 1  # "0" for MRE, "1" for SA, "2" for RSE


if __name__ == '__main__':

    for i in range(6):

        if i == 0:
            list_temp = []
            time0 = time.time()
            for i in range(repeats):
                list_temp.append(CART_DE(data, methods))
            run_time0 = str(time.time() - time0)

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            # print(list_output)
            print("median for CART_DE:", np.median(list_output))
            # print("runtime for CART_DE:", run_time0)

            with open("../results/scott_knott_results.txt", "w") as output:
                output.write("CART_DE" + '\n')
                for i in sorted(list_output):
                    output.write(str(i) + " ")

        if i == 1:
            list_temp = []
            time1 = time.time()
            for i in range(repeats):
                list_temp.append(CART_FLASH(data, methods))
            run_time1 = str(time.time() - time1)

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            # print(list_output)
            print("median for ROME:", np.median(list_output))
            # print("runtime for ROME:", run_time1)

            with open("../results/scott_knott_results.txt", "a+") as output:
                output.write('\n\n' + "ROME" + '\n')
                for i in sorted(list_output):
                    output.write(str(i) + " ")

        if i == 2:
            list_temp = []
            time2 = time.time()
            for i in range(repeats):
                list_temp.append(CART(data)[methods])
            run_time2 = str(time.time() - time2)

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            # print(list_output)
            print("median for CART0:", np.median(list_output))
            # print("runtime for CART0:", run_time2)

            with open("../results/scott_knott_results.txt", "a+") as output:
                output.write('\n\n' + "CART0" + '\n')
                for i in sorted(list_output):
                    output.write(str(i) + " ")

        if i == 3:
            list_temp = []
            time3 = time.time()
            for i in range(repeats):
                list_temp.append(RF(data)[methods])
            run_time3 = str(time.time() - time3)

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            # print(list_output)
            print("median for RF:", np.median(list_output))
            # print("runtime for RF:", run_time3)

            with open("../results/scott_knott_results.txt", "a+") as output:
                output.write('\n\n' + "RF" + '\n')
                for i in sorted(list_output):
                    output.write(str(i) + " ")

        if i == 4:
            list_temp = []
            time4 = time.time()
            for i in range(repeats):
                list_temp.append(SVM(data)[methods])
            run_time4 = str(time.time() - time4)

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            # print(list_output)
            print("median for SVR:", np.median(list_output))
            # print("runtime for SVR:", run_time4)

            with open("../results/scott_knott_results.txt", "a+") as output:
                output.write('\n\n' + "SVR" + '\n')
                for i in sorted(list_output):
                    output.write(str(i) + " ")

        if i == 5:
            list_temp = []
            time5 = time.time()
            for i in range(repeats):
                list_temp.append(KNN(data)[methods])
            run_time5 = str(time.time() - time5)

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            # print(list_output)
            print("median for ABE0:", np.median(list_output))
            # print("runtime for ABE0:", run_time5)

            with open("../results/scott_knott_results.txt", "a+") as output:
                output.write('\n\n' + "ABE0" + '\n')
                for i in sorted(list_output):
                    output.write(str(i) + " ")
