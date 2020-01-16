from experiments.useful_tools import KFold_df, normalize, mre_calc, sa_calc
from sklearn.tree import DecisionTreeRegressor
from experiments.optimizers import *


def CART_DE(dataset, methods):

    dataset = normalize(dataset)
    mre_list = []
    sa_list = []
    rse_list = []

    for train, test in KFold_df(dataset, 3):

        train_input = train.iloc[:, :-1]
        train_actual_effort = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_actual_effort = test.iloc[:, -1]
        # max_depth: [1:12], min_samples_leaf: [1:12], min_samples_split: [2:21]

        def cart_builder(a, b, c):
            model = DecisionTreeRegressor(max_depth=a, min_samples_leaf=b, min_samples_split=c)
            model.fit(train_input, train_actual_effort)
            test_predict_effort = model.predict(test_input)
            test_predict_Y = test_predict_effort
            test_actual_Y = test_actual_effort.values
            # mre_list.append(mre_calc(test_predict_Y, test_actual_Y))
            # sa_list.append(sa_calc(test_predict_Y, test_actual_Y))
            if methods == 0:
                return mre_calc(test_predict_Y, test_actual_Y)  ############# MRE
            if methods == 1:
                return sa_calc(test_predict_Y, test_actual_Y)  ############# SA
            # return rse_calc(test_predict_Y, test_actual_Y)  ############# RSE

        metrics = methods

        output = de(cart_builder, metrics, bounds=[(1, 12), (0.00001, 0.5), (0.00001, 1)])
        if methods == 0:
            mre_list.append(output)  ############# MRE
        if methods == 1:
            sa_list.append(output)  ############# SA
        # rse_list.append(output)  ############# RSE

    if methods == 0:
        return mre_list  ############# MRE
    if methods == 1:
        return sa_list  ############# SA
    # return rse_list  ############# RSE


def CART_FLASH(dataset, methods):
    dataset = normalize(dataset)
    result_list = []

    for train, test in KFold_df(dataset, 3):
        train_input = train.iloc[:, :-1]
        train_actual_effort = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_actual_effort = test.iloc[:, -1]
        # max_depth: [1:12], min_samples_leaf: [1:12], min_samples_split: [2:21]

        metrics = methods
        output = flash(train_input, train_actual_effort, test_input, test_actual_effort, metrics, 10)
        result_list.append(output)

    return result_list



if __name__ == '__main__':

    import time
    from data_source.data_touse import *

    data = data_github_0()
    repeats = 20
    list_CART = []

    time1 = time.time()
    for i in range(repeats):
        list_CART.append(CART_FLASH(data))
    run_time1 = str(time.time() - time1)

    flat_list = np.array(list_CART).flatten()
    cart0_output = sorted(flat_list.tolist())

    print(cart0_output)
    print("median for CART0:", np.median(cart0_output))
    # print("mean for CART0:", np.mean(cart0_output))
    print("runtime for CART0:", run_time1)
    #
    # with open("./output/test_sk_mre.txt", "w") as output:
    #     output.write("CART0" + '\n')
    #     for i in sorted(cart0_output):
    #         output.write(str(i)+" ")
