
"""
*** Important ***

Please un-comment the relevant questions you would like to test in the 'main' method at the end of the file, in order to run each and every question.
The results and explanations are attached as part of the code itself as you initially asked. (it might take a while for all the algorithms to be completed... ;-)
We wrote explanation on each and every method and question.

Good Luck!

"""

import os
import math
import pandas as pd
import numpy as np

from scipy.spatial.distance import squareform

# user id | item id | rating | timestamp

PROJECT_ROOT = os.path.dirname(os.getcwd())


class RecommenderSystem(object):
    def __init__(self, file_name='u1.base'):
        """
        This is the base object of the Recommender System.
        The main assumption is that for every similarity function,
        the user will need to create a new similarity_matrix to work with
        """
        self.df = self.build_df(file_name)
        self.item_to_vector_dict = {}
        self.item_flat_vector = {}
        self.vector_of_distances = []
        self.algo_type = 'cosine_similarity'

    def build_similarity_matrix(self, function='cosine_similarity'):
        """
        this function build the similarity_matrix and output it into csv file
        in order for next sections of the HW to use
        :return:
        """
        # this function create the vector for every item
        self.item_to_vector()
        # this function will flat the vectors
        self.flat_vectors()
        # now we build the matrix using the flat vectors
        self.vector_of_distances = self.calc_vector_of_distances(function=function)
        # turn the vector to similarity_matrix
        self.dist_list = [dist[1] for dist in sorted(self.vector_of_distances.items())]
        self.similarity_matrix = pd.DataFrame(squareform(self.dist_list), index=self.all_item, columns=self.all_item)
        # file the diagonal with ones
        np.fill_diagonal(self.similarity_matrix.values, 1)
        # export it to a csv file
        csv_name = 'similarity_matrix_by_{}.csv'.format(function)
        self.similarity_matrix.to_csv(csv_name)

    def item_to_vector(self):
        """
        this function create first dict =  self.item_to_vector_dict based on item_id as the key in the dict
        :return:
        """
        self.all_item = list(set(self.df['item_id']))
        self.all_users = list(set(self.df['user_id']))

        for index, (user_id, item_id, rating) in enumerate(zip(self.df.user_id, self.df.item_id, self.df.rating)):
            # we build a dictionary were the key is the item_id and the value is a tuple contain the user id and the rating
            self.item_to_vector_dict.setdefault(item_id, []).append((user_id, rating))

    def flat_vectors(self):
        """
        this function will form one list, per item id were the
        user id is the index (of the list) and the value in the rating
        :return:
        """
        for key, items in self.item_to_vector_dict.items():
            # set the flatten list is size of all users,
            # we do -1 to maintain the index of the list with item's index,
            # starting from 1 in the data set an start from 0 in the list
            temp_list = [0] * (len(self.all_users))
            for item in items:
                # take out the unnecessary item
                temp_list.pop(item[0] - 1)
                # set the right value in the index
                temp_list.insert(item[0] - 1, item[1])
            # set the flatten list as value and the item as key in the new dict
            self.item_flat_vector[str(key)] = temp_list

    def calc_vector_of_distances(self, function):
        """
        this function goes over every element in the item_flat_vector (his ratings list) and
        compare it to all other ratings lists in order to build the similarity matrix
        :param function: the type of the function to build the similarity with
        :return: a dict with all the results of the comparision
        """
        D_pair_value = dict()

        for item_1, ranting_list_1 in self.item_flat_vector.items():
            for item_2, ranting_list_2 in self.item_flat_vector.items():
                # going over half of the matrix based on similarity distance(a,b) == distance(b,a)
                if float(item_1) > float(item_2):
                    pair = (item_1, item_2)
                    value = self.create_similarity(function, ranting_list_1, ranting_list_2)
                    D_pair_value[pair] = value
                else:
                    pass

        return D_pair_value

    def build_df(self, file_name):
        """
        this function build the df based on a file name located in the same dir as this code
        :param file_name: the name of the file to build Dataframe from
        :return: df
        """
        data = pd.read_csv(file_name, sep="	", header=None)
        data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        return data

    def create_similarity(self, similarity_function, ranting_list_1, ranting_list_2):
        """
        this function direct to the right similarity_function based on the user input.
        :param similarity_function:
        :param ranting_list_1: data 1
        :param ranting_list_2: data 2
        :return:
        """
        if similarity_function == 'cosine_similarity':
            return_value = self.cosine_similarity(ranting_list_1, ranting_list_2)
        elif similarity_function == 'adjusted_cosine_similarity':
            return_value = self.adjusted_cosine_similarity(ranting_list_1, ranting_list_2)
        elif similarity_function == 'jaccard_coefficient':
            return_value = self.jaccard_coefficient(ranting_list_1, ranting_list_2)
        else:
            return_value = self.dice_coefficient(ranting_list_1, ranting_list_2)

        return return_value

    def prediction_setup(self):
        """
        this function makes the necessary setup for the prediction section
        :return:
        """
        self.file_name = ''
        try:
            for file in os.listdir():
                if file.endswith(self.algo_type + ".csv"):
                    self.file_name = file
                    break;
            print('found file {}'.format(self.file_name))

        except Exception as e:
            print("didn't find a similarity matrix to work with")
            raise e
        # if the file was found we build a data frame from it
        df = pd.read_csv(self.file_name)
        df_1 = df.rename(columns={'Unnamed: 0': 'Item'}, inplace=True)
        df_1 = df.reset_index()
        # setup parameters for the prediction
        self.prediction_df = df_1
        self.user_rating_df = self.df
        self.all_similarities_as_list = []
        self.all_items_list = self.prediction_df['Item']
        self.file_similarity_matrix = df_1

    def make_a_single_prediction(self, user_id, item_id, k_nearest_items):
        """
        This function is for the second section in the HW.
        this is the function for the user, were he set his parameters,
        then we run the prediction setup, makes the prediction and print it /return it to the user
        All the data here is based on the formula in HW.
        :param user_id: the user id to use for the prediction
        :param item_id: the item id to use for the prediction
        :param k_nearest_items: the k nearest items to use for the prediction
        :return: the prediction value
        """
        self.prediction_setup()
        self.make_prediction(user_id, item_id, k_nearest_items)
        # print(self.single_prediction)
        print("The prediction result using: {func} with k Neighbours number of {k} = {m}".format(func=self.algo_type,
                                                                                               k=k_nearest_items,
                                                                                               m=self.single_prediction))

        return self.single_prediction

    def make_prediction(self, user_id, item_id, k_nearest_items):
        """
        this function makes the prediction based on the similarity matrix form last section
        :param user_id: the user id to use for the prediction
        :param item_id: the item id to use for the prediction
        :param k_nearest_items: the k nearest items to use for the prediction
        :return: the prediction value
        """
        # find the item id column base on item_id input
        try:
            self.all_similarities_as_list = self.prediction_df[str(item_id)]
            self.get_nearest_items_and_similarities_as_lists(user_id, k_nearest_items)
            self.single_prediction = self.calc_prediction()
        except Exception as e:
            # throw exception if not founded
            print("didn't find the item id {} in the similarity matrix, return -1".format(item_id))
            self.single_prediction = -1
        return self.single_prediction

    def get_nearest_items_and_similarities_as_lists(self, user_id, k_nearest_items):
        """
        This function find k the nearest items based of similarity matrix.
        :param k_nearest_items:
        :return:
        """
        self.similarities_as_list = []
        self.items_list_to_rank = []
        self.rating_list_to_prediction = []
        # create pairs: (new item id, similarity to the users item id)
        pairs = zip(self.all_items_list, self.all_similarities_as_list)
        for item, sim in sorted(pairs, key=lambda x: x[1], reverse=True):
            if len(self.similarities_as_list) < k_nearest_items:
                # print("with user id {} for Item id: {}, the item {} similarity is {}".format(user_id, item_id, item, sim))
                self.similarities_as_list.append(sim)
                self.items_list_to_rank.append(item)
                rating = 0
                try:
                    user_condition = self.user_rating_df['user_id'] == user_id
                    item_condition = self.user_rating_df['item_id'] == item
                    temp_df = self.user_rating_df[user_condition & item_condition]
                    if not temp_df.empty:
                        rating = temp_df.rating.values[0]
                except Exception as e:
                    rating = 0
                self.rating_list_to_prediction.append(rating)
            else:
                break

    def test(self, file_name='u1.test', k_nearest_items_to_predict=10):
        """
        This function is for the third section in the HW.
        The user can pass the file name to test the number of k_nearest_items_to_predict.
        may predict different results for different values
        :param file_name: the file name of the data you would like to test on.
        make sure to keep the data in the same format:  user id | item id | rating | timestamp
        :param k_nearest_items_to_predict: how many items to include in the test
        :return: mean absolute error of the prediction against the test rating data
        """
        test_df = self.build_df(file_name)
        test_user_ids = test_df['user_id']
        test_item_ids = test_df['item_id']
        test_ratings = test_df['rating']

        self.prediction_setup()
        # make predictions
        list_of_predictions = []
        list_of_ratings_to_compare = []
        for index, (user_id_to_predict, item_id_to_predict) in enumerate(zip(test_user_ids, test_item_ids)):
            prediction = self.make_prediction(user_id_to_predict, item_id_to_predict, k_nearest_items_to_predict)
            # the the item id was not in the prediction we will not add it to the calculation
            if prediction > -1:
                list_of_predictions.append(prediction)
                list_of_ratings_to_compare.append(test_ratings[index])
        # operate a mean_absolute_error evaluation on the list_of_predictions compared to the list_of_ratings_to_compare.
        # not that only items ids that were included both in the test file and the similarity matrix will be included in the evaluation function
        mae = self.evaluation_mean_absolute_error(list_of_predictions, list_of_ratings_to_compare)
        print("The result MAE using: {func} with k Neighbours number of {k} = {m}".format(func=self.algo_type,
                                                                                        k=k_nearest_items_to_predict,
                                                                                        m=mae))
        return mae

    def evaluation_mean_absolute_error(self, list_of_prediction, list_of_true_ratings):
        """
        preform evaluation_mean_absolute_error based on prediction and test data
        :param list_of_prediction: prediction data
        :param list_of_true_ratings: test data
        :return:
        """
        list_of_abs = []
        number_of_elements = len(list_of_prediction)
        for prediction, true_ratings in zip(list_of_prediction, list_of_true_ratings):
            abs = math.fabs(prediction - true_ratings)
            list_of_abs.append(abs)
        res = sum(list_of_abs) / number_of_elements
        return res

    # helper functions
    def calc_prediction(self):
        numerator = self.calc_numerator_cosine_similarity(self.rating_list_to_prediction, self.similarities_as_list)
        denominator = sum(self.similarities_as_list)
        return numerator / denominator

    def cosine_similarity(self, ranting_list_1, ranting_list_2):
        numerator = self.calc_numerator_cosine_similarity(ranting_list_1, ranting_list_2)
        denominator = self.calc_denominator_cosine_similarity(ranting_list_1, ranting_list_2)
        rv = numerator / denominator
        return rv

    def adjusted_cosine_similarity(self, ranting_list_1, ranting_list_2):
        avg_list_1 = sum(ranting_list_1) / float(len(ranting_list_1))
        avg_list_2 = sum(ranting_list_2) / float(len(ranting_list_2))
        numerator = self.calc_numerator_cosine_similarity(ranting_list_1, ranting_list_2,
                                                          avg_list_1, avg_list_2)
        denominator = self.calc_denominator_cosine_similarity(ranting_list_1, ranting_list_2,
                                                              avg_list_1, avg_list_2)
        rv = numerator / denominator
        return rv

    def jaccard_coefficient(self, ranting_list_1, ranting_list_2):
        numerator = self.calc_numerator_jaccard_coefficient(ranting_list_1, ranting_list_2)
        denominator = self.calc_denominator_jaccard_coefficient(ranting_list_1, ranting_list_2)
        return numerator / denominator

    def dice_coefficient(self, ranting_list_1, ranting_list_2):
        numerator = 2 * self.calc_numerator_jaccard_coefficient(ranting_list_1, ranting_list_2)
        denominator = self.calc_denominator_dice_coefficient(ranting_list_1, ranting_list_2)
        return numerator / denominator

    def calc_numerator_cosine_similarity(self, list_1, list_2, avg_list_1=0, avg_list_2=0):
        """
        This function calculate numerator for cosin similarity types
        for using adjusted cosin similarity pass the average similarity parameter for each list
        otherwise the function behave as cosin similarity
        """
        rv = 0.0
        for index, (rating_1, rating_2) in enumerate(zip(list_1, list_2)):
            temp = 0
            if rating_1 > 0 and rating_2 > 0:
                temp = (rating_1 - avg_list_1) * (rating_2 - avg_list_2)
                rv += temp
        return rv

    def calc_denominator_cosine_similarity(self, list_1, list_2, avg_list_1=0, avg_list_2=0):
        """
        """
        sum_item_1 = 0.0
        sum_item_2 = 0.0
        # increment over all list_1
        for item_1 in list_1:
            if item_1 > 0:
                sum_item_1 += math.pow(item_1 - avg_list_1, 2)
        # increment over all list_2
        for item_2 in list_2:
            if item_2 > 0:
                sum_item_2 += math.pow(item_2 - avg_list_2, 2)
        # return sqrt of both of them
        rv = math.sqrt(sum_item_1 * sum_item_2)
        return rv

    def calc_numerator_jaccard_coefficient(self, list_1, list_2):
        conjunction = [x and y for x, y in zip(list_1, list_2)]
        conjunction_elements = sum(1 for i in conjunction if i > 0)
        return conjunction_elements

    def calc_denominator_jaccard_coefficient(self, list_1, list_2):
        conjunction = [x or y for x, y in zip(list_1, list_2)]
        conjunction_elements = sum(1 for i in conjunction if i > 0)

        return conjunction_elements

    def calc_denominator_dice_coefficient(self, list_1, list_2):
        return sum(1 for i in list_1 if i > 0) + sum(1 for i in list_2 if i > 0)


    def evaluation_with_param(self):
        """
        Question 5 - the function is generating recommendation similarity results
                     using different similarity measurements across range of neighbours
                     in order to analyse the different behaviors
        """

        simi_types = ['cosine_similarity', 'adjusted_cosine_similarity',
                     'jaccard_coefficient', 'dice_coefficient']

        for simi_type in simi_types:
            self.algo_type = simi_type
            self.build_similarity_matrix(simi_type)
            for neighbours in [10, 20, 50, 100]:
                self.test(k_nearest_items_to_predict=neighbours)

    def calculate_rate_skipping_prediction(self, user_id, h, k):
        """
        Question 6 - the function calculate the rate for each item of user_id
                     that does not appear in the base file using the method described in the article
        params :
        h - h number of items which the user rated highest
        k - k number of items that have the highest similarity with unrated item
        """

        if not hasattr(self, 'user_rating_df'):
            self.prediction_setup()

        user_unranked_items = self.get_unrank_itmes(user_id)
        h_user_items_max_rank = list(self.get_user_max_rank_items(h, user_id))
        rank_items = pd.DataFrame(columns=('item_id', 'rating'))

        for item in user_unranked_items:
            simi_rank = 0
            k_highest_item_similarity = self.get_items_with_highest_similarity(item, k)
            for i, k_h_item in k_highest_item_similarity.iteritems():
                if i in h_user_items_max_rank:
                    simi_rank += float(k_h_item)
            if simi_rank > 0:
                rank_items.loc[len(rank_items)] = ([str(item), simi_rank])

        return rank_items

    def get_user_max_rank_items(self, h, user_id):
        user_ranks = self.user_rating_df[self.user_rating_df.user_id == user_id]
        user_item_ranks = user_ranks[['item_id', 'rating']]
        return user_item_ranks.nlargest(h, 'rating')['item_id']

    def get_unrank_itmes(self, user_id):
        user_ranks = self.user_rating_df[self.user_rating_df.user_id == user_id]
        mask = ~self.all_items_list.isin(user_ranks.item_id)
        unrank_items = self.all_items_list[mask]
        return unrank_items

    def get_items_with_highest_similarity(self, item, k):
        item_similarity = self.file_similarity_matrix[str(item)]
        top_k_highest_similarity = item_similarity.nlargest(k)
        return top_k_highest_similarity

    def evaluate_using_AUC(self, h=5, k=5, file_name='u1.test'):
        from sklearn.metrics import roc_auc_score
        if not hasattr(self, 'user_rating_df'):
            self.prediction_setup()

        test_df = self.build_df(file_name)
        total_score = 0
        successful_prdeictions = 0
        for user_id in self.user_rating_df['user_id'].unique():
            prediction = self.calculate_rate_skipping_prediction(user_id, h, k)
            prediction['base_line'] = 0
            user_test_df = test_df[test_df.user_id == user_id][['item_id', 'rating']]
            prediction['item_id'] = prediction['item_id'].astype('int64')
            mask = prediction.item_id.isin(user_test_df.item_id)
            prediction.loc[mask, 'base_line'] = 1

            # print appropriate msg if there no relevant predictions for user id
            if 0 not in prediction.base_line.values or 1 not in prediction.base_line.values:
                print("No relevant prediction for user id #{user}".format(user=user_id))
            elif len(prediction.index) > 0 and 1 in prediction.base_line.values and 0 in prediction.base_line.values:
                base_line = np.array(prediction.base_line)
                scores = np.array(prediction.rating)
                auc = roc_auc_score(base_line, scores)
                total_score += auc
                successful_prdeictions += 1
                print("The result AUC result for user id {user_id} using: {func} with k as {k} and h as {h} is = {m}"
                      .format(user_id=user_id, func=self.algo_type, k=k, h=h, m=auc))

        print("The average AUC score is:{score}".format(score=total_score/successful_prdeictions))


if __name__ == "__main__":
    rs = RecommenderSystem()

    # Question #1: un-comment the below line
    #rs.build_similarity_matrix()

    # Question #2: un-comment the below line
    #rs.make_a_single_prediction(1, 1, 10)

    # Results for #2#
    #
    # The prediction result using: cosine_similarity with k Neighbours number of 10 = 2.0743002597771647



    # Question #3: un-comment the below line
    #rs.test(k_nearest_items_to_predict=10)



    # Questions #4+5: un-comment the below line
    # rs.evaluation_with_param()

    # Results for #5#
    #
    # found file similarity_matrix_by_cosine_similarity.csv
    # The result MAE using: cosine_similarity with k Neighbours number of 10 = 3.307249997545796
    # The result MAE using: cosine_similarity with k Neighbours number of 20 = 3.2883908542826545
    # The result MAE using: cosine_similarity with k Neighbours number of 50 = 3.274750867950121
    # The result MAE using: cosine_similarity with k Neighbours number of 100 = 3.2702755400752737
    #
    # found file similarity_matrix_by_adjusted_cosine_similarity.csv
    # The result MAE using: adjusted_cosine_similarity with k Neighbours number of 10 = 3.307964803589016
    # The result MAE using: adjusted_cosine_similarity with k Neighbours number of 20 = 3.2896094900544175
    # The result MAE using: adjusted_cosine_similarity with k Neighbours number of 50 = 3.2763106376586797
    # The result MAE using: adjusted_cosine_similarity with k Neighbours number of 100 = 3.2709438603506467
    #
    # found file similarity_matrix_by_jaccard_coefficient.csv
    # The result MAE using: jaccard_coefficient with k Neighbours number of 10 = 3.311927356928509
    # The result MAE using: jaccard_coefficient with k Neighbours number of 20 = 3.2945631864945883
    # The result MAE using: jaccard_coefficient with k Neighbours number of 50 = 3.27962518490281
    # The result MAE using: jaccard_coefficient with k Neighbours number of 100 = 3.2720862415397725
    #
    # found file similarity_matrix_by_dice_coefficient.csv
    # The result MAE using: dice_coefficient with k Neighbours number of 10 = 3.2892453721384323
    # The result MAE using: dice_coefficient with k Neighbours number of 20 = 3.2778968864264395
    # The result MAE using: dice_coefficient with k Neighbours number of 50 = 3.2688994495525416
    # The result MAE using: dice_coefficient with k Neighbours number of 100 = 3.264162246431683

    # We can clearly see that the mean absolute error (MAE), is getting better and better for each K used in each algorithm.
    # In addition, we can see that between the algorithms themselves, the last one (dice_coefficient) introduce the best result with K=100.


    # Question # 6: un-comment the below line
    #print(rs.calculate_rate_skipping_prediction(1, 5, 5))

    # Results for #6#
    #
    #     item_id    rating
    # 0      10  1.000000
    # 1      14  1.000000
    # 2      17  1.000000
    # 3      86  0.317830
    # 4     235  0.327710
    # 5     423  0.392823



    # Question # 7: un-comment the below line
    #rs.evaluate_using_AUC()

    # Results for #7.a#
    #
    # The result AUC result for user id 1 using: cosine_similarity with k as 5 and h as 5 is = 0.6
    # The result AUC result for user id 2 using: cosine_similarity with k as 5 and h as 5 is = 0.018518518518518545
    # The result AUC result for user id 3 using: cosine_similarity with k as 5 and h as 5 is = 0.8541666666666666
    # No relevant prediction for user id #4
    # The result AUC result for user id 5 using: cosine_similarity with k as 5 and h as 5 is = 0.75
    # The result AUC result for user id 6 using: cosine_similarity with k as 5 and h as 5 is = 0.8333333333333334
    # The result AUC result for user id 7 using: cosine_similarity with k as 5 and h as 5 is = 0.47619047619047616
    # The result AUC result for user id 8 using: cosine_similarity with k as 5 and h as 5 is = 0.8
    # The result AUC result for user id 9 using: cosine_similarity with k as 5 and h as 5 is = 0.7407407407407407
    # The result AUC result for user id 10 using: cosine_similarity with k as 5 and h as 5 is = 0.8333333333333334
    # The result AUC result for user id 11 using: cosine_similarity with k as 5 and h as 5 is = 0.09090909090909094
    # No relevant prediction for user id #12
    # The result AUC result for user id 13 using: cosine_similarity with k as 5 and h as 5 is = 1.0
    # The result AUC result for user id 14 using: cosine_similarity with k as 5 and h as 5 is = 0.8461538461538461
    # The result AUC result for user id 15 using: cosine_similarity with k as 5 and h as 5 is = 0.6428571428571428
    # No relevant prediction for user id #16
    # The result AUC result for user id 17 using: cosine_similarity with k as 5 and h as 5 is = 0.5833333333333334
    # The result AUC result for user id 18 using: cosine_similarity with k as 5 and h as 5 is = 0.2954545454545454
    # No relevant prediction for user id #19
    # No relevant prediction for user id #20
    # The result AUC result for user id 21 using: cosine_similarity with k as 5 and h as 5 is = 0.6086956521739131
    # The result AUC result for user id 22 using: cosine_similarity with k as 5 and h as 5 is = 0.0
    # The result AUC result for user id 23 using: cosine_similarity with k as 5 and h as 5 is = 0.75
    # The result AUC result for user id 24 using: cosine_similarity with k as 5 and h as 5 is = 0.8333333333333334
    # The result AUC result for user id 25 using: cosine_similarity with k as 5 and h as 5 is = 0.0
    # The result AUC result for user id 26 using: cosine_similarity with k as 5 and h as 5 is = 0.5
    # No relevant prediction for user id #27
    # No relevant prediction for user id #28
    # The result AUC result for user id 29 using: cosine_similarity with k as 5 and h as 5 is = 0.8777777777777778
    # No relevant prediction for user id #30
    # No relevant prediction for user id #31
    # The result AUC result for user id 32 using: cosine_similarity with k as 5 and h as 5 is = 0.0
    # No relevant prediction for user id #33
    # ...
    # ...
    # ...
    # No relevant prediction for user id #938
    # No relevant prediction for user id #939
    # No relevant prediction for user id #940
    # No relevant prediction for user id #941
    # No relevant prediction for user id #942
    # No relevant prediction for user id #943

    # Results for #7.b#
    #
    # The average AUC score is:0.541313666547423






