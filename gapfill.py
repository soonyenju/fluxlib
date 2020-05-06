# coding: utf-8
import numpy as np
import pandas as pd
from pathlib import Path
from pysy.toolbox.utils import Yaml, create_all_parents, mount_drive
from sklearn.feature_selection import f_regression
import sys, pickle
from scipy.spatial.distance import cosine
from scipy import stats
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

class Candidate(object):
    def __init__(self, config_file):
        self.config = Yaml(config_file).load()

    @classmethod
    def __encode_igbp(self, igbp_list, normalize = False):
        igbp_dict = {}
        length = len(igbp_list)
        if length == 1: # avoid the divided by 0 error.
            length = 2
        for idx, igbp in enumerate(igbp_list):
            if normalize:
                igbp_dict[igbp] = idx /  (length - 1)
            else:
                igbp_dict[igbp] = idx
        return igbp_dict

    @staticmethod
    def __ensure_list(input):
        if not isinstance(input, list):
            return [input]
        else:
            return input

    def preprocess(self, normalize = True):
        '''
        get global min, max values of each variable
        get encoded igbps (normalized optional)
        get f_regression evaluations
        '''
        # list to log f_regression scores and p-values:
        self.f_scores_list = []
        self.p_values_list = []
        # ensure both x and y vars are list:
        x_vars = self.__ensure_list(self.config["x_vars"])
        y_vars = self.__ensure_list(self.config["y_vars"])

        # x_vars = self.config["x_vars"]
        # y_vars = self.config["y_vars"]
        # if not isinstance(x_vars, list):
        #     x_vars = [x_vars]
        # if not isinstance(y_vars, list):
        #     y_vars = [y_vars]
        
        # initialize a dict recording min and max of each var:
        min_max = {}
        for variable in x_vars + y_vars:
            min_max[variable] = [0, 0] # min, max
        paths = Path(self.config["workspace"].joinpath(self.config["data_folder"])).glob(r"*.csv")

        # iterate all sites' extrama while digitalizing their igpb:
        igbp_list = []
        for p in paths:
            igbp_list.append(p.stem.split("_")[1])
            df = pd.read_csv(p, index_col = self.config["time_tag"])
            if "Unnamed: 0" in df.columns:
                df.drop(["Unnamed: 0"], axis = 1, inplace = True)
            df = df.loc[:, x_vars + y_vars]
            if df.isin([-9999]).values.any():
                df = df.replace(-9999, np.nan)

            # mask non-nan values:
            Xs = df.loc[:, x_vars]
            ys = df.loc[:, y_vars]
            Xs_mask = pd.DataFrame(np.isfinite(Xs))
            ys_mask = pd.DataFrame(np.isfinite(ys))
            # iterate columns:
            mask = ys_mask.values
            if mask.ndim == 2:
                assert mask.shape[1] == 1
                mask = mask.ravel()
            for _, col in Xs_mask.items():
                mask = np.logical_and(mask, col)
            # sklearn f_regression: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
            f_scores, p_values = f_regression(Xs.loc[mask, :], ys.loc[mask, :].values.ravel())
            del(Xs, ys, Xs_mask, ys_mask, mask)
            self.f_scores_list.append(np.round(f_scores, 2))
            self.p_values_list.append(np.round(p_values, 2))
            for column_name in df.columns:
                df_temp = df[column_name]
                df_min = np.nanmin(df_temp)
                df_max = np.nanmax(df_temp)
                if df_min < min_max[column_name][0]:
                    min_max[column_name][0] = df_min

                if df_max > min_max[column_name][1]:
                    min_max[column_name][1] = df_max
        igbp_list = list(set(igbp_list))
        self.igbp_dict = self.__encode_igbp(igbp_list, normalize = normalize)
        self.min_max = min_max
        # print(self.igbp_dict, self.min_max, self.f_scores_list, self.p_values_list)

    def dispatch(self):
        seed_size = self.config["seed_size"]
        if self.config['save_file_suffix'] == None:
            self.config['save_file_suffix'] = ""
        if self.config["colab"]:
            root = mount_drive().joinpath(self.config["workspace"])
        else:
            root = Path(self.config["workspace"])
        paths = Path(root.joinpath(self.config["data_folder"])).glob(r"*.csv")
        paths = Path(self.config["data_folder"]).glob(r"*.csv")
        run_sites = self.config["run_sites"]

        if run_sites:
            run_sites = self.__ensure_list(run_sites)
        # if run_sites and (not isinstance(run_sites, list)):
        #     run_sites = [run_sites]
        # print(run_sites)

        for p in paths:
            site_name, igpb, _, _, _, _ = p.stem.split("_")
            if run_sites and (site_name not in run_sites):
                continue
            print(site_name)
            save_file = root.joinpath(self.config["output_folder"]).joinpath(f"{site_name}_{igpb}{self.config['save_file_suffix']}.pkl") # print(save_file) # print(save_file)
            create_all_parents(save_file, flag = "f")
            if save_file.exists():
                continue
            df = pd.read_csv(p, index_col = self.config["time_tag"])
            if "Unnamed: 0" in df.columns:
                df.drop("Unnamed: 0", axis = 1, inplace = True)

            df.index = pd.to_datetime(df.index, format = self.config["tag_format"])

            # randomly select seed_size from all data:
            seed_idxs = np.random.choice(df.shape[0], seed_size)
            seed_timestamps = df.index[seed_idxs]

            # print(df.head(3))
            # print(df.tail(3))

            save_temp_ts_file = root.joinpath(self.config["output_folder"]).joinpath(f"temp_ts_{site_name}{self.config['save_file_suffix']}.pkl")
            if not save_temp_ts_file.exists():
                read_ts = []
                with open(save_temp_ts_file, "wb") as f:
                    pickle.dump(read_ts, f, protocol = 4)
            save_temp_ds_file = root.joinpath(self.config["output_folder"]).joinpath(f"temp_ds_{site_name}{self.config['save_file_suffix']}.pkl")
            if not save_temp_ds_file.exists():
                Xs, ys = [], []
                temp_ds = {
                    "Xs": Xs,
                    "ys": ys
                }
                with open(save_temp_ds_file, "wb") as f:
                    pickle.dump(temp_ds, f, protocol = 4)

            for count, (target_idx, time_stamp) in enumerate(zip(seed_idxs, seed_timestamps)):
                try:
                    with open(save_temp_ds_file, "rb") as f:
                        temp_ds = pickle.load(f)
                        Xs = temp_ds["Xs"]
                        ys = temp_ds["ys"]
                    with open(save_temp_ts_file, "rb") as f:
                        read_ts = pickle.load(f)
                    if time_stamp in read_ts:
                        print(f"{time_stamp} exists...")
                        continue
                    print(time_stamp, count)
                    X, y = self.pipeline(df, target_idx)
                    # X shape should be 3, 3, if not continue:
                    try:
                        assert X.shape == (3, 3), f"X shape {X.shape}"
                    except Exception as e:
                        print(e)
                        continue
                    Xs.append(X)
                    ys.append(y)
                    temp_ds = {
                        "Xs": Xs,
                        "ys": ys
                    }
                    with open(save_temp_ds_file, "wb") as f:
                        pickle.dump(temp_ds, f, protocol = 4)
                    read_ts.append(time_stamp)
                    with open(save_temp_ts_file, "wb") as f:
                        pickle.dump(read_ts, f, protocol = 4)
                    # break
                except Exception as e:
                    print(e)
                    continue
            dataset = {
                "Xs": np.array(Xs)[:, np.newaxis, :, :],
                "ys": np.array(ys)
            }
            # print(dataset["Xs"].shape, dataset["ys"].shape)
            with open(save_file, "wb") as f:
                pickle.dump(dataset, f, protocol = 4)
            print(f"{save_file} is saved...")
            # break
        print("ok")

    def pipeline(self, df, target_idx):
        similarity_col_names = self.config["x_vars"]
        nee_col_name = self.config["y_vars"]
        nee_col_name = self.__ensure_list(nee_col_name)

        # degrade the similarity test if vector to be compared has zeros {
        target_similarity_vector = df[similarity_col_names].iloc[target_idx, :]
        similarity_col_names_no_zero = target_similarity_vector[target_similarity_vector != 0].index.tolist()
        # }

        # split df to target and "X" by similarity columns and nee: {
        target_similarity_vector = df[similarity_col_names_no_zero].iloc[target_idx, :] # print(target_similarity_vector.name)
        df_similarity = df[similarity_col_names_no_zero].drop([target_similarity_vector.name]) # print(df_similarity.head(10))

        target_nee = df[nee_col_name].iloc[target_idx, :] # print(target_nee)
        nee_column =  df[nee_col_name].drop([target_similarity_vector.name])

        # Get seasons
        # example:
        # >>> [(month%12 + 3)//3 for month in range(1, 13)]
        # [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
        seasons = (df_similarity.index.month%12 + 3) // 3 # print(seasons)
        # }

        nee_clean = self.measure_similarity(df_similarity, similarity_col_names_no_zero, target_similarity_vector, nee_column, seasons)
        nee_clean
        major_cluster_timestamp = self.clustering_sieve(nee_clean)
        df_candidates = self.vote_candidate(target_idx, df, nee_col_name, major_cluster_timestamp)
        # print(target_nee)
        # print(df_candidates)
        return df_candidates.values, target_nee.values

    def measure_similarity(self, df_similarity, similarity_col_names_no_zero, target_similarity_vector, nee_column, seasons):
        similarity_threshold = 1e-4
        max_iter = 5
        night_flag = False
        # determine if night time or day time
        hour = target_similarity_vector.name.hour
        if (hour >= 5) and (hour < 19):
            # daytime
            hour_tolerance = 6
        else:
            # nighttime
            hour_tolerance = 2
            night_flag = True

        the_month = (target_similarity_vector.name.month%12 + 3) // 3
        refined_idxs = np.where(
            (seasons == the_month).tolist() \
                and \
            (np.abs(target_similarity_vector.name - df_similarity.index).seconds/3600 < hour_tolerance).tolist()
        )[0]
        df_similarity_refined = df_similarity.iloc[refined_idxs, :] # print(df_similarity_refined)

        if df_similarity_refined.shape[0] > 1000:
            refined_idxs = np.where(
                np.abs(target_similarity_vector.name.year - df_similarity_refined.index.year) < 1
            )[0]
            df_similarity_refined = df_similarity_refined.iloc[refined_idxs, :] # print(df_similarity_refined)
        del(df_similarity)
        del(refined_idxs)
            
        # # time_difference_object: (time_stamp - target_similarity_vector.name)
        #         np.abs(
        #             (time_stamp - target_similarity_vector.name).days * 48 + \
        #                (time_stamp - target_similarity_vector.name).seconds/1800
        #         ), # unit: 30 mins
        container = [
            [
                cosine(target_similarity_vector.values, row.values),
                time_stamp,
                np.abs(target_similarity_vector.name - time_stamp).seconds/3600, # hours
                np.abs((time_stamp - target_similarity_vector.name).days), # days
                
            ] 
            for time_stamp, row in df_similarity_refined.iterrows()
        #     if np.abs(target_similarity_vector.name - time_stamp).seconds/3600 < hour_tolerance
        ]

        keys = [
            lambda x: (x[2], x[3], x[0]), # key order: hours, days, similarity
            lambda x: (x[3], x[2], x[0]) # key order: days, hours, similarity
        ]

        nee_containers = []
        for key in keys:
            sieve = container
            sieve.sort(key = key)

            sieve = np.array(sieve)
            for iter_ in range(max_iter):
                idx = np.where(sieve[:, 0] < similarity_threshold)[0]
                if idx.shape[0] > 100:
                    similarity_threshold /= 10
                else:
                    break
            try:
                assert idx.shape[0] >= 10, "Sieve shorter than 10..."
            except Exception as e:
                print(e)
                idx = np.where(sieve[:, 0] < similarity_threshold * 10)[0]

            timestamps = sieve[idx, 1][0: 10]
            nee_with_outliers = nee_column.loc[timestamps] # print(nee_with_outliers)
            del(sieve)
            del(idx)
            nee_containers.append(nee_with_outliers)

        nee_with_outliers = pd.concat(nee_containers).sort_index() # print(nee_with_outliers)
        del(container)
        del(nee_containers)
        # print(target_nee)

        # remove_outliers:
        if night_flag:
            threshold_z = 0.8
            repeat_time = 3
        else:
            threshold_z = 1
            repeat_time = 1
        max_iter = 2

        repeat_threshold = [
            target_similarity_vector.name - timedelta(hours = 1), 
            target_similarity_vector.name + timedelta(hours = 1)
        ]
        repeat_index = nee_with_outliers[
                (nee_with_outliers.index > repeat_threshold[0]) &
                (nee_with_outliers.index <= repeat_threshold[1])
        ].index
        nee_with_outliers = pd.concat([
            nee_with_outliers.loc[
                np.repeat(repeat_index, repeat_time)
            ],
            nee_with_outliers
        ])

        # remove outliers using IQR: {
        # calculate the interquartile range (IQR)
        # get the first and third quantiles:
        Q1 = np.quantile(nee_with_outliers, 0.25)
        Q3 = np.quantile(nee_with_outliers, 0.75)
        # get IQR
        IQR = Q3 - Q1 # print(IQR)
        idx_clean = np.where((nee_with_outliers >= (Q1 - 1.5 * IQR)) & (nee_with_outliers <= (Q3 + 1.5 * IQR)))
        nee_clean = nee_with_outliers.iloc[idx_clean[0]]
        # print(nee_clean) # print(idx_clean[0][-1], idx_clean[0].shape[0])
        # }

        # print(np.mean(nee_clean).values, np.std(nee_clean).values)
        # print(nee_clean)

        for iter_ in range(max_iter):
            # print(iter_)
            # remove outliers using z-score: {

            z = np.abs(stats.zscore(nee_clean)) # print(z)
            idx_clean = np.where(z < threshold_z)
            nee_clean_temp = nee_clean.iloc[idx_clean[0]]
            if nee_clean_temp.__len__() < 5:
                threshold_z += 1
            else:
                nee_clean = nee_clean_temp
                break
        # print(nee_clean)
        # assert not (nee_clean.values == 0.).all(), "All values are equal to zero..."
        # }
        return nee_clean

    # get the largest cluster
    def __get_majority(self, clustering):
        # get the cluster lable with takes the largest proportion: {
        bin_percent = np.bincount(clustering.labels_) / clustering.labels_.shape[0]
        # sort by percentage
        label_and_percent = list(zip(np.unique(clustering.labels_), bin_percent))
        label_and_percent.sort(key = lambda x: x[1], reverse = True)
        # print(label_and_percent)
        majority = label_and_percent[0]
        del(bin_percent, label_and_percent)
        # }
        return majority

    def clustering_sieve(self, nee_clean):
        from sklearn.cluster import KMeans, AffinityPropagation

        n_cluster = 5
        majority_cluster_thershold = 0.67
        max_iter = 5

        try: 
            assert nee_clean.shape[0] >= n_cluster, "Bad point..."
        except Exception as e:
            print(e)
            n_cluster = nee_clean.shape[0]

        # first clustering, AffinityPropagation doesn't need setting n_cluster, but it failed to work.
        # clustering = AffinityPropagation().fit(nee_clean.values.reshape(-1, 1))
        clustering = KMeans(n_clusters = n_cluster, random_state = 0).fit(nee_clean.values.reshape(-1, 1))
        # print(f"clustering lables: {clustering.labels_}, centers: {clustering.cluster_centers_}")
        majority = self.__get_majority(clustering)

        # if majority cluster is less than threshold, re-clustering via Kmeans by reducing n_cluster
        n_cluster = clustering.cluster_centers_.shape[0]
        if n_cluster < max_iter:
            max_iter = n_cluster
        iter_ = 1
        while (majority[1] < majority_cluster_thershold) and iter_ < max_iter:
            n_cluster -= 1
            
            clustering = KMeans(n_clusters = n_cluster, random_state=0)
            clustering.fit(nee_clean.values.reshape(-1, 1)) # print(f"clustering lables: {clustering.labels_}, centers: {clustering.cluster_centers_}")
            majority = self.__get_majority(clustering) # print(f"majority: {majority}, iter num: {iter_}")
            iter_ += 1

        # marjority structure: [label, proportion] 
        major_cluster_idx = np.where(clustering.labels_ == majority[0])
        major_cluster_timestamp = nee_clean.iloc[major_cluster_idx].index # print(major_cluster_timestamp)
        return major_cluster_timestamp

    def vote_candidate(self, target_idx, df, nee_col_name, major_cluster_timestamp):
        from sklearn.cluster import KMeans
        from sklearn.datasets import make_circles
        from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
        from sklearn.decomposition import TruncatedSVD
        from sklearn.naive_bayes import BernoulliNB

        n_clusters = 3
        # component number of truncated SVD, equals to cluster number
        n_components = n_clusters

        # columns for hash embedding
        # Xs are variables to learn, y is the nee
        hash_col_names = ["TA_ERA", "SW_IN_ERA", "LW_IN_ERA", "VPD_ERA", "PA_ERA", "P_ERA", "WS_ERA"]
        X_train = df.loc[major_cluster_timestamp, hash_col_names]
        y_train = df.loc[major_cluster_timestamp, nee_col_name]

        # use RandomTreesEmbedding to transform data
        hasher = RandomTreesEmbedding(n_estimators = 10, random_state = 0, max_depth = 3)
        X_transformed = hasher.fit_transform(X_train) # print(X_transformed.shape, X_train.shape, y_train.shape)
        svd = TruncatedSVD(n_components = n_components)
        X_reduced = svd.fit_transform(X_transformed) # print(X_reduced.shape)

        # clustering NEE in to n_cluster bins, and get mean and percentage of each bin: {

        clustering = KMeans(n_clusters = n_clusters, random_state = 0).fit(X_reduced, y_train)
        # unique_labels is uniqued labels, labels euquals to clustering.labels_, counts are count of each label in order.
        unique_labels, labels, counts = np.unique(clustering.labels_, return_inverse = True, return_counts = True)
        # percentage of each bin
        bin_percent = counts / np.sum(counts)

        # mean values of each bin
        # y_train is a 2d array, shape [:, 1] , bitcount accepts 1d array
        assert (y_train.values.ndim == 2) and (y_train.values.shape[1] == 1)
        bin_sum = np.bincount(labels, weights = y_train.values.ravel())
        bin_mean = bin_sum / counts 
        # print(pd.DataFrame({"mean": bin_mean, "per(%)": bin_percent}))

        # alternative to do this:
        # bin_percent = np.bincount(clustering.labels_) / clustering.labels_.shape[0]
        # df_temp = pd.DataFrame({'nee':y.values.ravel(), 'label':clustering.labels_})
        # df_bin_mean = df_temp.groupby('label').mean()
        # }


        X_target = df[hash_col_names].iloc[target_idx, :]
        assert (X_target.ndim == 1)
        target_X_transformed = hasher.transform(X_target.values.reshape(1, -1))
        target_X_reduced = svd.transform(target_X_transformed)

        # # Learn a Naive Bayes classifier on the transformed data
        nb = BernoulliNB()
        nb.fit(X_reduced, clustering.labels_)
        y_pred_prob = nb.predict_proba(target_X_reduced)
        # y_pred_prob is a column vector
        assert (y_pred_prob.ndim == 2) and (y_pred_prob.shape[0] == 1)
        return (pd.DataFrame({"mean": bin_mean, "per(%)": bin_percent, "prob(%)": y_pred_prob.ravel()}))

