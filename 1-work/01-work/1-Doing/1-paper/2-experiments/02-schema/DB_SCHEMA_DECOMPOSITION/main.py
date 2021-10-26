import sys
import time
import pdb
import pandas as pd
import scipy.stats as sc
import numpy as np
import math
from itertools import combinations
def read_db(path):
    pdb.set_trace()
    data = pd.read_csv(path)
    return data

class schema_decomposition:
    def __init__(self,data):
        xxx
        self.data=data
        #pd.dataframe
        self.T=data.shape[0]


    def prune(self):
        #When to stop decompostion and storage
        return 0
    def prefix_blocks(self, L):
        '''
        Procedure PREFIX_BLOCKS described in [1]
        '''
        blocks = {}
        for atts in L:
            blocks.setdefault(atts[:-1], []).append(atts)
        return blocks.values()

    def generate_next_level(self, L):
        '''
        Procedure GENERATE_NEXT_LEVEL described in [1]
        '''
        #pdb.set_trace()
        self.new_level()
        next_L = set([])
        for k in self.prefix_blocks(L):
            for i, j in combinations(k, 2):
                if i[-1] < j[-1]:
                    X = i + (j[-1],)
                else:
                    X = j + (i[-1],)
                if all(X[:a]+X[a+1:] in L for a, x in enumerate(X)):
                    next_L.add(X)
                    # WE ADD THIS LINE, SEEMS A BETTER ALTERNATIVE TO CALCULATE THE PARTITION HERE WHEN
                    # WE HAVE REFERENCES TO BOTH PARTITIONS USED TO CALCULATE IT
                    self.pmgr.register_partition(X, i, j)
        #pdb.set_trace()
        return next_L

    def new_level(self):
        """
        Create a cache for the new level
        :return:
        """
        self.current_level+=1
        #data的存储要与不要

    def purge_old_level(self):
        """
        memory wipe of unused cache
        :return:
        """

    def is_primarykey(self):
        """
        which is primary key?
        :return:
        """

    def remove_duplicate_row(self,X):
        """
        可以用slice取 Y_data_column = data[:, target_variable_index]
        X是需要去掉重复行的属性名
        """
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            data_numpy = self.data.to_numpy()
        else:
            data_numpy=self.data
        data_X=data_numpy[:,X]
        data_X=np.unique(data_X,axis=0)
        return data_X

    def size_and_counts_of_attribute(self,X):
        """
        Returns the size, and the value counts of X
        """
        pdb.set_trace()
        X = self.merge_columns(X)
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            counts = X.value_counts()
            length = len(X.index)
        elif isinstance(X, np.ndarray):
            counts = np.unique(X, return_counts=True, axis=0)[1]
            length = np.size(X, 0)

        return length, counts
    def number_of_columns(self,X):
        """ Returns the number of columns of X, taking into account different shapes"""
        pdb.set_trace()
        if isinstance(X, pd.DataFrame):
            return X.shape[1]
        elif isinstance(X, np.ndarray):
            num_dim = X.ndim
            if num_dim == 2:
                return np.size(X, 1)
            elif num_dim == 1:
                return 1
        elif isinstance(X, pd.Series):
            return 1
    def to_numpy_if_not(self,X):
        """ Returns the numpy representation if dataframe"""
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return X
    def append_two_arrays(X, Z):
        """
        Appends X and Z horizontally
        stack属性
        """
        pdb.set_trace()
        if Z is None:
            return X

        if X is None:
            return Z

        if X is None and Z is None:
            raise ValueError('Both arrays cannot be None')

        return np.column_stack((X, Z))


    #将多个属性混合在一起。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
    def merge_columns(self, X):
        """
        Combines multiple columns into one with resulting domain the distinct JOINT values of the input columns
        entropy统计每一列的数据，计算entropy用的，对列数据的统计
        """
        if isinstance(X, pd.DataFrame):
            num_columns = X.shape[1]
            if num_columns > 1:
                return X[X.columns].astype('str').agg('-'.join, axis=1)
            else:
                return X
        elif isinstance(X, np.ndarray):
            num_dim = X.ndim
            if num_dim == 2:
                return np.unique(X, return_inverse=True, axis=0)[1]
            elif num_dim == 1:
                return X
        elif isinstance(X, pd.Series):
            return X

    def append_and_merge(self, X, Y):
        """
        Appends X and Y horizontally and then merges
        对组合属性进行重新命名
        """
        pdb.set_trace()
        Z = self.append_two_arrays(X, Y)
        return self.merge_columns(Z)

    def empirical_distribution_from_counts(self, counts, size=None):
        """
        Computes the empirical distribution of an attribute
        given the counts of its domain values (a.k.a distinct values)
        """
        if size == None:
            size = np.sum(counts);

        empirical_distribution = counts / size;
        assert np.isclose(np.sum(empirical_distribution), 1, rtol=1e-05, atol=1e-08,
                          equal_nan=False), "Sum of empirical distibution should be 1";
        return empirical_distribution


    def size_and_counts_of_attribute(self,X):
        """
        Returns the size, and the value counts of X
        """
        X = self.merge_columns(X)
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            counts = X.value_counts()
            length = len(X.index)
        elif isinstance(X, np.ndarray):
            counts = np.unique(X, return_counts=True, axis=0)[1]
            length = np.size(X, 0)

        return length, counts


    def empirical_statistics(self,X):
        """
        Returns the empirical distribution (a.k.a relative frequencies), counts,
        and size of an attribute
        """
        length, counts = self.size_and_counts_of_attribute(X)
        empirical_distribution = self.empirical_distribution_from_counts(counts)
        return empirical_distribution, len(counts), length

    def entropy(self,prob_vector):
        """
        Computes the Shannon entropy of a probability distribution corresponding to
        a random variable
        """
        return sc.entropy(prob_vector, base=2)

    def entropy_plugin(self,X,return_statistics=False):
        """
        The plugin estimator for Shannon entropy H(X) of an attribute X. Can optionally
        return the domain size and length of X
        """
        empiricalDistribution, domainSize, length = self.empirical_statistics(X)
        if return_statistics == True:
            return self.entropy(empiricalDistribution), domainSize, length
        else:
            return self.entropy(empiricalDistribution)

    def entropy_model(self,data_X,X_l):
        """
        不同表格上的熵怎么处理呢？
        顺便确定主键

        """
        entropy_X_l=[]
        primary_key={}
        for x in range(X_l):
            entropy_X_l.append(self.entropy_plugin(x,False))
            if entropy_X_l[-1]==math.log10(data_X.shape[0]):
                primary_key["X_l"]=X_l[x]
        sorted(entropy_X_l,key=lambda en:en[0]) #按熵分两类
        #find_functional_dependencies()
        #分解，怎么分呢？

    def run(self):
        """
        层的移动与测试
        :return:
        """
        data
        L1=set([tuple(i for i in self.R)])
        L=[L1]
        l=0
        X
        =self.data.columns
        while bool(L[l]):
            #attribute set>=2
            data_X=self.remove_duplicate_row(X[L])
            self.entropy_model(data_X,X[l])
            #self.is_primarykey()
            l=l+1
            L.append(self.generate_next_level())


if __name__ == "__main__":
    pdb.set_trace()
    data = read_db("diagnostics.csv")
    #pdb.set_trace()
    tane = schema_decomposition(data)
    t0 = time.time()
    tane.run()
    pdb.set_trace()
    print ("\t=> Execution Time: {} seconds".format(time.time()-t0))
    print ('\t=> {} Rules Found'.format(len(tane.rules)))