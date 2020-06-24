from scipy import stats as st


class Ttest(object):

    def __init__(self, data_frame1, data_frame2, alpha=0.05):
        self.data_frame1 = data_frame1
        self.data_frame2 = data_frame2
        self.alpha = alpha

    def normal_distribution_test(self):
        _, p1 = st.shapiro(self.data_frame1)
        _, p2 = st.shapiro(self.data_frame2)
        print("p1: " + str(p1))
        print("p2: " + str(p2))
        if p1 > self.alpha and p2 > self.alpha:
            print("両方とも正規分布の可能性がある")
            return True
        else:
            print("少なくとも一方が正規分布ではない")
            return False

    def equal_variance_test(self, bool_normal):
        if bool_normal:
            _, p = st.bartlett(self.data_frame1, self.data_frame2)
            print("eq_val p: " + str(p))
            if p > self.alpha:
                print("等分散性がある可能性がある")
                return True
            else:
                print("等分散性はない")
                return False
        else:
            _, p = st.levene(self.data_frame1, self.data_frame2)
            if p > self.alpha:
                print("等分散性がある可能性がある")
                return True
            else:
                print("等分散性はない")
                return False

    def t_test(self):
        bool_normal = self.normal_distribution_test()
        bool_eq_val = self.equal_variance_test(bool_normal)
        if bool_eq_val:
            t, p = st.ttest_ind(self.data_frame1, self.data_frame2, equal_var=True)
            print("t_test p: " + str(p))
            if p > self.alpha:
                print("平均に差がない")
            else:
                print("平均に差がある")
            t, p = st.ttest_ind(self.data_frame1, self.data_frame2, equal_var=False)
            print("t_test p: " + str(p))
            if p > self.alpha:
                print("平均に差がない")
            else:
                print("平均に差がある")
        else:
            t, p = st.ttest_ind(self.data_frame1, self.data_frame2, equal_var=False)
            print("t_test p: " + str(p))
            if p > self.alpha:
                print("平均に差がない")
            else:
                print("平均に差がある")
