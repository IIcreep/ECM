import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import kstest, exponweib, weibull_min
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, root_scalar
from scipy.spatial.distance import cdist

# Read CSV files
df1 = pd.read_csv("/Users/hu/Desktop/Battery/parameter/ECM_I_25C01.csv")
df2 = pd.read_csv("/Users/hu/Desktop/Battery/parameter/ECM_I_25C02.csv")
df3 = pd.read_csv("/Users/hu/Desktop/Battery/parameter/ECM_I_25C03.csv")
df4 = pd.read_csv("/Users/hu/Desktop/Battery/parameter/ECM_I_25C04.csv")
df5 = pd.read_csv("/Users/hu/Desktop/Battery/parameter/ECM_I_25C05.csv")
df6 = pd.read_csv("/Users/hu/Desktop/Battery/parameter/ECM_I_25C06.csv")
df7 = pd.read_csv("/Users/hu/Desktop/Battery/parameter/ECM_I_25C07.csv")
df8 = pd.read_csv("/Users/hu/Desktop/Battery/parameter/ECM_I_25C08.csv")

# df_existing = pd.read_csv('/Users/hu/Desktop/Battery/parameter/ECM_25C_pdf.csv')
icon1 = "Q2"
all_data1 = [df1[icon1].iloc[169],df1[icon1].iloc[170],df1[icon1].iloc[171], df2[icon1].iloc[179],df2[icon1].iloc[180],df2[icon1].iloc[181], df5[icon1].iloc[170],df5[icon1].iloc[171],df5[icon1].iloc[172], df6[icon1].iloc[138],df6[icon1].iloc[139],df6[icon1].iloc[140]]
data1 = np.array(all_data1)
icon2 = "alpha2"
all_data2 = [df1[icon2].iloc[169],df1[icon2].iloc[170],df1[icon2].iloc[171], df2[icon2].iloc[179],df2[icon2].iloc[180],df2[icon2].iloc[181], df5[icon2].iloc[170],df5[icon2].iloc[171],df5[icon2].iloc[172], df6[icon2].iloc[138],df6[icon2].iloc[139],df6[icon2].iloc[140]]
data2 = np.array(all_data2)

# Define mixture Weibull CDF
def mixed_weibull_cdf(x, w1, a1, c1, a2, c2):
    w2 = 1 - w1
    cdf1 = exponweib.cdf(x, a=1, c=c1, loc=0, scale=a1)
    cdf2 = exponweib.cdf(x, a=1, c=c2, loc=0, scale=a2)
    return w1 * cdf1 + w2 * cdf2

def joint_neg_log_likelihood_with_gumbel(params, data1, data2):
    w1_1, a1_1, c1_1, a2_1, c2_1, w1_2, a1_2, c1_2, a2_2, c2_2, theta = params
    
    # Calculate CDF and PDF
    cdf_data1 = np.array([mixed_weibull_cdf(x, w1_1, a1_1, c1_1, a2_1, c2_1) for x in data1])
    cdf_data2 = np.array([mixed_weibull_cdf(x, w1_2, a1_2, c1_2, a2_2, c2_2) for x in data2])

    pdf_data1 = w1_1 * exponweib.pdf(data1, a=1, c=c1_1, loc=0, scale=a1_1) + (1 - w1_1) * exponweib.pdf(data1, a=1, c=c2_1, loc=0, scale=a2_1)
    pdf_data2 = w1_2 * exponweib.pdf(data2, a=1, c=c1_2, loc=0, scale=a1_2) + (1 - w1_2) * exponweib.pdf(data2, a=1, c=c2_2, loc=0, scale=a2_2)

    epsilon = 1e-5
    cdf_data1 = np.clip(cdf_data1, epsilon, 1 - epsilon)
    cdf_data2 = np.clip(cdf_data2, epsilon, 1 - epsilon)

    # Using Gumbel Copula Density
    temp = cdf_data1 * cdf_data2 * ((-np.log(cdf_data1))**theta + (-np.log(cdf_data2))**theta)**(2 - 2/theta)

    temp = np.clip(temp, 1e-5, None)
    copula_pdf = theta * ((-np.log(cdf_data1))**(theta-1)) * ((-np.log(cdf_data2))**(theta-1)) * np.exp(-(((-np.log(cdf_data1)) ** theta + (-np.log(cdf_data2)) ** theta) ** (1 / theta))) / temp
    
    epsilon = 1e-50
    pdf_data1 = np.clip(pdf_data1, epsilon, None)
    pdf_data2 = np.clip(pdf_data2, epsilon, None)
    copula_pdf = np.clip(copula_pdf, epsilon, None)

    # logarithm of joint density
    log_copula_pdf = np.log(copula_pdf)
    log_pdf_data1 = np.log(pdf_data1)
    log_pdf_data2 = np.log(pdf_data2)
    joint_log_pdf = log_copula_pdf + log_pdf_data1 + log_pdf_data2

    # joint negative log-likelihood
    joint_neg_log_likelihood = -np.sum(joint_log_pdf)
    
    return joint_neg_log_likelihood

bounds = [(0.5, 1), (0.01, 100), (0.01, 20), (0.01, 100), (0.01, 20),
          (0.5, 1), (0.01, 5), (0.01, 20), (0.01, 5), (0.01, 20),
          (1.1, 20)]

# Fitting using differential evolution algorithm
result = differential_evolution(joint_neg_log_likelihood_with_gumbel, bounds, args=(data1, data2), strategy='best1bin', popsize=30, tol=1.e-2)

# 混合Weibull分布的分位数函数（PPF，也就是CDF的逆）
def mixed_weibull_ppf(p_values, w1, a1, c1, a2, c2):
    results = []
    for p in p_values:
        # fa = mixed_weibull_cdf(0, w1, a1, c1, a2, c2) - p
        # fb = mixed_weibull_cdf(100, w1, a1, c1, a2, c2) - p
        # print(f"f(5) = {fa}, f(30) = {fb}")
        root_result = root_scalar(lambda x: mixed_weibull_cdf(x, w1, a1, c1, a2, c2) - p, bracket=[0, 100], method='bisect')
        if root_result.converged:
            results.append(root_result.root)
        else:
            raise ValueError("Root finding for mixed_weibull_ppf did not converge")
    return np.array(results)

print("Optimization Result:")
print(result.x)
w1_1, a1_1, c1_1, a2_1, c2_1, w1_2, a1_2, c1_2, a2_2, c2_2, theta = result.x

# Multivariate Cramér–von Mises test
def multivariate_cvm_test(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)

    Y = cdist(sample1, sample2, 'euclidean')
    X = cdist(sample1, sample1, 'euclidean')

    T1 = np.sum(np.exp(-0.5 * X ** 2)) / (n1 * (n1 - 1))
    T2 = np.sum(np.exp(-0.5 * Y ** 2)) / (n1 * n2)
    T3 = np.sum(np.exp(-0.5 * Y.T ** 2)) / (n2 * n1)
    T4 = 2 * np.exp(-0.5)

    W_square = T1 - T2 - T3 + T4

    return W_square

def gumbel_copula_sample(theta, size=100):
    # Step 1: Generate independent U1, U2 from uniform distribution
    u1 = np.random.uniform(0, 1, size)
    u2 = np.random.uniform(0, 1, size)
    
    # Generate a random variable from exponential distribution
    exp_rand_var = -np.log(np.random.uniform(0, 1, size))
    
    # Step 2: Apply Gumbel copula function
    c = np.exp(-(((-np.log(u1))**theta + (-np.log(u2))**theta)**(1/theta)))
    
    # Generate copula samples
    v1 = np.exp(-((-np.log(c) + exp_rand_var) / theta)**(1/theta))
    v2 = np.exp(-((-np.log(c) + exp_rand_var) / theta)**(1/theta))
    
    return v1, v2

def monte_carlo_p_value(obs_data, params_fitted, num_simulations=500):
    observed_w_square = multivariate_cvm_test(obs_data, obs_data)
    print(observed_w_square)
    count = 0
    theta = params_fitted[-1]

    w1_1, a1_1, c1_1, a2_1, c2_1, w1_2, a1_2, c1_2, a2_2, c2_2 = params_fitted[:10]
    
    for _ in range(num_simulations):
        # Generate simulation data for Gumbel Copula
        v1, v2 = gumbel_copula_sample(theta, 1)

        # Generating simulated data from a mixed Weibull distribution using the inverse of CDF values
        sim_data1 = mixed_weibull_ppf(v1, w1_1, a1_1, c1_1, a2_1, c2_1)
        sim_data2 = mixed_weibull_ppf(v2, w1_2, a1_2, c1_2, a2_2, c2_2)
        sim_data = np.column_stack((sim_data1, sim_data2))

        # Calculate W^2 for simulated data
        sim_w_square = multivariate_cvm_test(obs_data, sim_data)

        if sim_w_square >= observed_w_square:
            count += 1

    p_value = count / num_simulations
    return p_value

params_fitted = [w1_1, a1_1, c1_1, a2_1, c2_1, w1_2, a1_2, c1_2, a2_2, c2_2, theta]
obs_data = np.column_stack((data1, data2))
p_value = monte_carlo_p_value(obs_data, params_fitted)
print(p_value)

# KS test
fit_cdf_data1 = np.array([mixed_weibull_cdf(x, w1_1, a1_1, c1_1, a2_1, c2_1) for x in np.sort(data1)])
fit_cdf_data2 = np.array([mixed_weibull_cdf(x, w1_2, a1_2, c1_2, a2_2, c2_2) for x in np.sort(data2)])

ks_stat1, p_value1 = kstest(data1, mixed_weibull_cdf, args=(w1_1, a1_1, c1_1, a2_1, c2_1))
ks_stat2, p_value2 = kstest(data2, mixed_weibull_cdf, args=(w1_2, a1_2, c1_2, a2_2, c2_2))

print(f"KS statistic for data1: {ks_stat1}, p-value: {p_value1}")
print(f"KS statistic for data2: {ks_stat2}, p-value: {p_value2}")

# Define Gumbel Copula sample generation function
def gumbel_copula_sample(theta, n):
    u1 = np.random.uniform(0, 1, n)
    u2 = np.random.uniform(0, 1, n)
    c = np.exp(-((-np.log(u1))**theta + (-np.log(u2))**theta)**(1/theta))
    return u1, c / u1

# Define the PPF function of the mixed Weibull distribution
def mixed_weibull_ppf(u, w1, a1, c1, a2, c2):
    return np.where(u < w1, weibull_min.ppf(u / w1, c1, scale=a1), weibull_min.ppf((u - w1) / (1 - w1), c2, scale=a2))

# Generate simulation data for Gumbel Copula
u1, u2 = gumbel_copula_sample(theta, 1000)

# Generating simulated data from a mixed Weibull distribution using the inverse of CDF values
sim_data1 = mixed_weibull_ppf(u1, w1_1, a1_1, c1_1, a2_1, c2_1)
sim_data2 = mixed_weibull_ppf(u2, w1_2, a1_2, c1_2, a2_2, c2_2)

# Draw 3D images
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sim_data1, sim_data2, u1, c='r', marker='o')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('CDF U1')
plt.show()

from sklearn.ensemble import IsolationForest
from sklearn import svm

# 定义混合Weibull CDF
def mixed_weibull_cdf(x, w1, a1, c1, a2, c2):
    w2 = 1 - w1
    cdf1 = exponweib.cdf(x, a=1, c=c1, loc=0, scale=a1)
    cdf2 = exponweib.cdf(x, a=1, c=c2, loc=0, scale=a2)
    return w1 * cdf1 + w2 * cdf2

def joint_pdf_with_gumbel(params, data1, data2):
    w1_1, a1_1, c1_1, a2_1, c2_1, w1_2, a1_2, c1_2, a2_2, c2_2, theta = params
    
    # 计算CDF和PDF
    cdf_data1 = mixed_weibull_cdf(data1, w1_1, a1_1, c1_1, a2_1, c2_1)
    cdf_data2 = mixed_weibull_cdf(data2, w1_2, a1_2, c1_2, a2_2, c2_2)

    pdf_data1 = w1_1 * exponweib.pdf(data1, a=1, c=c1_1, loc=0, scale=a1_1) + (1 - w1_1) * exponweib.pdf(data1, a=1, c=c2_1, loc=0, scale=a2_1)
    pdf_data2 = w1_2 * exponweib.pdf(data2, a=1, c=c1_2, loc=0, scale=a1_2) + (1 - w1_2) * exponweib.pdf(data2, a=1, c=c2_2, loc=0, scale=a2_2)

    epsilon = 1e-5
    cdf_data1 = np.clip(cdf_data1, epsilon, 1 - epsilon)
    cdf_data2 = np.clip(cdf_data2, epsilon, 1 - epsilon)

    # 使用Gumbel Copula密度
    temp = cdf_data1 * cdf_data2 * ((-np.log(cdf_data1))**theta + (-np.log(cdf_data2))**theta)**(2 - 2/theta)
    # print(temp)
    temp = np.clip(temp, 1e-5, None)
    copula_pdf = theta * ((-np.log(cdf_data1))**(theta-1)) * ((-np.log(cdf_data2))**(theta-1)) * np.exp(-(((-np.log(cdf_data1)) ** theta + (-np.log(cdf_data2)) ** theta) ** (1 / theta))) / temp
    
    epsilon = 1e-50
    pdf_data1 = np.clip(pdf_data1, epsilon, None)
    pdf_data2 = np.clip(pdf_data2, epsilon, None)
    copula_pdf = np.clip(copula_pdf, epsilon, None)

    joint_pdf = pdf_data1 * pdf_data2 * copula_pdf
    
    return joint_pdf

# 二维数据点
x1, x2 = 0.3636732413445145,0.4439736205240621

# 联合混合韦伯分布参数
params = 0.921,0.271,3.309,0.269,3.820,0.508,2.13e-02,1.21,1.2e-02,4.092,2.1
w1_1, a1_1, c1_1, a2_1, c2_1, w1_2, a1_2, c1_2, a2_2, c2_2, theta = params

copula_pdf = joint_pdf_with_gumbel(params,x1, x2)
print("The PDF value of the data point in the joint distribution is:", copula_pdf)

# 混合韦伯CDF函数
def mixed_weibull_cdf(x, w1, a1, c1, a2, c2):
    return w1 * weibull_min.cdf(x, c1, scale=a1) + (1 - w1) * weibull_min.cdf(x, c2, scale=a2)

# Gumbel Copula的CDF函数
def gumbel_copula_cdf(u1, u2, theta):
    return np.exp(-((-np.log(u1))**theta + (-np.log(u2))**theta)**(1/theta))

# 计算在边缘混合韦伯分布中的CDF值
u1 = mixed_weibull_cdf(x1, w1_1, a1_1, c1_1, a2_1, c2_1)
u2 = mixed_weibull_cdf(x2, w1_2, a1_2, c1_2, a2_2, c2_2)

# 计算Gumbel Copula的CDF值
copula_cdf = gumbel_copula_cdf(u1, u2, theta)

print("The CDF value of the data point in the joint distribution is:", copula_cdf)

# X_train is normal battery data
clf = IsolationForest(contamination=0.1)
clf.fit(X_train)

# X_test is the battery data to be tested
y_pred = clf.predict(X_test)
print(y_pred)

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

y_pred = clf.predict(X_test)
print(y_pred)
