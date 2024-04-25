import random
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import svm
from tca import TCA
from sklearn.model_selection import KFold,cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import MinMaxScaler

#读取数据
data=pd.read_excel("文件路径",header=,sheet_name=)#读取文件
data_feature_df=data.drop(columns=[])#输入特征
data_target_df=data[]#输出标签
#格式转换
data_feature=np.array(data_feature_df)
data_target=np.array(data_target_df)
data_target=np.log10(np.array(data_target_df)).reshape(-1,1)#寿命取对数
#最大最小值归一化
scalarx=MinMaxScaler()
scalary=MinMaxScaler()
data_feature=scalarx.fit_transform(data_feature)
data_target=scalary.fit_transform(data_target)

#划分源域&目标域
Xs=data_feature[:n_source,:]
Ys=data_target[:n_source]
Xt=data_feature[n_source:,:]
Yt=data_target[n_source:]

#参数设置
kf=5#5折交叉
num=20#多次取均值
r2=np.zeros([num])#R2
r2_mean=np.zeros([16])
rsme=np.ones([num])#RSME
rsme_mean=np.ones([16])
rr=np.ones([num])#随机种子
kk=('linear','poly','rbf')#SVM核函数寻优范围
svmp={'c':(1,50),'gam':(1e-2,10),'dgr':(3,5),'co':(1e-2,10),'ker':(0,2.9)}#SVM超参数寻优范围
tcap={'lam':(1,50)}#TCA超参数寻优范围

for u in [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:#源域融合数量
    for z in range(num):
        r=random.randint(0,1000)
        rr[z]=r
        
        #源域融合
        X1,X2,Y1,Y2=train_test_split(
            Xt,Yt,train_size=u/66+1e-8,random_state=r)#随机抽取源域融合数据
        Xst=np.concatenate((Xs,X1))
        Yst=np.concatenate((Ys,Y1))
    
        #聚类指标
        def tcao(lam):
            tca=TCA(kernel_type='linear',dim=30,gamma=1,lamb=lam)
            Xss,Xtt=tca.fit(Xst,X2)
            km1=KMeans(n_clusters=1)    #分别聚类
            km1.fit(Xss)
            cc1=km1.cluster_centers_
            km2=KMeans(n_clusters=1)
            km2.fit(Xtt)
            cc2=km2.cluster_centers_
            dis=(cc1-cc2)**2            #计算聚类中心距离作为指标
            score=-1*np.mean(dis)
            return score
        #设定贝叶斯优化
        optimizertca=BayesianOptimization(
            f=tcao,  # 目标函数（聚类指标）
            pbounds=tcap,  # 取值空间
            verbose=2,  
            allow_duplicate_points=True)
        #运行贝叶斯优化
        optimizertca.maximize(  
            init_points=5,  
            n_iter=10)  
        #获得最优参数
        tcabp=optimizertca.max['params']
        lambb=tcabp['lam']
        
        #进行TCA映射
        tca=TCA(kernel_type='linear',dim=30,gamma=1,lamb=lambb)
        Xss,Xtt=tca.fit(Xst,X2)
        
        #基预测器
        #优化SVM超参数
        def svmt(c,ker,gam,co,dgr):
            val = cross_val_score(svm.SVR(kernel=kk[int(ker)],gamma=gam,C=c,degree=int(dgr),coef0=co),
                                  Xss, Yst,cv=kf).mean()#SVM交叉验证
            return val
        optimizert=BayesianOptimization(
            f=svmt,  
            pbounds=svmp,  
            verbose=2, 
            allow_duplicate_points=True)
        optimizert.maximize(  
            init_points=5,  
            n_iter=10) 
        tp=optimizert.max['params']
        ct=tp['c']
        kt=tp['ker']
        gt=tp['gam']
        cot=tp['co']
        dgrt=tp['dgr']
        #训练基预测器
        svr=svm.SVR(kernel=kk[int(kt)],gamma=gt,C=ct,degree=int(dgrt),coef0=cot)
        svr.fit(Xss,Yst)
        #预测测试集
        ypp=svr.predict(Xtt)                         #预测值
        r2[z]=r2_score(Y2,ypp)                      #R2
        rsme[z]=mean_squared_error(Y2,ypp)**0.5     #RSME
        yp=10**scalary.inverse_transform(ypp)
    r2_mean[u-5]=np.mean(r2)
    rsme_mean[u-5]=np.mean(rsme)

for i in [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    print(i,'组融合数据：r2:',r2_mean[i-5],' rsme:',rsme_mean[i-5])