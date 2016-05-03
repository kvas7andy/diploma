import numpy as np

def nonstatRegress(assets, fund, lam, outPoint, dynamicmodel ):
    n = np.size(assets,1)
    T = np.size(assets,0)

    Q =  np.zeros([T * n, T * n])
    Y = np.zeros([T * n,1])
    regul = 0*np.eye(n)


    VtUVt = np.eye(n)
    VtU   = np.eye(n)
    lambd = np.zeros([n,1]) +1
    for i in range(n):
        lambd[i] = lam# lam/abs(np.mean(assets[:,i]))

    U = np.diagflat(lambd)
    for t in range(T):

        returns_t = assets[t,:]

        for i in range(n):
            if dynamicmodel>0:
               # import ipdb; ipdb.set_trace()
                vti  = (1+returns_t[0,i]/100)/(1+fund[t]/100)

            else:
                vti = 1
            VtU[i,i] = vti*lambd[i]
            VtUVt[i,i] = vti*lambd[i]*vti

        if outPoint[t]==0:
            xxT = np.dot(returns_t.T,returns_t) + regul
            yx  = fund[t]* returns_t
        else:
            xxT = np.zeros([n,n])
            yx  = np.zeros([n])
        a, b, c = t * n, (t + 1) * n, (t + 2) * n

        if t==0:
            Q[a:b, a:b] =  xxT + VtUVt
        elif t==T-1:
            Q[a:b, a:b] =  xxT + U
        else:
            Q[a:b, a:b] =  xxT + VtUVt + U
        #import ipdb; ipdb.set_trace()
        Y[a:b,0] = yx

        if t < T-1: Q[b:c, a:b] = Q[a:b, b:c] = -VtU


    beta = np.dot(np.linalg.inv(Q), Y)
    beta = np.reshape(beta,(T,n))
    fund_est = np.zeros([T,1])
    for t in range(T):
        returns_t = assets[t,:]
        beta_t = beta[t,:]
        fund_est[t,0]  = np.dot(beta[t,:],assets[t,:].T)
    return beta, fund_est

def AIC(assets, fund, lam, dynamicmodel):
    n = np.size(assets,1)
    T = np.size(assets,0)
    outPoint = np.zeros([T,1])
    arrR2 = np.zeros([T,1])
    fund_est = np.zeros([T,1])
    beta, fund_est = nonstatRegress(assets, fund, lam, outPoint, dynamicmodel)
    VtUVt = np.eye(n)
    VtU   = np.eye(n)
    Q =  np.zeros([T * n, T * n])
    XXT =  np.zeros([T * n, T * n])
    Blam =  np.zeros([T * n, T * n])

    lambd = np.zeros([n,1]) +1
    for i in range(n):
        lambd[i] = lam# lam/abs(np.mean(assets[:,i]))
    U = np.diagflat(lambd)

    for t in range(0,T):

       # Model
       # import ipdb; ipdb.set_trace()
        beta_t = beta[t,:]
    
        assets_t = assets[t,:]
        
        fund_t = fund[t,0]
      #  import ipdb; ipdb.set_trace()

        fund_t_est = np.dot(beta_t,assets_t.T)
        fund_est[t,0] = fund_t_est
        arrR2[t,0] = (fund_t-fund_t_est)*(fund_t-fund_t_est)

        # Penalty

        for i in range(n):
            if dynamicmodel>0:
               # import ipdb; ipdb.set_trace()
                vti  = (1+assets_t[0,i]/100)/(1+fund[t]/100)

            else:
                vti = 1
            VtU[i,i] = vti*lambd[i]
            VtUVt[i,i] = vti*lambd[i]*vti


        xxT = np.dot(assets_t.T,assets_t)


        a, b, c = t * n, (t + 1) * n, (t + 2) * n

        if t==0:

            XXT[a:b, a:b]  =  xxT
            Blam[a:b, a:b] = VtUVt
        elif t==T-1:
            XXT[a:b, a:b]  =  xxT
            Blam[a:b, a:b] = U

        else:
            XXT[a:b, a:b]  =  xxT
            Blam[a:b, a:b] = VtUVt + U
        if t < T - 1: Blam[b:c, a:b] = Blam[a:b, b:c] = -VtU
    Q = np.dot(XXT,np.linalg.inv(XXT+Blam))
    trace = np.trace(Q)
    r2 = np.mean(arrR2)
    return r2,trace


def LeaveOneOut(assets, fund, lam, dynamicmodel, band=1):
    n = np.size(assets,1)
    T = np.size(assets,0)
    outPoint = np.zeros([T,1])
    arrR2 = np.zeros([T,1])
    fund_est = np.zeros([T,1])

    for t in range(band,T-band):
  #      if t == 26 or t == 39 or t== 41 or t == 47 or t==49:
  #        continue
       # Model
       # import ipdb; ipdb.set_trace()

        outPoint = np.zeros([T,1])
        outPoint[t] = 1
        assets_t = assets[t,:]
        beta, fund_est = nonstatRegress(assets, fund, lam, outPoint, dynamicmodel)
        beta_t = beta[t,:]
        fund_t = fund[t,0]
      #  import ipdb; ipdb.set_trace()
        fund_t_est = np.dot(beta_t,assets_t.T)
        fund_est[t,0] = fund_t_est
        arrR2[t,0] = (fund_t-fund_t_est)*(fund_t-fund_t_est)



    r2 = np.mean(arrR2)
    return arrR2,r2,fund_est

def LeaveHalfOut(assets, fund, lam, dynamicmodel):
    n = np.size(assets,1)
    T = np.size(assets,0)
    outPoint = np.zeros([T,1])
    arrR2 = np.zeros([T,1])
    fund_est = np.zeros([T,1])
    for t in range(1,T-1,2):
        outPoint[t,0] = 1

    beta1, fund_est = nonstatRegress(assets, fund, lam, outPoint, dynamicmodel)
    for t in range(1,T-1,2):
        assets_t = assets[t,:]
        beta_t = beta1[t,:]
        fund_t = fund[t,0]
      #  import ipdb; ipdb.set_trace()
        fund_t_est = np.dot(beta_t,assets_t.T)
        fund_est[t,0] = fund_t_est
        arrR2[t,0] = (fund_t-fund_t_est)*(fund_t-fund_t_est)
    outPoint = np.zeros([T,1])
    for t in range(2,T-1,2):
        outPoint[t,0] = 1

    beta2, fund_est = nonstatRegress(assets, fund, lam, outPoint, dynamicmodel)
    for t in range(2,T-1,2):
        beta_t = beta1[t,:]
        fund_t = fund[t,0]
        assets_t = assets[t,:]
      #  import ipdb; ipdb.set_trace()
        fund_t_est = np.dot(beta_t,assets_t.T)
        fund_est[t,0] = fund_t_est
        arrR2[t,0] = (fund_t-fund_t_est)*(fund_t-fund_t_est)



    r2 = np.mean(arrR2)
    return arrR2,r2,fund_est

def CumReturn(ret):
    T = np.size(ret,0)
    cumret = np.zeros(np.size(ret))
    cumret[0] = 100
    for t in range(1,T-1):
       cumret[t] = cumret[t-1] + cumret[t-1]*ret[t,0]/100
    cumret[T-1] = cumret[t-1]  
    return cumret

def ROC(e_array):
    sort_e = np.sort(e_array,0)
    sort_e = sort_e[::-1,0]
    T = np.size(e_array,0)
    RROCX = np.zeros([T+2,1])
    RROCY = np.zeros([T+2,1])
    RROCX[0] = 0
    RROCY[0] =  float("inf")
    for t in range(0,T):
        s = -sort_e[t]
        error_s = e_array + s
        RROCX[t+1] = np.sum(error_s[error_s>0])
        RROCY[t+1] = np.sum(error_s[error_s<=0])
    RROCX[T+1] = float("inf")
    RROCY[T+1] = 0
    AOC = 0.0
    for t in range(1,T):
        temp =  - 1/2*(RROCY[t+1] + RROCY[t])*(RROCX[t+1]-RROCX[t])
        AOC = AOC + temp# 1/2*(RROCY[t+1] - RROCY[t])*(RROCX[t+1]+RROCX[t])
    return AOC, RROCX, RROCY
