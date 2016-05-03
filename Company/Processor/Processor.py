import numpy as np
import scipy as scp

from matplotlib import pyplot as plt
%matplotlib inline


class Processor:


    def __init__(self):
        self.N = self.n_signal = 0
        self.alpha = 1
        self.Y = self.y0 = None

    def fit(self, Y_fit, y0_fit):
        if self.Y is not None or self.y0 is not None:
            throw("Error: fit already fitted Processor. Use add_fit().")
        self.Y = Y_fit.copy()
        self.y0 = y0_fit.copy()
        self.n_signal = Y_fit.shape[-1]
        self.N = y0_fit.size
        if (Y_fit.shape[0] != self.N):
            print("Error: not matching shapes for profit_item and, profit_case")

    def params_init(self):
        n = self.n_signal; N = self.N
        y0 = self.y0; Y = self.Y
        self.P = np.zeros((n, n-1))
        self.P[np.arange(n-1), np.arange(n-1)] = 1
        self.P[n-1] = -1
        #self.P[-1, -1] = -1
        self.p = np.zeros((n, 1))
        self.p[n-1] = 1

        self.At = np.zeros((N, n, n))
        #print(n, (1+y0)[np.newaxis, :].shape, self.At[:,np.arange(n),np.arange(n)].shape)
        self.At[:,np.arange(n),np.arange(n)] = (1+Y)/(1+y0)[:, np.newaxis]
        #np.diagonal(self.A, axis1=1, axis2=2) =

        self.U = np.diag(np.ones((n,))*self.alpha)


    def add_fit(self, Y_add, y0_add):
        if self.n_signal != Y_add.shape[-1]:
            throw("Error: not appropriate added signals length.")
        self.N += Y_add.shape[0]
        self.Y = np.concatenate((self.Y, Y_add))
        self.y0 = np.concatenate((self.y0, y0_add))

    def CalcRegr_prepare(self, equal_constraint=True):
        n = self.n_signal
        N = self.N
        # transform due to equality constraints
        Y = self.Y.copy()
        y0 = self.y0.copy()
        U = self.U.copy()
        At = self.At.copy()
        P = self.P
        p = self.p
        alpha = self.alpha

        if equal_constraint:
            y0 = y0 - self.Y[:, -1]
            Y = self.Y[:, :-1] - self.Y[:, -1].reshape(-1, 1)
            devider = np.sum(Y*Y, axis=1)
            devider[devider == 0] = 1
            self.a0 = a0 = (y0/devider)[:,np.newaxis]*Y
            a0[devider == 0, :] = 0
            self.Q0 = Q0 = Y[:, np.newaxis, :]*Y[:, :, np.newaxis]
            # not needed self.b0 = b0 = np.zeros((,)) # needed???
            Tmp = np.dot(At[1:,...].transpose(0,2,1), U)
            Tmp = np.sum(Tmp.transpose(0, 2, 1)[...,np.newaxis]*(np.eye(n)[np.newaxis,...] - At[1:,...])[...,np.newaxis,:], -3)
            #print(At[1:,...].transpose(0,2,1).shape, Tmp.shape)
            Tmp = (np.concatenate((np.zeros((1, n, n)), Tmp), axis=0) +
                U.dot(At-np.eye(n)[np.newaxis,...]).transpose(1, 0, 2))
            a0 += (1/devider[:, np.newaxis])*np.dot(P.T, Tmp).transpose(1, 0, 2)[:, :, -1]

            Tmp = np.dot(P.T, U)
            At = np.transpose(np.dot((np.linalg.inv(Tmp.dot(P))).dot(Tmp), At), axes=(1, 0, 2)).dot(P)
            U = Tmp.dot(P)
        else:
            devider = np.sum(Y*Y, axis=1)
            devider[devider == 0] = 1
            self.a0 = a0 = (y0/devider)[:,np.newaxis]*Y
            a0[devider == 0, :] = 0
            self.Q0 = Q0 = Y[:, np.newaxis, :]*Y[:, :, np.newaxis]

        self.U = U
        self.At = At
        self.trans_Y = Y
        self.trans_y0 = y0
        return Y, y0, U, At

    def CalcRegression(equal_const=True):
        n = self.n_signal
        s_len = self.sig_len
        # transform due to equality constraints

    def ProcessTo(self):
        N = self.N; n = self.n_signal
        al = self.al = np.ones((N, n-1))
        Ql = self.Ql = np.empty((N, n-1, n-1))
        bl = self.bl = np.empty((N, n-1))

        a0 = self.a0
        Q0 = self.Q0

        al[0] = self.a0[0]
        bl[0] = np.zeros((n-1,))
        Ql[0] = self.Q0[0]

        U = self.U
        At = self.At

        for t in range(1, N):
            Tmp = U.dot(At[t]).dot(np.linalg.inv((At[t].T).dot(U).dot(At[t]) + Ql[t-1])).dot(
                Ql[t-1]).dot(np.linalg.inv(At[t]))
            Ql[t] = Q0[t] + Tmp
            al[t] = np.linalg.inv(Ql[t]).dot(Q0[t].dot(a0[t]) + Tmp.dot(At[t]).dot(al[t-1]))
            bl[t] =(bl[t-1]+np.dot((a0[t] - al[t]).T, Q0[t]).dot(a0[t]) +
                np.dot((al[t-1]-((At[t]).T).dot(al[t])).T, Ql[t]-Q0[t]).dot(al[t-1]))

    def ProcessFrom(self):
        N = self.N; n = self.n_signal
        ar = self.ar = np.empty((N, n-1))
        Qr = self.Qr = np.empty((N, n-1, n-1))
        br = self.br = np.empty((N, n-1))

        a0 = self.a0
        Q0 = self.Q0

        ar[N-1] = self.a0[N-1]
        br[N-1] = np.zeros((n-1,))
        Qr[N-1] = self.Q0[N-1]

        U = self.U
        At = self.At

        for t in range(N-2, -1, -1):
            Tmp = (At[t+1].T).dot(U).dot(np.linalg.inv(U+Qr[t+1])).dot(Qr[t+1])
            Qr[t] = Q0[t] + Tmp.dot(At[t+1])
            ar[t] = np.linalg.inv(Qr[t]).dot(Q0[t].dot(a0[t]) + Tmp.dot(ar[t+1]))
            br[t] =(br[t+1]+np.dot((a0[t] - ar[t]).T, Q0[t]).dot(a0[t]) +
                np.dot((ar[t+1]-((At[t+1]).T).dot(ar[t])).T, U).dot(
                    np.linalg.inv(U+Qr[t+1])).dot(Qr[t+1].dot(ar[t+1])))

    def Merge(self):
        ar = self.ar; al = self.al; a0 = self.a0
        Qr = self.Qr; Ql = self.Ql; Q0 = self.Q0
        br = self.br; bl = self.bl
        N = self.N; n = self.n_signal

        self.part_item = np.zeros((N, n-1))
        for t in range(N):
            if 0<t<N-1:
                Tmp1 = (At[t+1].T).dot(U).dot(np.linalg.inv(U+Qr[t+1])).dot(Qr[t+1])
                Tmp2 = U.dot(At[t]).dot(np.linalg.inv((At[t].T).dot(U).dot(At[t]) + Ql[t-1])).dot(
                        Ql[t-1])
                Q_t = Tmp1.dot(At[t+1]) + Q0[t]+Tmp2.dot(np.linalg.inv(At[t]))
                self.part_item[t] = np.linalg.inv(Q_t).dot(Tmp1.dot(ar[t+1])+Q0[t].dot(a0[t])+Tmp2.dot(al[t-1]))
            if not t:
                Tmp1 = (At[t+1].T).dot(U).dot(np.linalg.inv(U+Qr[t+1])).dot(Qr[t+1])
                self.part_item[t] = np.linalg.inv(Qr[t]+Q0[t]).dot(Tmp1.dot(ar[t+1])+Q0[t].dot(a0[t]))
            if t==N-1:
                Tmp2 = U.dot(At[t]).dot(np.linalg.inv((At[t].T).dot(U).dot(At[t]) + Ql[t-1])).dot(
                        Ql[t-1])
                self.part_item[t] = np.linalg.inv(Ql[t]+Q0[t]).dot(Tmp2.dot(al[t-1])+Q0[t].dot(a0[t]))
