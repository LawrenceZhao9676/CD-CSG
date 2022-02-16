#! -*- coding: utf-8 -*-
import math
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import tensorflow as tf


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    Q = I - unit / n
    return np.dot(np.dot(Q, K), Q)


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / sigma / sigma
    np.exp(KX, KX)
    return KX


def HSIC(X, Y):
    return np.sum(centering(rbf(X)) * centering(rbf(Y)))


def GenLearning(data):
    """Generalized learning in the star graph.

    Args:
        data (np.ndarray): Data of the star graph {ui,...,un,vc}.
    Returns:
        return (np.ndarray): The intermediate variable of the star graph's
                             non-central variables.
    """
    # load data
    cidata = data
    N = cidata.shape[0]
    D = cidata.shape[1]
    u_train = cidata[:, :(D - 1)]
    v_train = cidata[:, D - 1]
    u_train = u_train.astype('float32')
    v_train = v_train.astype('float32')

    batch_size = N
    original_dim = D - 1
    latent_dim = 1
    h1 = 15
    h2 = 7
    h3 = 15
    epochs = 50

    u = Input(shape=(original_dim,))  # N-15-7-15-1
    v = Input(shape=(1,))
    m1 = Dense(h1, activation='relu',
               kernel_initializer='random_uniform',
               bias_initializer='zeros')(u)
    m2 = Dense(h2, activation='relu')(m1)
    m3 = Dense(h3, activation='relu')(m2)

    # mean var
    M_mean = Dense(latent_dim)(m3)
    M_log_var = Dense(latent_dim)(m3)

    # reparameterization 
    def sampling(args):
        M_mean, M_log_var = args
        epsilon = K.random_normal(shape=K.shape(M_mean))
        return M_mean + K.exp(M_log_var / 2) * epsilon

    M = Lambda(sampling, output_shape=(latent_dim,))([M_mean, M_log_var])

    d1 = h3
    d2 = h2
    d3 = h1
    # decoder
    decoder1 = Dense(d1, activation='relu')(M)  # 1-15-7-15-N
    decoder2 = Dense(d2, activation='relu')(decoder1)
    decoder3 = Dense(d3, activation='relu')(decoder2)
    u_hat = Dense(original_dim, activation='sigmoid')(decoder3)

    # build model
    vae = Model([u, v], u_hat)

    xent_loss = K.mean(K.square(u - u_hat), axis=-1)
    func_loss = K.mean(K.square(v - u), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + M_log_var - K.square(M_mean) - K.exp(M_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + func_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')

    vae.fit([u_train, v_train],
            shuffle=True,
            epochs=epochs,
            verbose=0,
            batch_size=batch_size
            )

    # encode
    encoder1 = Model(u, M_mean)
    encoder2 = Model(u, M_log_var)
    u_test_encoded1 = encoder1.predict(u_train, batch_size=batch_size)
    u_test_encoded2 = encoder2.predict(u_train, batch_size=batch_size)
    M = Lambda(sampling, output_shape=(latent_dim,))([u_test_encoded1, u_test_encoded2])
    M = K.eval(M)
    return M


def CSG_model(data, seed=0):
    """The forward and backward model of CD-CSG.

    Args:
        data (np.ndarray): Data of the star graph {ui,...,un,vc}.
        seed (int): The random seed.
    Returns:
        return (np.ndarray): Causality of ui and vc. 0: vc->ui; 1: ui->vc; -1: Non-identifiable.
    """

    cidata = data
    D = cidata.shape[1]
    u = cidata[:, :(D - 1)]
    v = cidata[:, D - 1]
    result = np.zeros(D - 1)

    tf.random.set_seed(seed)
    # forward
    M = GenLearning(data)
    v = v.reshape(-1, )
    M = M.reshape(-1, )
    f1 = np.polyfit(M, v, 5)
    p1 = np.poly1d(f1)
    nu = p1(M) - v
    M = M.reshape(-1, 1)
    nu = nu.reshape(-1, 1)
    u_to_v = HSIC(M, nu)
    # backward
    for i in range(0, D - 1):
        v = v.reshape(-1, )
        u_t = u[:, i].reshape(-1, )
        f2 = np.polyfit(v, u_t, 5)
        p2 = np.poly1d(f2)
        nv = p2(v) - u_t
        v = v.reshape(-1, 1)
        nv = nv.reshape(-1, 1)
        v_to_u = HSIC(v, nv)
        delta = 0.05 * min(u_to_v, v_to_u)
        if u_to_v + delta < v_to_u:
            result[i] = 1
        elif v_to_u + delta < u_to_v:
            result[i] = 0
            '''
        else:
            result[i] = -1
            '''
    return result

