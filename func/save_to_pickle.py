import pickle
import sys


def picklesave(dataA, sigmaA, maskA, maskANull, dataB1, sigmaB1, dataB2, sigmaB2, dataB3, sigmaB3, dataC, sigmaC, dataD,
               sigmaD, maskD, maskDNull, dataE, sigmaE, sumsim, xraysim1, xraysim2, xraysim3, szsim):
    """
    save projected data to pickle for test
    :param dataA:
    :param sigmaA:
    :param maskA:
    :param maskANull:
    :param dataB1:
    :param sigmaB1:
    :param dataB2:
    :param sigmaB2:
    :param dataB3:
    :param sigmaB3:
    :param dataC:
    :param sigmaC:
    :param dataD:
    :param sigmaD:
    :param maskD:
    :param maskDNull:
    :param dataE:
    :param sigmaE:
    :param sumsim:
    :param xraysim1:
    :param xraysim2:
    :param xraysim3:
    :param szsim:
    :return:
    """

    data = {'dataA': dataA, 'dataB1': dataB1, 'dataB2': dataB2, 'dataB3': dataB3, 'dataC': dataC, 'dataD': dataD,
            'dataE': dataE}
    sigmas = {'sigmaA': sigmaA, 'sigmaB1': sigmaB1, 'sigmaB2': sigmaB2, 'sigmaB3': sigmaB3, 'sigmaC': sigmaC,
              'sigmaD': sigmaD, 'sigmaE': sigmaE}
    masks = {'maskA': maskA, 'maskANull': maskANull, 'maskD': maskD, 'maskDNull': maskDNull}
    sims = {'sumsim': sumsim, 'xraysim1': xraysim1, 'xraysim2': xraysim2, 'xraysim3': xraysim3, 'szsim': szsim}

    pickle_output = open('projected_data_for_test.pkl', 'wb')
    pickle.dump(data, pickle_output)
    pickle.dump(sigmas, pickle_output)
    pickle.dump(masks, pickle_output)
    pickle.dump(sims, pickle_output)
    pickle_output.close()
    print "Done with pickle"
    sys.stdout.flush()
    sys.exit()


def pickleread():
    pickle_output = open('projected_data_for_test.pkl', 'rb')
    data = pickle.load(pickle_output)
    sigmas = pickle.load(pickle_output)
    masks = pickle.load(pickle_output)
    sims = pickle.load(pickle_output)
    dataA = data['dataA']
    dataB1 = data['dataB1']
    dataB2 = data['dataB2']
    dataB3 = data['dataB3']
    dataC = data['dataC']
    dataD = data['dataD']
    dataE = data['dataE']
    sigmaA = sigmas['sigmaA']
    sigmaB1 = sigmas['sigmaB1']
    sigmaB2 = sigmas['sigmaB2']
    sigmaB3 = sigmas['sigmaB3']
    sigmaC = sigmas['sigmaC']
    sigmaD = sigmas['sigmaD']
    sigmaE = sigmas['sigmaE']
    maskA = masks['maskA']
    maskANull = masks['maskANull']
    maskD = masks['maskD']
    maskDNull = masks['maskDNull']
    sumsim = sims['sumsim']
    xraysim1 = sims['xraysim1']
    xraysim2 = sims['xraysim2']
    xraysim3 = sims['xraysim3']
    szsim = sims['szsim']
    return (
    dataA, sigmaA, maskA, maskANull, dataB1, sigmaB1, dataB2, sigmaB2, dataB3, sigmaB3, dataC, sigmaC, dataD, sigmaD,
    maskD, maskDNull, dataE, sigmaE, sumsim, xraysim1, xraysim2, xraysim3, szsim)