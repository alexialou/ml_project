#
#	runs stacked auto encoder on provided dataset
#

import argparse
import numpy as np
import pandas as pd
import sdae
from sklearn.model_selection import train_test_split
import theano
import timeit

def run(datapath):
    normed = pd.read_csv(datapath)
    normed = normed.loc[normed['clerk_school'].notnull()]
    normed = normed.drop(['judge'], axis=1)
    normed['clerk_school'] = pd.Categorical(normed.clerk_school).codes
    y_data = normed['clerk_school'].as_matrix()
    X_data = normed.drop(['clerk_school'], axis=1).as_matrix()

    uniq_sch = len(np.unique(y_data))
    feature_num = X_data.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = theano.shared(X_train.astype('float64'))
    y_train = theano.shared(y_train.astype('int32'))
    X_val = theano.shared(X_val.astype('float64'))
    y_val = theano.shared(y_val.astype('int32'))
    X_test = theano.shared(X_test.astype('float64'))
    y_test = theano.shared(y_test.astype('int32'))

    datasets = np.array([(X_train, y_train), (X_val, y_val), (X_test, y_test)])


    finetune_lr=0.1
    pretraining_epochs=15
    pretrain_lr=0.001,
    training_epochs=500
    batch_size=1

    X_train = datasets[0][0]
    n_train_batches = X_train.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    numpy_rng = np.random.RandomState(89677)
    encoder = sdae.SdA(numpy_rng=numpy_rng,
                       n_ins=feature_num,
                       hidden_layers_sizes=[1000, 1000, 1000],
                       n_outs=uniq_sch)
    pretraining_fns = encoder.pretraining_functions(train_set_x=X_train, batch_size=batch_size)
    print('... pre-training the model')

    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in range(encoder.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, corruption=corruption_levels[i]))
    #             c.append(pretraining_fns[i](index=batch_index,
    #                                         corruption=corruption_levels[i],
    #                                         lr=pretrain_lr))

            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c, dtype='float64')))
    end_time = timeit.default_timer()
    print(('The pretraining code ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = encoder.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    print('... finetunning the model')
        # early-stopping parameters
    patience = 100 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                                # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        # test it on the test set
                        test_losses = test_model()
                        test_score = np.mean(test_losses, dtype='float64')
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
                if patience <= iter:
                    done_looping = True
                break

    end_time = timeit.default_timer()
    print((
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training code ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath') 
    args = parser.parse_args()
    run(args.datapath)

