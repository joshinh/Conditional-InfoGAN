import numpy as np
import tensorflow as tf
import pdb
import random
import json
from scipy.stats import mode

import data_utils
import plotting
import model
import utils
import eval

from time import time
from math import floor
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio

tf.logging.set_verbosity(tf.logging.ERROR)

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
if not settings['data'] == "load":
    samples_new, cond_data, pdf, labels = data_utils.get_samples_and_labels(settings)
    samples = {}
    if cond_data is not None:
        cond_samples = {}
        samples['train'] = np.expand_dims(samples_new['train'][:,:,0],-1)
        cond_samples['train'] = np.expand_dims(samples_new['train'][:,:,1], -1)
        samples['vali'] = np.expand_dims(samples_new['vali'][:,:, 0], -1)
        cond_samples['vali'] = np.expand_dims(samples_new['vali'][:,:,1], -1)
        samples['test'] = np.expand_dims(samples_new['test'][:,:,0], -1)
        cond_samples['test'] = np.expand_dims(samples_new['test'][:,:,1], -1)

        cond_samples_train = cond_samples['train']
    else:
        cond_samples_train = None
        samples = samples_new


# --- save settings, data --- #
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t',  k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

if not data == 'load':
    data_path = './experiments/data/' + identifier + '.data.npy'
    np.save(data_path, {'samples': samples, 'pdf': pdf, 'labels': labels})
    print('Saved training data to', data_path)
    if cond_data is not None:
        cond_path = './experiments/cond_data/' + identifier + '.data.npy'
        np.save(cond_path, {'cond_samples': cond_samples_train})
        print('Saved cond images to ', cond_path)

# --- build model --- #

if generate_test:
    eval.sine_plot(identifier, 250)
    exit()

if cond_dim > 0:
    num_signals = 1

if info:
    latent_C, Z, X, CG, CD, CS, cond_sine = model.create_placeholders(batch_size, seq_length, latent_dim, 
                                    num_signals, cond_dim, info, latent_C_dim)
else:
    Z, X, CG, CD, CS, cond_sine = model.create_placeholders(batch_size, seq_length, latent_dim, 
                                    num_signals, cond_dim, info)
    latent_C = None

discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean']
discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 
                  'num_generated_features', 'cond_dim', 'learn_scale']
generator_settings = dict((k, settings[k]) for k in generator_vars)

CGAN = (cond_dim > 0)
if CGAN: assert not predict_labels

if info:
    info_loss, D_loss, G_loss = model.GAN_loss(info, latent_C, latent_C_dim, Z, X, generator_settings, discriminator_settings, 
            kappa, CGAN, CG, CD, CS, cond_sine, wrong_labels=wrong_labels)
    info_solver, D_solver, G_solver, priv_accountant = model.GAN_solvers(info, info_loss, D_loss, G_loss, learning_rate, batch_size, 
            total_examples=samples['train'].shape[0], l2norm_bound=l2norm_bound,
            batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)
else:
    info_loss = None
    D_loss, G_loss = model.GAN_loss(info, latent_C, latent_C_dim, Z, X, generator_settings, discriminator_settings, 
            kappa, CGAN, CG, CD, CS, cond_sine, wrong_labels=wrong_labels)
    D_solver, G_solver, priv_accountant = model.GAN_solvers(info, info_loss, D_loss, G_loss, learning_rate, batch_size, 
            total_examples=samples['train'].shape[0], l2norm_bound=l2norm_bound,
            batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)
    info_solver = None
G_sample = model.generator(info, latent_C, Z, **generator_settings, reuse=True, c=cond_sine)

# --- evaluation --- #

# frequency to do visualisations
vis_freq = max(14000//num_samples, 1)
eval_freq = max(7000//num_samples, 1)

# get heuristic bandwidth for mmd kernel from evaluation samples
heuristic_sigma_training = median_pairwise_distance(samples['vali'])
best_mmd2_so_far = 1000

# optimise sigma using that (that's t-hat)
batch_multiplier = 5000//batch_size
eval_size = batch_multiplier*batch_size
eval_eval_size = int(0.2*eval_size)
if cond_dim > 0:
    num_generated_features = 1
eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
n_sigmas = 2
sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)
with tf.variable_scope("SIGMA_optimizer"):
    sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
    #sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
    #sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
sigma_opt_iter = 2000
sigma_opt_thresh = 0.001
sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

vis_Z = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
if info:
    vis_latent_C = model.sample_latent_C(batch_size, latent_C_dim, "continuous", 0, 1)
if CGAN:
    vis_C = model.sample_C(batch_size, cond_dim, max_val, one_hot)
    vis_cond = data_utils.get_cond_batch(cond_samples_train, batch_size, 0, None)[0]
    
    
    if 'mnist' in data:
        if one_hot:
            if cond_dim == 6:
                vis_C[:6] = np.eye(6)
            elif cond_dim == 3:
                vis_C[:3] = np.eye(3)
                vis_C[3:6] = np.eye(3)
            else:
                raise ValueError(cond_dim)
        else:
            if cond_dim == 6:
                vis_C[:6] = np.arange(cond_dim)
            elif cond_dim == 3:
                vis_C = np.tile(np.arange(3), 2)
            else:
                raise ValueError(cond_dim)
    elif 'eICU_task' in data:
        vis_C = labels['train'][np.random.choice(labels['train'].shape[0], batch_size, replace=False), :]
    if info:
        vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C, latent_C: vis_latent_C, cond_sine: vis_cond})
    else:
        vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C, cond_sine: vis_cond})
else:
    if info:
        vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, latent_C: vis_latent_C})
    else:    
        vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
    vis_C = None

vis_real_indices = np.random.choice(len(samples['vali']), size=6)
vis_real = np.float32(samples['vali'][vis_real_indices, :, :])
if not labels['vali'] is None:
    vis_real_labels = labels['vali'][vis_real_indices]
else:
    vis_real_labels = None
if data == 'mnist':
    if predict_labels:
        assert labels['vali'] is None
        n_labels = 1
        if one_hot: 
            n_labels = 6
            lab_votes = np.argmax(vis_real[:, :, -n_labels:], axis=2)
        else:
            lab_votes = vis_real[:, :, -n_labels:]
        labs, _ = mode(lab_votes, axis=1) 
        samps = vis_real[:, :, :-n_labels]
    else:
        labs = None
        samps = vis_real
    if multivariate_mnist:
        plotting.save_mnist_plot_sample(samps.reshape(-1, seq_length**2, 1), 0, identifier + '_real', n_samples=6, labels=labs)
    else:
        plotting.save_mnist_plot_sample(samps, 0, identifier + '_real', n_samples=6, labels=labs)
elif 'eICU' in data:
    plotting.vis_eICU_patients_downsampled(vis_real, resample_rate_in_min, 
            identifier=identifier + '_real', idx=0)
else:
    plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=6, 
                            num_epochs=num_epochs)

# for dp
target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
dp_trace = open('./experiments/traces/' + identifier + '.dptrace.txt', 'w')
dp_trace.write('epoch ' + ' eps' .join(map(str, target_eps)) + '\n')

trace = open('./experiments/traces/' + identifier + '.trace.txt', 'w')
trace.write('epoch time D_loss G_loss mmd2 that pdf real_pdf\n')

# --- train --- #
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 
              'latent_dim', 'num_generated_features', 'cond_dim', 'max_val', 
              'WGAN_clip', 'one_hot']
train_settings = dict((k, settings[k]) for k in train_vars)


t0 = time()
best_epoch = 0
if info:
    print('epoch\ttime\tinfo_loss\tD_loss\tG_loss\tmmd2\tthat\tpdf_sample\tpdf_real')
else:
    print('epoch\ttime\tD_loss\tG_loss\tmmd2\tthat\tpdf_sample\tpdf_real')
for epoch in range(num_epochs):
    if info:
        info_loss_curr, D_loss_curr, G_loss_curr = model.train_epoch(info, latent_C, latent_C_dim, epoch, samples['train'], cond_samples_train,  labels['train'],
                                            sess, Z, X, CG, CD, CS, cond_sine, info_loss,
                                            D_loss, G_loss, info_solver,
                                            D_solver, G_solver, 
                                            **train_settings)

    else:

        D_loss_curr, G_loss_curr = model.train_epoch(info, latent_C, latent_C_dim, epoch, samples['train'], cond_samples_train, labels['train'],
                                            sess, Z, X, CG, CD, CS, cond_sine, None,
                                            D_loss, G_loss, info_solver,
                                            D_solver, G_solver, 
                                            **train_settings)
        # -- eval -- #

    # visualise plots of generated samples, with/without labels
    if epoch % vis_freq == 0:
        if info:
            if CGAN:
                vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C, latent_C: vis_latent_C, cond_sine: vis_cond})
            else:
                vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, latent_C: vis_latent_C})
        else:            
            if CGAN:
                vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C, cond_sine: vis_cond})
            else:
                vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
        plotting.visualise_at_epoch(vis_sample, data, 
                predict_labels, one_hot, epoch, identifier, num_epochs,
                resample_rate_in_min, multivariate_mnist, seq_length, labels=vis_C)
   
    # compute mmd2 and, if available, prob density
    if epoch % eval_freq == 0:
        ## how many samples to evaluate with?
        eval_Z = model.sample_Z(eval_size, seq_length, latent_dim, use_time)
        eval_latent_C = model.sample_latent_C(batch_size, latent_C_dim, "continuous", 0, 1)
        if 'eICU_task' in data:
            eval_C = labels['vali'][np.random.choice(labels['vali'].shape[0], eval_size), :]
        else:
            eval_C = model.sample_C(eval_size, cond_dim, max_val, one_hot)
            if CGAN:
                eval_cond = data_utils.get_cond_batch(cond_samples_train, batch_size, 0, None)[0]
        eval_sample = np.empty(shape=(eval_size, seq_length, num_signals))
        for i in range(batch_multiplier):
            if CGAN:
                if info:
                    eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size], CG: eval_C[i*batch_size:(i+1)*batch_size],
                        latent_C: eval_latent_C, cond_sine: eval_cond })
                else:
                    eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size], CG: eval_C[i*batch_size:(i+1)*batch_size],
                        cond_sine: eval_cond})
            else:
                if info:
                   eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size],
                    latent_C: eval_latent_C})
                else:
                    eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size]})
        eval_sample = np.float32(eval_sample)
        eval_real = np.float32(samples['vali'][np.random.choice(len(samples['vali']), size=batch_multiplier*batch_size), :, :])
       
        eval_eval_real = eval_real[:eval_eval_size]
        eval_test_real = eval_real[eval_eval_size:]
        eval_eval_sample = eval_sample[:eval_eval_size]
        eval_test_sample = eval_sample[eval_eval_size:]
        
        ## MMD
        # reset ADAM variables
        sess.run(tf.initialize_variables(sigma_opt_vars))
        sigma_iter = 0
        that_change = sigma_opt_thresh*2
        old_that = 0
        while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
            new_sigma, that_np, _ = sess.run([sigma, that, sigma_solver], feed_dict={eval_real_PH: eval_eval_real, eval_sample_PH: eval_eval_sample})
            that_change = np.abs(that_np - old_that)
            old_that = that_np
            sigma_iter += 1
        opt_sigma = sess.run(sigma)
        mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(eval_test_real, eval_test_sample,biased=False, sigmas=sigma))
       
        ## save parameters
        if mmd2 < best_mmd2_so_far and epoch > 10:
            best_epoch = epoch
            best_mmd2_so_far = mmd2
            model.dump_parameters(identifier + '_' + str(epoch), sess)
       
        ## prob density (if available)
        if not pdf is None:
            pdf_sample = np.mean(pdf(eval_sample[:, :, 0]))
            pdf_real = np.mean(pdf(eval_real[:, :, 0]))
        else:
            pdf_sample = 'NA'
            pdf_real = 'NA'
    else:
        # report nothing this epoch
        mmd2 = 'NA'
        that = 'NA'
        pdf_sample = 'NA'
        pdf_real = 'NA'
    
    ## get 'spent privacy'
    if dp:
        spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)
        # get the moments
        deltas = []
        for (spent_eps, spent_delta) in spent_eps_deltas:
            deltas.append(spent_delta)
        dp_trace.write(str(epoch) + ' ' + ' '.join(map(str, deltas)) + '\n')
        if epoch % 10 == 0: dp_trace.flush()

    ## print
    t = time() - t0
    if info:
        try:
            print('%d\t%.2f\t%.4f\t\t%.4f%.4f\t%.5f\t%.0f\t%.2f\t%.2f' % (epoch, t, info_loss_curr, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real))
        except TypeError:       # pdf are missing (format as strings)
            print('%d\t%.2f\t%.4f\t%.4f\t%.4f\t%.5f\t%.0f\t %s\t %s' % (epoch, t, info_loss_curr, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real))
    else:
        try:
            print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t%.0f\t%.2f\t%.2f' % (epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real))
        except TypeError:       # pdf are missing (format as strings)
            print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t%.0f\t %s\t %s' % (epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real))

    ## save trace
    trace.write(' '.join(map(str, [epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real])) + '\n')
    if epoch % 10 == 0: 
        trace.flush()
        plotting.plot_trace(identifier, xmax=num_epochs, dp=dp)

    if shuffle:     # shuffle the training data 
        perm = np.random.permutation(samples['train'].shape[0])
        samples['train'] = samples['train'][perm]
        if cond_dim > 0:
            cond_samples_train = cond_samples_train[perm]
        if labels['train'] is not None:
            labels['train'] = labels['train'][perm]
    
    if epoch % 50 == 0:
        model.dump_parameters(identifier + '_' + str(epoch), sess)

trace.flush()
plotting.plot_trace(identifier, xmax=num_epochs, dp=dp)
model.dump_parameters(identifier + '_' + str(epoch), sess)

## after-the-fact evaluation
#n_test = vali.shape[0]      # using validation set for now TODO
#n_batches_for_test = floor(n_test/batch_size)
#n_test_eval = n_batches_for_test*batch_size
#test_sample = np.empty(shape=(n_test_eval, seq_length, num_signals))
#test_Z = model.sample_Z(n_test_eval, seq_length, latent_dim, use_time)
#for i in range(n_batches_for_test):
#    test_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: test_Z[i*batch_size:(i+1)*batch_size]})
#test_sample = np.float32(test_sample)
#test_real = np.float32(vali[np.random.choice(n_test, n_test_eval, replace=False), :, :])
## we can only get samples in the size of the batch...
#heuristic_sigma = median_pairwise_distance(test_real, test_sample)
#test_mmd2, that = sess.run(mix_rbf_mmd2_and_ratio(test_real, test_sample, sigmas=heuristic_sigma, biased=False))
##print(test_mmd2, that)
