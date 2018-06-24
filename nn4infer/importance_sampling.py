import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from collections import namedtuple

def softplus(x, limit=30):
  if x > limit:
    return x
  else:
    return np.log(1.0 + np.exp(x))

LossInfo = namedtuple('LossInfo',
    ('samples, sample_logits, weights, log_weights, log_ps,'
      'log_qs, kl_divergence, penalty, loss, errors, gvs'))

def make_loss(log_p, inference_dist, logits, logits_scale=1e+2,
              penalty_scale=1e+0, n_samples=10, use_logits=True):
  """
  Args:
    log_p: Callable from tensor of the shape `[n_dims]` to scalar.
    inference_dist: An instance of `tfd.Distribution`.
    logits: Callable from tensor of the shape `[n_samples, n_dims]`
      to tensor of the shape `[n_samples]`.
    logits_scale: Positive float.
    n_samples: Positive integer.
  Returns:
    An instance of `LossInfo`.
  """
  with tf.name_scope('Samples'):
    # shape `[n_samples, n_dims]`
    samples = inference_dist.sample(n_samples)
  with tf.name_scope('Logits'):
    # shape: `[n_samples]`
    if use_logits:
      sample_logits = logits(samples)
    else:
      sample_logits = tf.zeros([n_samples])
  with tf.name_scope('KLDivergence'):
    # shape: `[n_samples]`
    weights = tf.nn.softmax(sample_logits) * n_samples
    # shape: `[n_samples]`
    log_weights = tf.log(weights)
    # The batch-supplement may not ensured in `log_p`,
    # so we employ `tf.map_fn` for vectorization
    # shape: `[n_samples]`
    log_ps = tf.map_fn(log_p, samples)
    # Notice `tfd.Distribution.log_prob()` is batch-supplemented,
    # shape: `[n_samples]`
    log_qs = inference_dist.log_prob(samples)
    # shape: `[]`
    kl_divergence = tf.reduce_mean(
      weights * (log_weights + log_qs - log_ps),
      axis=0)
  with tf.name_scope('Penalty'):
    # shape: `[]`
    _, sample_logits_std = tf.nn.moments(sample_logits/logits_scale,
                                         axes=[0])
    # shape: `[]`
    penalty = penalty_scale * tf.nn.relu(sample_logits_std - 1.0)
  with tf.name_scope('Loss'):
    # shape: `[]`
    loss = kl_divergence + penalty
  with tf.name_scope('LogEvidence'):
    # shape: `[]`
    log_evidence = (tf.reduce_logsumexp(log_ps - log_qs, axis=0)
                    - tf.log(tf.to_float(n_samples)))
  with tf.name_scope('Errors'):
    # shape: `[n_samples]`
    #errors = log_ps - log_qs - log_weights - kl_divergence
    errors = log_ps - log_qs - log_weights - log_evidence
  with tf.name_scope('Gradients'):  # test!
    sub_err = log_ps - log_qs - log_weights
    f = log_ps - log_qs
    grad_sub_err = ...
    grad_f = ...
    grad_logsumexp_f = tf.reduce_sum(tf.exp(f - tf.reduce_max(f)) * grad_f) \
                     / tf.reduce_sum(tf.exp(f - tf.reduce_max(f)))
    grad_errors = ...
    grad_loss = tf.reduce_sum(tf.sign(errors) * grad_errors)
    # XXX
  return LossInfo(samples, sample_logits, weights, log_weights, log_ps,
                  log_qs, kl_divergence, penalty, loss, errors, None)


if __name__ == '__main__':
  """Test."""

  from tensorflow.contrib.layers import xavier_initializer
  np.random.seed(1)
  tf.set_random_seed(1)

  def get_inference_dist(n_dims):
    with tf.name_scope('InferenceDistribution'):
      loc = tf.get_variable('loc', [n_dims], 'float32')
      scale = tf.get_variable('scale', [n_dims], 'float32')
      inference_dist = tfd.Independent(
          tfd.NormalWithSoftplusScale(loc, scale))
    return inference_dist

  def get_logits(n_hiddens_list, scale):
    def logits(samples):
      """The `logits` argument of `make_loss()`."""
      with tf.name_scope('LogitsNeuralNetwork'):
        hiddens = samples
        for n_hiddens in n_hiddens_list:
          hiddens = tf.layers.dense(
              hiddens, n_hiddens, activation=tf.nn.relu,
              kernel_initializer=xavier_initializer())
          # Dropping out demages the representability of the ANN.
          #hiddens = tf.layers.dropout(hiddens)  # thus shall not use.
        # shape: `[n_samples, 1]`
        #outputs = tf.layers.dense(hiddens, 1)
        outputs = scale * tf.layers.dense(
            hiddens, 1, activation=tf.tanh,
            kernel_initializer=xavier_initializer())
        # shape: `[n_samples]`
        return tf.squeeze(outputs, axis=1)
    return logits

  def test(log_p, n_dims, n_hiddens_list, use_logits, logits_scale=1e+2,
           penalty_scale=1e+0, lr=1e-3, n_iters=10**4, max_trails=2,
           log_importance_threshold=-5, scale=10):
    inference_dist = get_inference_dist(n_dims)
    logits = get_logits(n_hiddens_list, scale)
    loss_info = make_loss(log_p, inference_dist, logits,
                          logits_scale, penalty_scale,
                          use_logits=use_logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    gvs = optimizer.compute_gradients(
        loss_info.loss, tf.trainable_variables())
    #gvs = [(tf.clip_by_value(g, -5., 5.), v) for g, v in gvs]  # clip.
    train_op = optimizer.apply_gradients(gvs)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Loss (before):', sess.run(loss_info.loss))
    print('Penalty (before):', sess.run(loss_info.penalty))

    loss_vals = []
    for step in range(1, n_iters+1):
      sess.run(train_op)
      
      # The value of loss can sometimes temporally be `NaN`, and in
      # the next `sess.run()` becomes non-`NaN` (strange!). So, we
      # employ the following strategy:
      loss_val = sess.run(loss_info.loss)
      n_trials = 0
      while np.isnan(loss_val) and n_trials < max_trails:
        loss_val = sess.run(loss_info.loss)
        n_trials += 1
      if n_trials == max_trails:
        print(sess.run([loss_info.log_weights, loss_info.log_ps,
                        loss_info.log_qs, loss_info.kl_divergence,
                        loss_info.penalty]))
        print('Always `NaN`, finally stopped at step {}.'.format(step))
        sess.close()
        return
      loss_vals.append(loss_val)
      
    print('Loss (after):', sess.run(loss_info.loss))
    print('Penalty (after):', sess.run(loss_info.penalty))
    visualize(loss_vals, loss_info, sess, log_importance_threshold)
    sess.close()

  def visualize(loss_vals, loss_info, sess,
                log_importance_threshold):
    samples = []
    log_weights = []
    logits = []
    errors = []
    for i in range(100):
      result = sess.run([loss_info.samples, loss_info.log_weights,
                         loss_info.sample_logits, loss_info.errors])
      sample_vals, log_weight_vals, logit_vals, error_vals = result
      samples += [_ for _ in sample_vals]
      log_weights += [_ for _ in log_weight_vals]
      logits += [_ for _ in logit_vals]
      errors += [_ for _ in error_vals]
    samples = np.array(samples)
    log_weights = np.array(log_weights)
    logits = np.array(logits)

    #err_mean = np.mean(errors)
    #err_std = np.std(errors)
    #errors = np.array([_ for _ in errors if abs(_ - err_mean) < 3*err_std])
    errors = np.array(errors)

    # Visualize
    fig = plt.figure()

    ax1 = fig.add_subplot(222)
    ax1.plot(loss_vals, label='loss')
    ax1.legend()

    ax2 = fig.add_subplot(221)
    ax3 = fig.add_subplot(223)
    count = 0
    for l, e, lw in zip(logits, errors, log_weights):
      if e + lw > log_importance_threshold:
        count += 1
        ax2.scatter(e, lw, c='blue', alpha=0.2)
        ax3.scatter(e, l, c='blue', alpha=0.2)
    ax2.set_ylabel('$\ln \omega$')
    ax2.set_xlabel('$\mathcal{E}$')
    ax3.set_ylabel('logits')
    print('{} important samples'.format(count))
    print('Importance ratio:', count / len(log_weights))
    
    plt.show()

  def get_log_p_1(n_dims):
    target_dist = tfd.Independent(
        tfd.NormalWithSoftplusScale(loc=tf.zeros(n_dims),
                                    scale=10*tf.ones(n_dims)))
    return target_dist.log_prob

  def get_log_p_2(n_dims):
    dist = tfd.Independent(tfd.Gamma(1.2*tf.ones([n_dims]),
                                     1.0*tf.ones([n_dims])))
    def log_p(x):
      return dist.log_prob(tf.exp(x))
    return log_p

  n_dims = 100
  log_p = get_log_p_2(n_dims)
  kwargs = {
    'use_logits': True,
    'lr': 1e-3,
    'n_hiddens_list': [10, 10, 10],
    'n_iters': 20000,
    'log_importance_threshold': -5,
    'scale': 10,
  }
  print('\n --------- \n')
  print('PARAMETERS:', kwargs, '\n')
  test(log_p, n_dims, **kwargs)
  print('\n --------- \n')