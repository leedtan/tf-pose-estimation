import tensorflow as tf
import numpy as np  # noqa


def dense_scaled(prev_layer, layer_size, name=None, reuse=False, scale=1.0):
  output = tf.layers.dense(prev_layer, layer_size, reuse=reuse) * scale
  return output


def dense_relu(dense_input, layer_size, scale=1.0):
  dense = dense_scaled(dense_input, layer_size, scale=scale)
  output = tf.nn.leaky_relu(dense)

  return output


# gradients


def get_grad_norm(opt_fcn, loss):
  gvs = opt_fcn.compute_gradients(loss)
  grad_norm = tf.sqrt(
      tf.reduce_sum([
          tf.reduce_sum(tf.square(grad)) for grad, var in gvs
          if grad is not None
      ]))
  return grad_norm


def apply_clipped_optimizer(opt_fcn,
                            loss,
                            clip_norm=.1,
                            clip_single=.03,
                            clip_global_norm=False):
  gvs = opt_fcn.compute_gradients(loss)

  if clip_global_norm:
    gs, vs = zip(*[(g, v) for g, v in gvs if g is not None])
    capped_gs, grad_norm_total = tf.clip_by_global_norm([g for g in gs],
                                                        clip_norm)
    capped_gvs = list(zip(capped_gs, vs))
  else:
    grad_norm_total = tf.sqrt(
        tf.reduce_sum([
            tf.reduce_sum(tf.square(grad)) for grad, var in gvs
            if grad is not None
        ]))
    capped_gvs = [(tf.clip_by_value(grad, -1 * clip_single, clip_single), var)
                  for grad, var in gvs if grad is not None]
    capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var)
                  for grad, var in capped_gvs if grad is not None]

  optimizer = opt_fcn.apply_gradients(
      capped_gvs, global_step=tf.train.get_global_step())

  return optimizer, grad_norm_total

def apply_clipped_optimizer_pose(opt_fcn,
                            loss,
                            clip_norm=.1,
                            clip_single=.03,
                            clip_global_norm=False,
                            global_step = None):
  gvs = opt_fcn.compute_gradients(loss)

  if clip_global_norm:
    gs, vs = zip(*[(g, v) for g, v in gvs if g is not None])
    capped_gs, grad_norm_total = tf.clip_by_global_norm([g for g in gs],
                                                        clip_norm)
    capped_gvs = list(zip(capped_gs, vs))
  else:
    grad_norm_total = tf.sqrt(
        tf.reduce_sum([
            tf.reduce_sum(tf.square(grad)) for grad, var in gvs
            if grad is not None
        ]))
    capped_gvs = [(tf.clip_by_value(grad, -1 * clip_single, clip_single), var)
                  for grad, var in gvs if grad is not None]
    capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var)
                  for grad, var in capped_gvs if grad is not None]

  optimizer = opt_fcn.apply_gradients(
      capped_gvs, global_step=global_step)

  return optimizer, grad_norm_total


def locally_connected_mlp(x,
                          hidden_sizes,
                          kernel_size=3,
                          output_size=None,
                          name='',
                          reuse=False,
                          scale=1.,
                          init_scale=1.,
                          kernel_size_output=None):
  prev_layer = x
  if kernel_size_output is None:
    kernel_size_output = kernel_size
  for idx, l in enumerate(hidden_sizes):
    prev_layer = locally_connected(
        prev_layer,
        l,
        kernel_size=kernel_size,
        name='lc_mlp' + name + '_' + str(idx))
    prev_layer = tf.nn.leaky_relu(prev_layer) * scale

  output = prev_layer

  if output_size is not None:
    output = locally_connected(
        prev_layer,
        output_size,
        kernel_size=kernel_size_output,
        name='lc_mlp' + name + 'final')

  return output


def mlp(x, hidden_sizes, output_size=None, name='', scale=1., reuse=False):
  prev_layer = x

  for idx, l in enumerate(hidden_sizes):
    dense = dense_scaled(
        prev_layer, l, name='mlp' + name + '_' + str(idx), scale=scale)
    prev_layer = tf.nn.leaky_relu(dense)

  output = prev_layer

  if output_size is not None:
    output = dense_scaled(
        prev_layer, output_size, name='mlp' + name + 'final', scale=scale)

  return output


def local_deconv(inputs, filters, kernel_size=3):
  kernel_size = 3
  paddings = [[0, 0], [0, 1], [0, 0]]
  spacial_input = inputs.get_shape()[1].value
  filters_input = inputs.get_shape()[2].value
  zeros = tf.zeros_like(inputs)
  stacked = tf.stack([zeros, inputs], 2)
  padded_inners = tf.reshape(stacked, (-1, 2 * spacial_input, filters_input))
  padded = tf.pad(
      padded_inners, paddings, mode='CONSTANT', name=None, constant_values=0)
  conv = tf.keras.layers.LocallyConnected1D(
      filters,
      kernel_size,
  )(padded)
  return conv


def locally_connected(inputs,
                      filters,
                      kernel_size,
                      strides=1,
                      padding='same',
                      residual=False,
                      name=None,
                      use_bias=True):
  #batch, kernel, filters
  # Hard coding because currently only supports 1d conv
  if isinstance(kernel_size, list):
    kernel_size = kernel_size[0]
  if isinstance(inputs, list):
    inputs = tf.concat(inputs, -1)
  if kernel_size > 1 and padding == 'same':
    if kernel_size == 2:
      paddings = [[0, 0], [0, 1], [0, 0]]
    if kernel_size == 3:
      paddings = [[0, 0], [1, 1], [0, 0]]
    if kernel_size == 5:
      paddings = [[0, 0], [2, 2], [0, 0]]
    padded = tf.pad(
        inputs, paddings, mode='CONSTANT', name=None, constant_values=0)
  else:  # Kerne size 1 or padding == 'valid'
    padded = inputs
  conv = tf.keras.layers.LocallyConnected1D(
      filters,
      kernel_size,
      strides=strides,
      use_bias=use_bias,
  )(padded)
  if residual:
    conv = conv + inputs
  return tf.identity(conv, name=name)


def pad(inputs, kernel_size, padding, num_dims):
  if num_dims == 3:

    if kernel_size > 1 and padding == 'same':
      if kernel_size == 2:
        paddings = [[0, 0], [0, 1], [0, 0]]
      if kernel_size == 3:
        paddings = [[0, 0], [1, 1], [0, 0]]
      if kernel_size == 5:
        paddings = [[0, 0], [2, 2], [0, 0]]
      padded = tf.pad(
          inputs, paddings, mode='CONSTANT', name=None, constant_values=0)
    else:  # Kerne size 1 or padding == 'valid'
      padded = inputs
    return padded
  else:
    raise NotImplementedError('pad only build for 3 dimensional objects')


def locally_connected_residual_BN(inputs,
                                  filters,
                                  kernel_size,
                                  strides=1,
                                  is_training=False,
                                  padding='same',
                                  residual=False,
                                  name=None,
                                  use_bias=True):
  # Hard coding because currently only supports 1d conv
  if isinstance(kernel_size, list):
    kernel_size = kernel_size[0]
  if isinstance(inputs, list):
    inputs = tf.concat(inputs, -1)
  padded = pad(inputs, kernel_size, padding, 3)
  conv = tf.keras.layers.LocallyConnected1D(
      filters,
      kernel_size,
      strides=strides,
      use_bias=use_bias,
  )(padded)
  conv = tf.contrib.layers.batch_norm(
      conv,
      decay=0.98,
      center=True,
      scale=False,
      is_training=is_training,
  )
  conv = tf.nn.leaky_relu(conv)
  padded = pad(conv, kernel_size, padding, 3)
  conv = tf.keras.layers.LocallyConnected1D(
      filters,
      kernel_size,
      strides=strides,
      use_bias=use_bias,
  )(padded)
  conv = tf.contrib.layers.batch_norm(
      conv,
      decay=0.999,
      center=True,
      scale=False,
      is_training=is_training,
  )
  if residual:
    conv = conv + inputs
  return tf.identity(conv, name=name)


def locally_connected_residual_drop(inputs,
                                    filters,
                                    kernel_size,
                                    strides=1,
                                    is_training=False,
                                    padding='same',
                                    name=None,
                                    use_bias=True):
  # Hard coding because currently only supports 1d conv
  if isinstance(kernel_size, list):
    kernel_size = kernel_size[0]
  if isinstance(inputs, list):
    inputs = tf.concat(inputs, -1)
  padded = pad(inputs, kernel_size, padding, 3)
  conv = tf.keras.layers.LocallyConnected1D(
      filters,
      kernel_size,
      strides=strides,
      use_bias=use_bias,
  )(padded)
  conv = tf.layers.dropout(conv, rate=.2, training=is_training)
  conv = tf.nn.leaky_relu(conv)
  padded = pad(conv, kernel_size, padding, 3)
  conv = tf.keras.layers.LocallyConnected1D(
      filters,
      kernel_size,
      strides=strides,
      use_bias=use_bias,
  )(padded)
  '''
    conv = tf.layers.dropout(
        conv,
        rate=.2,
        training=is_training)
    '''
  conv = conv + inputs
  # conv = tf.nn.leaky_relu(conv, alpha=.5)
  return tf.identity(conv, name=name)


def smart_conv(inputs,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               residual=False):
  cnn = tf.layers.conv1d(
      inputs,
      filters=filters // 2,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding)
  lcn = locally_connected(
      inputs,
      filters=filters // 2,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding)
  cnn_lcn = tf.concat((cnn, lcn), 2)
  if residual and inputs.shape[1] == cnn_lcn.shape[1] and inputs.shape[2] == cnn_lcn.shape[2]:
    cnn_lcn = cnn_lcn + inputs
  return cnn_lcn


def dynamic_rnn(x, num_layers, cells, initial_states, name=''):
  prev_layer = x
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope='dynamicrnn' + str(idx))
    returncells.append(c)
    hiddenlayers.append(prev_layer)
    # prev_layer = tf.nn.leaky_relu(prev_layer)
  return prev_layer, returncells, hiddenlayers


def dynamic_cascade_rnn(x, num_layers, cells, initial_states, name=''):
  prev_layer = x
  returncells = []
  for idx in range(num_layers):
    next_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope='dynamicrnn' + str(idx))
    returncells.append(c)
    activation = tf.nn.leaky_relu(next_layer)
    prev_layer = tf.concat((prev_layer, activation), 2)
  return next_layer, returncells


def dynamic_conv_rnn(x, num_layers, cells, initial_states, name=''):
  prev_layer = x
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'dynamicrnn' + str(idx))
    returncells.append(c)
    prev_layer = tf.nn.leaky_relu(prev_layer)
    hiddenlayers.append(prev_layer)
  return prev_layer, returncells, hiddenlayers


def allocating_rnn(x,
                   num_layers,
                   cells,
                   initial_states,
                   num_predict_periods,
                   hidden_filters,
                   tf_batch_size,
                   name=''):
  def get_allocations_predictions_weights(layer,
                                          idx,
                                          num_predict_periods,
                                          hidden_filters,
                                          tf_batch_size,
                                          shortcut=True):

    input_dense = tf.reshape(layer, (-1, num_predict_periods, hidden_filters))
    # input dense is batch * time, predict_periods, filters
    output = locally_connected_mlp(
        input_dense, [
            32,
        ],
        kernel_size=3,
        output_size=32,
        scale=30.,
        name='lc_mlp_out' + str(idx) + '_')
    # batch * time, predict_periods, 3
    allocations_predictions_weights = tf.reshape(
        output, (tf_batch_size, -1, num_predict_periods, 32))
    if shortcut:
      delayed = allocations_predictions_weights[:, :-1, :, :]
      paddings = [[0, 0], [1, 0], [0, 0], [0, 0]]
      delayed_padded = tf.pad(
          delayed, paddings, mode='CONSTANT', name=None, constant_values=0)
      allocations_predictions_weights = tf.concat(
          (delayed_padded, allocations_predictions_weights), -1)
    return allocations_predictions_weights

  prev_layer = x
  shortcut = x
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'allocating_rnn' + str(idx))
    prev_layer = tf.nn.leaky_relu(prev_layer) * 10
    returncells.append(c)
    allocations_predictions_weights = get_allocations_predictions_weights(
        prev_layer, idx, num_predict_periods, hidden_filters, tf_batch_size)
    prev_layer = tf.concat((prev_layer, allocations_predictions_weights), -1)
    hiddenlayers.append(prev_layer)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers
    prev_layer = tf.concat((prev_layer, shortcut), 3)


def local_predicting_residual_rnn(x,
                                  num_layers,
                                  cells,
                                  initial_states,
                                  num_predict_periods,
                                  hidden_filters,
                                  tf_batch_size,
                                  hidden_output=32,
                                  num_outputs=16,
                                  name=''):
  def get_allocations_predictions_weights(layer,
                                          idx,
                                          num_predict_periods,
                                          hidden_filters,
                                          tf_batch_size,
                                          hidden_output,
                                          num_outputs,
                                          shortcut=True):

    input_dense = tf.reshape(layer, (-1, num_predict_periods, hidden_filters))
    # input dense is batch * time, predict_periods, filters
    output = locally_connected_mlp(
        input_dense, [
            hidden_output,
        ],
        kernel_size=3,
        output_size=num_outputs,
        scale=1.,
        name='lc_mlp_out' + str(idx) + '_')
    # batch * time, predict_periods, 3
    allocations_predictions_weights = tf.reshape(
        output, (tf_batch_size, -1, num_predict_periods, num_outputs))
    if shortcut:
      delayed = allocations_predictions_weights[:, :-1, :, :]
      paddings = [[0, 0], [1, 0], [0, 0], [0, 0]]
      delayed_padded = tf.pad(
          delayed, paddings, mode='CONSTANT', name=None, constant_values=0)
      allocations_predictions_weights = tf.concat(
          (delayed_padded, allocations_predictions_weights), -1)
    return allocations_predictions_weights

  prev_layer = x
  shortcut = x
  last_layer = None
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'lcl_res_rnn' + str(idx))
    prev_layer = tf.nn.leaky_relu(prev_layer) * 10
    returncells.append(c)
    allocations_predictions_weights = get_allocations_predictions_weights(
        prev_layer, idx, num_predict_periods, hidden_filters, tf_batch_size,
        hidden_output, num_outputs)
    prev_layer = tf.concat((prev_layer, allocations_predictions_weights), -1)
    if idx > 0:
      prev_layer = prev_layer + last_layer
    last_layer = prev_layer
    hiddenlayers.append(prev_layer)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers
    prev_layer = tf.concat((prev_layer, shortcut), 3)


def local_shortcut_rnn(x,
                       num_layers,
                       cells,
                       initial_states,
                       num_predict_periods=None,
                       hidden_filters=None,
                       tf_batch_size=None,
                       name=''):

  prev_layer = x
  shortcut = x
  last_layer = None
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'lcl_res_rnn' + str(idx))
    prev_layer = tf.nn.leaky_relu(prev_layer) * 10
    returncells.append(c)
    if 0:  # idx > 0:
      prev_layer = prev_layer + last_layer
    last_layer = prev_layer
    hiddenlayers.append(prev_layer)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers
    prev_layer = tf.concat((prev_layer, shortcut), 3)


def local_self_attention(x, max_history, name=''):
  #x: bs, time, pp, filters
  shape = tf.shape(x)
  bs, time, pp, filters = [shape[idx] for idx in range(4)]
  pp = x.shape[2]
  filters = x.shape[3]
  x_compressed = tf.reshape(x, (bs * time, pp, filters))

  gaussianP = locally_connected_mlp(
      x=x_compressed,
      hidden_sizes=[128],
      kernel_size=3,
      output_size=2,
      kernel_size_output=1)
  means_raw, stds_raw = [gaussianP[:, :, idx] for idx in range(2)]
  means_compressed, stds_compressed = [
      tf.sigmoid(var) * max_history for var in [means_raw, stds_raw]
  ]
  stds_compressed = stds_compressed * 2
  #bs, attention head, 1 (attention tail), pp, filters (removed)
  means = -tf.reshape(means_compressed, (bs, time, 1, pp))
  stds = tf.reshape(stds_compressed, (bs, time, 1, pp)) + 1e-4
  positions = tf.cast(tf.range(time), tf.float32)
  #bs, 1 (attention head), time (attention tail), pp, filters (removed)
  positions_tail = tf.reshape(positions, (1, 1, time, 1))
  positions_head = tf.reshape(positions, (1, time, 1, 1))
  position_offset = positions_tail - positions_head
  gaussian_values_raw = tf.exp(-.5 * tf.square(
      (position_offset - means) / stds))
  #bs, attention head, attention tail (head - 30: head), pp, filters (removed)

  diag = tf.eye(time, num_columns=time + max_history)
  mask_compressed = tf.zeros((time, time))
  for idx in range(max_history):
    new_addition = diag[:, idx:time + idx]
    mask_compressed += new_addition

  mask_expanded = tf.reshape(mask_compressed, (1, time, time, 1))

  #mask_expanded = tf.transpose(mask_expanded, (0, 2, 1, 3))

  gaussian_values_masked = gaussian_values_raw * mask_expanded
  gaussian_values_denominator = tf.expand_dims(
      tf.reduce_sum(gaussian_values_masked, 2), 2) + 1e-4
  gaussian_values_normalized = gaussian_values_masked / gaussian_values_denominator

  #add back in filter dimension
  gaussian_values_expanded = tf.expand_dims(gaussian_values_normalized, -1)
  #x: bs, time, pp, filters
  #add in attention head dimension:
  x_for_attention = tf.expand_dims(x, 1)
  #bs, time head, time tail, pp, filters
  attention_contributions = gaussian_values_expanded * x_for_attention

  #bs, time head, pp, filters
  attention_values = tf.reduce_sum(attention_contributions, 2)

  return attention_values, gaussian_values_raw, gaussian_values_masked, gaussian_values_denominator, \
    means_compressed, stds_compressed


def local_shortcut_attention_rnn(x,
                                 num_layers,
                                 cells,
                                 initial_states,
                                 num_predict_periods=None,
                                 hidden_filters=None,
                                 tf_batch_size=None,
                                 name=''):
  attention_layers, gaussian_values_raw, gaussian_values_masked, gaussian_values_denominator = [], [], [], []
  gaus_means, gaus_stds = [], []
  prev_layer = x
  shortcut = x
  last_layer = None
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'lcl_res_rnn' + str(idx))
    prev_layer = tf.nn.leaky_relu(prev_layer) * 10
    returncells.append(c)
    if 0:  # idx > 0:
      prev_layer = prev_layer + last_layer
    last_layer = prev_layer
    hiddenlayers.append(prev_layer)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers, attention_layers, gaussian_values_raw, \
        gaussian_values_masked, gaussian_values_denominator, tf.stack(gaus_means, 0), tf.stack(gaus_stds, 0)
    prev_layer = tf.concat((prev_layer, shortcut), 3)
    attention_layer, gaussian_values_raw1, gaussian_values_masked1, gaussian_values_denominator1, gaus_means1, gaus_stds1 = local_self_attention(
        prev_layer, max_history=10, name='attn_layer' + str(idx))
    attention_layers.append(attention_layer)
    gaussian_values_raw.append(gaussian_values_raw1)
    gaussian_values_masked.append(gaussian_values_masked1)
    gaussian_values_denominator.append(gaussian_values_denominator1)
    gaus_means.append(gaus_means1)
    gaus_stds.append(gaus_stds1)
    prev_layer = tf.concat((prev_layer, attention_layer), 3)


def local_residual_rnn(x,
                       num_layers,
                       cells,
                       initial_states,
                       num_predict_periods=None,
                       hidden_filters=None,
                       tf_batch_size=None,
                       name=''):

  prev_layer = x
  shortcut = x
  last_layer = None
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'lcl_res_rnn' + str(idx))
    prev_layer = tf.nn.leaky_relu(prev_layer) * 10
    returncells.append(c)
    if idx > 0:
      prev_layer = prev_layer + last_layer
    last_layer = prev_layer
    hiddenlayers.append(prev_layer)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers
    prev_layer = tf.concat((prev_layer, shortcut), 3)


def allocating_residual_rnn(x,
                            num_layers,
                            cells,
                            initial_states,
                            num_predict_periods,
                            hidden_filters,
                            tf_batch_size,
                            name=''):
  def get_allocations_predictions_weights(layer,
                                          idx,
                                          num_predict_periods,
                                          hidden_filters,
                                          tf_batch_size,
                                          shortcut=True):

    input_dense = tf.reshape(layer, (-1, num_predict_periods, hidden_filters))
    # input dense is batch * time, predict_periods, filters
    output = locally_connected_mlp(
        input_dense, [
            32,
        ],
        kernel_size=3,
        output_size=32,
        scale=10.,
        name='lc_mlp_out' + str(idx) + '_')
    # batch * time, predict_periods, 3
    allocations_predictions_weights = tf.reshape(
        output, (tf_batch_size, -1, num_predict_periods, 32))
    if shortcut:
      delayed = allocations_predictions_weights[:, :-1, :, :]
      paddings = [[0, 0], [1, 0], [0, 0], [0, 0]]
      delayed_padded = tf.pad(
          delayed, paddings, mode='CONSTANT', name=None, constant_values=0)
      allocations_predictions_weights = tf.concat(
          (delayed_padded, allocations_predictions_weights), -1)
    return allocations_predictions_weights

  prev_layer = x
  shortcut = x
  last_layer = None
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'allocating_rnn' + str(idx))
    prev_layer = tf.nn.leaky_relu(prev_layer) * 10
    returncells.append(c)
    allocations_predictions_weights = get_allocations_predictions_weights(
        prev_layer, idx, num_predict_periods, hidden_filters, tf_batch_size)
    prev_layer = tf.concat((prev_layer, allocations_predictions_weights), -1)
    if idx > 0:
      prev_layer = prev_layer + last_layer
    last_layer = prev_layer
    hiddenlayers.append(prev_layer)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers
    prev_layer = tf.concat((prev_layer, shortcut), 3)


def dynamic_shortcut_conv_rnn(x, num_layers, cells, initial_states, name=''):
  prev_layer = x
  shortcut = x
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    prev_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'dynamic_shortcut_conv_rnn' + str(idx))
    prev_layer = tf.nn.leaky_relu(prev_layer) * 10
    returncells.append(c)
    hiddenlayers.append(prev_layer)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers
    prev_layer = tf.concat((prev_layer, shortcut), 3)


def dynamic_shortcut_rnn(x, num_layers, cells, initial_states, name=''):
  prev_layer = x
  shortcut = x
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    next_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'dynamicrnn' + str(idx))
    prev_layer = tf.nn.leaky_relu(next_layer)
    hiddenlayers.append(prev_layer)
    returncells.append(c)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers
    prev_layer = tf.concat((prev_layer, shortcut), 2)


def dynamic_shortcut_residual_rnn(x,
                                  num_layers,
                                  cells,
                                  initial_states,
                                  name=''):
  prev_layer_with_shortcut = x
  shortcut = x
  hiddenlayers = []
  returncells = []
  for idx in range(num_layers):
    next_layer, c = tf.nn.dynamic_rnn(
        cell=cells[idx],
        inputs=prev_layer_with_shortcut,
        initial_state=initial_states[idx],
        dtype=tf.float32,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=name + 'dynamicrnn' + str(idx))
    if idx == 0:
      prev_layer = next_layer
    else:
      prev_layer = prev_layer + next_layer
    prev_layer = tf.nn.leaky_relu(next_layer)
    hiddenlayers.append(prev_layer)
    returncells.append(c)
    if idx == num_layers - 1:
      return prev_layer, returncells, hiddenlayers
    prev_layer_with_shortcut = tf.concat((prev_layer, shortcut), 2)


def CascadeNet(x, hiddens, output_size, name='CascadeNet', reuse=False):
  prev_layer = x
  for idx, l in enumerate(hiddens):
    next_layer = tf.nn.leaky_relu(
        dense_scaled(prev_layer, l, name=name + '_' + str(idx), reuse=reuse))
    prev_layer = tf.concat((prev_layer, next_layer), 1)
  output = dense_scaled(
      prev_layer, output_size, name=name + 'final', reuse=reuse)
  return output
