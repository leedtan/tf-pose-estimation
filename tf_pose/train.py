import matplotlib as mpl
mpl.use(
    'Agg'
)  # training mode, no screen should be open. (It will block training loop)
#python tf_pose/train.py --training_name=adv_sampling2 --checkpoint=models/cs3033/adv_sampling/model-4000
import argparse
import logging
import os
import time
import sys
src_path = os.path.join(os.getcwd(), 'tf_pose' )
sys.path.append(src_path)
INIT_TIME = time.time()
LASTTIME = time.time()
import nn_utils
import pdb
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorpack.dataflow.remote import RemoteDataZMQ

from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale
from common import get_sample_images
from networks import get_network
import os

def checktime(name=''):
  global LASTTIME
  newtime = time.time()
  print('\n\n-------------------------------------')
  print('time', name, newtime - LASTTIME, 'total time', newtime - INIT_TIME)
  LASTTIME = newtime


checktime('import time')


class UsefulLogger(object):
  def __init__(self, filename="last_run_output.txt"):
    self.terminal = sys.stdout
    self.log = open(filename, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
    self.flush()

  def flush(self):
    self.log.flush()



logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
SHRINK = 1
if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Training codes for Openpose using Tensorflow')
  parser.add_argument('--model', default='mobilenet_thin', help='model name')
  parser.add_argument('--datapath', type=str, default='../dataset/annotations/')
  parser.add_argument('--training_name', type=str, default='default_name')
  parser.add_argument('--imgpath', type=str, default='../dataset/')
  parser.add_argument('--batchsize', type=int, default=16 * SHRINK * SHRINK)
  parser.add_argument('--gpus', type=int, default=1)
  parser.add_argument('--max-epoch', type=int, default=30)
  parser.add_argument('--gpu_num', type=int, default=0)
  parser.add_argument('--freezeframe', type=int, default=0)
  parser.add_argument('--advsample', type=int, default=1)
  parser.add_argument('--lr', type=float, default=0.03)
  parser.add_argument('--modelpath', type=str, default='models/cs3033/')
  parser.add_argument('--logpath', type=str, default='logs/')
  parser.add_argument('--checkpoint', type=str, default='')
  parser.add_argument('--tag', type=str, default='')
  parser.add_argument(
      '--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')

  parser.add_argument('--input-width', type=int, default=368 // SHRINK)
  parser.add_argument('--input-height', type=int, default=368 // SHRINK)
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)
  for directory in [args.modelpath, args.logpath]:
    if not os.path.exists(directory):
      os.makedirs(directory)
  sys.stdout = UsefulLogger("logs/" + str(os.path.basename(sys.argv[0])) +
                          str(time.time()) + ".txt")
  if args.gpus <= 0:
    raise Exception('gpus <= 0')
  print(args)
  training_name = args.training_name

  # define input placeholder
  set_network_input_wh(args.input_width, args.input_height)
  scale = 4

  if args.model in [
      'cmu', 'vgg', 'mobilenet_thin', 'mobilenet_try', 'mobilenet_try2',
      'mobilenet_try3', 'hybridnet_try'
  ]:
    scale = 8

  set_network_scale(scale)
  output_w, output_h = args.input_width // scale, args.input_height // scale

  logger.info('define model+')
  with tf.device(tf.DeviceSpec(device_type="CPU")):
    input_node = tf.placeholder(
        tf.float32,
        shape=(args.batchsize, args.input_height, args.input_width, 3),
        name='image')
    vectmap_node = tf.placeholder(
        tf.float32,
        shape=(args.batchsize, output_h, output_w, 38),
        name='vectmap')
    heatmap_node = tf.placeholder(
        tf.float32,
        shape=(args.batchsize, output_h, output_w, 19),
        name='heatmap')
    checktime('defined placeholders')

    # prepare data
    if not args.remote_data:
      df = get_dataflow_batch(
          args.datapath, True, args.batchsize, img_path=args.imgpath)
    else:
      # transfer inputs from ZMQ
      raise ValueError('tried to use remote data?')
      df = RemoteDataZMQ(args.remote_data, hwm=3)
    
    # for dp in df.get_data():
    #   a = dp
    #    feed = dict(zip([input_node, heatmap_node, vectmap_node], dp))
    enqueuer = DataFlowToQueue(
        df, [input_node, heatmap_node, vectmap_node], queue_size=100)
    q_inp, q_heat, q_vect = enqueuer.dequeue()
    checktime('enqueuer defined')

  df_valid = get_dataflow_batch(
      args.datapath, False, args.batchsize, img_path=args.imgpath)
  df_valid.reset_state()
  validation_cache = []
  checktime('defined got dataflow batch')

  val_image = get_sample_images(args.input_width, args.input_height)
  logger.info('tensorboard val image: %d' % len(val_image))
  logger.info(q_inp)
  logger.info(q_heat)
  logger.info(q_vect)
  checktime('got image samples')

  # define model for multi-gpu
  q_inp_split, q_heat_split, q_vect_split = tf.split(
      q_inp, args.gpus), tf.split(q_heat, args.gpus), tf.split(
          q_vect, args.gpus)

  output_vectmap = []
  output_heatmap = []
  losses = []
  last_losses_l1 = []
  last_losses_l2 = []
  outputs = []
  for gpu_id in range(args.gpus):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
      with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
        checktime('about to get network')
        net, pretrain_path, last_layer = get_network(args.model,
                                                     q_inp_split[gpu_id])
        checktime('got network')
        vect, heat = net.loss_last()
        output_vectmap.append(vect)
        output_heatmap.append(heat)
        outputs.append(net.get_output())

        l1s, l2s = net.loss_l1_l2()
        for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
          loss_l1 = tf.nn.l2_loss(
              tf.concat(l1, axis=0) - q_vect_split[gpu_id],
              name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
          loss_l2 = tf.nn.l2_loss(
              tf.concat(l2, axis=0) - q_heat_split[gpu_id],
              name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
          losses.append(tf.reduce_mean([loss_l1, loss_l2]))

        last_losses_l1.append(loss_l1)
        last_losses_l2.append(loss_l2)
        checktime('defined losses')

  outputs = tf.concat(outputs, axis=0)

  with tf.device(tf.DeviceSpec(device_type="CPU")):
    # define loss
    total_loss = tf.reduce_mean(losses)
    total_loss_ll_paf = tf.reduce_mean(last_losses_l1)
    total_loss_ll_heat = tf.reduce_mean(last_losses_l2)
    total_loss_ll = tf.reduce_mean([total_loss_ll_paf, total_loss_ll_heat])
    # define optimizer
    step_per_epoch = 121745 // args.batchsize
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.placeholder(tf.float32, None)
  opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    checktime('about to define optimizer')
    train_op, grad_norm = nn_utils.apply_clipped_optimizer_pose(
        opt, total_loss, global_step=global_step)
    checktime('defined optimizer')
  logger.info('define model-')

  grad_norm_loss, grad_norm_loss_paf, grad_norm_loss_heat = [
    nn_utils.get_grad_norm(opt, l) for l in [total_loss, total_loss_ll_paf, total_loss_ll_heat]]

  dLdX = tf.gradients(
    total_loss,
    q_inp)[0]
  # define summary
  tf.summary.scalar("loss", total_loss)
  tf.summary.scalar("loss_lastlayer", total_loss_ll)
  tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
  tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
  tf.summary.scalar("queue_size", enqueuer.size())
  merged_summary_op = tf.summary.merge_all()

  valid_loss = tf.placeholder(tf.float32, shape=[])
  valid_loss_ll = tf.placeholder(tf.float32, shape=[])
  valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
  valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
  sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
  sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
  train_img = tf.summary.image('training sample', sample_train, 4)
  valid_img = tf.summary.image('validation sample', sample_valid, 12)
  valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
  valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
  merged_validate_op = tf.summary.merge(
      [train_img, valid_img, valid_loss_t, valid_loss_ll_t])

  saver = tf.train.Saver(max_to_keep=1000)
  config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
  with tf.Session(config=config) as sess:
    # training_name = '{}_batch{}_lr{}_gpus{}_{}x{}_{}'.format(
    #     args.model, args.batchsize, args.lr, args.gpus, args.input_width,
    #     args.input_height, args.tag)
    logger.info('model weights initialization')
    checktime('about to initialize')
    sess.run(tf.global_variables_initializer())
    checktime('initialized')

    if args.checkpoint:
      logger.info('Restore from checkpoint...')
      #loader = tf.train.Saver(net.restorable_variables())
      #loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
      checktime('about to restore')
      saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
      checktime('restored')
      logger.info('Restore from checkpoint...Done')
    if pretrain_path:
      checktime('about to restore')

      logger.info('Restore pretrained weights...')
      if '.ckpt' in pretrain_path:
        loader = tf.train.Saver(net.restorable_variables())
        loader.restore(sess, pretrain_path)
      elif '.npy' in pretrain_path:
        net.load(pretrain_path, sess, False)
      checktime('restored')
      logger.info('Restore pretrained weights...Done')

    logger.info('prepare file writer')
    file_writer = tf.summary.FileWriter(args.logpath + training_name,
                                        sess.graph)

    logger.info('prepare coordinator')
    coord = tf.train.Coordinator()
    enqueuer.set_coordinator(coord)
    enqueuer.start()

    logger.info('Training Started.')
    time_started = time.time()
    last_gs_num = last_gs_num2 = 0
    initial_gs_num = sess.run(global_step)
    gs_num = 0
    adv_idx = 0
    checktime('starting optimization')
    started = 0
    while True:
      current_lr = args.lr/np.sqrt(gs_num + 10)
      if not args.freezeframe:
        cur_inpt, cur_vectmap, cur_heatmap = sess.run([q_inp, q_heat, q_vect])
      else:
          if started:
            cur_inpt, cur_vectmap, cur_heatmap = cur_inpt, cur_vectmap, cur_heatmap
          else:
            cur_inpt, cur_vectmap, cur_heatmap = sess.run([q_inp, q_heat, q_vect])
            started = 1
      fd_raw = {learning_rate: current_lr,
        q_inp: cur_inpt,
        q_heat: cur_vectmap,
        q_vect: cur_heatmap}
      if args.advsample:
        grad, l_raw = sess.run([dLdX, total_loss], fd_raw)
        adv_idx += 1
        if adv_idx > 2:
          adv_idx = 0
        if adv_idx == 0:
          adv_impact = ((grad > 0) * 1.0 + (grad < 0) * (-1.0) ) * 1e-1
        if adv_idx == 1:
          s, g = np.sign(grad), np.abs(grad)
          grad = s * np.sqrt(g)
        if adv_idx in [1, 2]:  
          norm = np.sqrt(np.square(grad).mean())
          adv_impact = (grad / norm) * 3e-2
        cur_inpt_adv = cur_inpt + adv_impact
        fd_adv = {learning_rate: current_lr,
          q_inp: cur_inpt_adv,
          q_heat: cur_vectmap,
          q_vect: cur_heatmap}
        _, gs_num, l_post = sess.run([train_op, global_step, total_loss], fd_adv)
        print(adv_idx, l_raw, l_post - l_raw)
      else:
        _, gs_num, l_post = sess.run([train_op, global_step, total_loss], fd_raw)
      if gs_num > step_per_epoch * args.max_epoch:
        break

      if gs_num == 2:
        train_loss, train_loss_ll, train_loss_ll_paf, \
          train_loss_ll_heat, summary, queue_size = sess.run(
            [
                total_loss, total_loss_ll, total_loss_ll_paf,
                total_loss_ll_heat, merged_summary_op,
                enqueuer.size()
            ])
        checktime('2 optimizations')
        batch_per_sec = (gs_num - initial_gs_num) / (
            time.time() - time_started)
        logger.info(
            'epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g, q=%d'
            % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize,
               current_lr, train_loss, train_loss_ll, train_loss_ll_paf,
               train_loss_ll_heat, queue_size))
        last_gs_num = gs_num

        file_writer.add_summary(summary, gs_num)
      #pdb.set_trace()

      if gs_num - last_gs_num >= 100 or (gs_num == 5):
        train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, summary, queue_size = sess.run(
            [
                total_loss, total_loss_ll, total_loss_ll_paf,
                total_loss_ll_heat, merged_summary_op,
                enqueuer.size()
            ])
        if gs_num < 10:
          checktime('5 optimizations')

        # log of training loss / accuracy
        batch_per_sec = (gs_num - initial_gs_num) / (
            time.time() - time_started)
        logger.info(
            'epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g, q=%d'
            % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize,
               current_lr, train_loss, train_loss_ll, train_loss_ll_paf,
               train_loss_ll_heat, queue_size))
        last_gs_num = gs_num
        if gs_num % 4 == 0:
            cur_grad = sess.run(
                grad_norm)
            file_writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag='Grad Norm Total', simple_value=cur_grad)]),
            gs_num)
        if gs_num % 4 == 1:
            cur_grad_loss = sess.run(
                grad_norm_loss)
            file_writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag='Grad Norm Loss', simple_value=cur_grad_loss)]),
            gs_num)
        if gs_num % 4 == 2:
            cur_grad_paf = sess.run(
                grad_norm_loss_heat)
            file_writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag='Grad Norm PAF', simple_value=cur_grad_paf)]),
            gs_num)
        if gs_num % 4 == 3:
            cur_grad_heat = sess.run(
                grad_norm_loss_paf)
            file_writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag='Grad Norm Heat', simple_value=cur_grad_heat)]),
            gs_num)
        file_writer.add_summary(summary, gs_num)

      if gs_num - last_gs_num2 >= 1000:
        # save weights
        directory = os.path.join(args.modelpath, training_name)
        if not os.path.exists(directory):
          os.makedirs(directory)
        saver.save(
            sess,
            os.path.join(args.modelpath, training_name, 'model'),
            global_step=global_step)

        average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat = 0
        total_cnt = 0

        if len(validation_cache) == 0:
          for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
            validation_cache.append((images_test, heatmaps, vectmaps))
          df_valid.reset_state()
          del df_valid
          df_valid = None

        # log of test accuracy
        for images_test, heatmaps, vectmaps in validation_cache:
          lss, lss_ll, lss_ll_paf, lss_ll_heat, vectmap_sample, heatmap_sample = sess.run(
              [
                  total_loss, total_loss_ll, total_loss_ll_paf,
                  total_loss_ll_heat, output_vectmap, output_heatmap
              ],
              feed_dict={
                  q_inp: images_test,
                  q_vect: vectmaps,
                  q_heat: heatmaps
              })
          average_loss += lss * len(images_test)
          average_loss_ll += lss_ll * len(images_test)
          average_loss_ll_paf += lss_ll_paf * len(images_test)
          average_loss_ll_heat += lss_ll_heat * len(images_test)
          total_cnt += len(images_test)

        logger.info(
            'validation(%d) %s loss=%f, loss_ll=%f, loss_ll_paf=%f, loss_ll_heat=%f'
            % (total_cnt, training_name, average_loss / total_cnt,
               average_loss_ll / total_cnt, average_loss_ll_paf / total_cnt,
               average_loss_ll_heat / total_cnt))
        last_gs_num2 = gs_num

        sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
        outputMat = sess.run(
            outputs,
            feed_dict={
                q_inp:
                np.array((sample_image + val_image) * (args.batchsize // 16))
            })
        pafMat, heatMat = outputMat[:, :, :, 19:], outputMat[:, :, :, :19]

        sample_results = []
        for i in range(len(sample_image)):
          test_result = CocoPose.display_image(
              sample_image[i], heatMat[i], pafMat[i], as_numpy=True)
          test_result = cv2.resize(test_result, (640, 640))
          test_result = test_result.reshape([640, 640, 3]).astype(float)
          sample_results.append(test_result)

        test_results = []
        for i in range(len(val_image)):
          test_result = CocoPose.display_image(
              val_image[i],
              heatMat[len(sample_image) + i],
              pafMat[len(sample_image) + i],
              as_numpy=True)
          test_result = cv2.resize(test_result, (640, 640))
          test_result = test_result.reshape([640, 640, 3]).astype(float)
          test_results.append(test_result)

        # save summary
        summary = sess.run(
            merged_validate_op,
            feed_dict={
                valid_loss: average_loss / total_cnt,
                valid_loss_ll: average_loss_ll / total_cnt,
                valid_loss_ll_paf: average_loss_ll_paf / total_cnt,
                valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                sample_valid: test_results,
                sample_train: sample_results
            })
        file_writer.add_summary(summary, gs_num)

    saver.save(
        sess,
        os.path.join(args.modelpath, training_name, 'model'),
        global_step=global_step)
  logger.info('optimization finished. %f' % (time.time() - time_started))
