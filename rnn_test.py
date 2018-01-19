# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:46:55 2018

@author: Andrei Ionut Damian
"""



import tensorflow as tf
import numpy as np


import os

def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    logger_lib = None
    print("Logger library not found in shared repo.", flush = True)
    #raise Exception("Couldn't find google drive folder!")
  else:  
    utils_path = os.path.join(drive_path, "_pyutils")
    print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
    logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
    print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return logger_lib

class SimpleLogger:
  def __init__(self):
    return
  def VerboseLog(self, _str, show_time):
    print(_str, flush=True)

  def P(self, _str, _st):
	print(_str, flush=True)

def LoadLogger(lib_name, config_file):
  module = load_module("logger", "logger.py")
  if module is not None:
    logger = module.Logger(lib_name = lib_name, config_file = config_file)
  else:
    logger = SimpleLogger()
  return logger

def rnd_p2s(preds, i2c):
  tt = ''
  for p in preds:
    ch = i2c[np.random.choice(range(preds.shape[1]), p=p)]
    tt += ch
  return tt

if __name__=='__main__':
  text_file = __file__
  log = load_module('logger','logger.py').Logger(lib_name='RNN1')
  log.P("Loading data file '{}'".format(text_file))
  data = open(text_file).read()
  seq_len = 15
  hidden_size = 64
  epochs = 2500
  sampling_size = 250
  full_step = True
  random_sampling = True
  start_text = "    while True:"[:seq_len]
  vocab = sorted(list(set(data)))  
  vsize = len(vocab)
  chr_to_idx = {c:i for i,c in enumerate(vocab)}
  idx_to_chr = {i:c for i,c in enumerate(vocab)}
  oh_mat = np.eye(vsize)
  str_to_oh = lambda txt: oh_mat[[chr_to_idx[c] for c in txt]]
  pred_to_str = lambda pre: "".join(idx_to_chr[i] for i in np.argmax(pre, axis=1))
  
  log.P("Test: {}".format(rnd_p2s(str_to_oh(start_text), idx_to_chr)))

  
  log.P("Creating graph...")
  g = tf.Graph()
  with g.as_default():  
    tf_x_seq = tf.placeholder(dtype=tf.float32, shape=[seq_len, vsize],
                              name='x_seq')
    tf_y_seq = tf.placeholder(dtype=tf.float32, shape=[seq_len, vsize],
                              name='y_seq')
    tf_h_ini = tf.placeholder(dtype=tf.float32, shape=[1, hidden_size],
                              name='h_init')
    
    tf_wxh = tf.Variable(np.random.randn(vsize, hidden_size)*0.01, dtype=tf.float32)
    tf_why = tf.Variable(np.random.randn(hidden_size, vsize)*0.01, dtype=tf.float32)
    tf_whh = tf.Variable(np.random.randn(hidden_size, hidden_size)*0.01, dtype=tf.float32)
    
    tf_hbias = tf.Variable(np.zeros((1, hidden_size)), dtype=tf.float32)
    tf_ybias = tf.Variable(np.zeros((1, vsize)), dtype=tf.float32)
    
    tf_h = tf_h_ini
    output_list = []
    seq_list = tf.split(tf_x_seq, seq_len)
    for unroll_step, tf_x_input in enumerate(seq_list):
      tf_h = tf.add(tf.matmul(tf_x_input, tf_wxh) + tf.matmul(tf_h,tf_whh), tf_hbias,
                    name='h_'+str(unroll_step))
      tf_h = tf.nn.tanh(tf_h)
      if unroll_step==1:
        tf_h_second = tf_h
      tf_y = tf.add(tf.matmul(tf_h, tf_why), tf_ybias, name='y_'+str(unroll_step))
      output_list.append(tf_y)
    
    tf_h_out = tf_h if full_step else tf_h_second
    
    tf_y_full_seq = tf.concat(output_list, axis=0)
    tf_y_preds = tf.nn.softmax(tf_y_full_seq)
    tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf_y_seq,
        logits=tf_y_full_seq))
    
    opt = tf.train.AdamOptimizer()
    tf_train_op = opt.minimize(tf_loss)
    tf_init_op = tf.global_variables_initializer()

  start_idx = 0
  sess = tf.Session(graph=g)
  sess.run(tf_init_op)
  step_size = seq_len if full_step else 1
  text_range = range(0, len(data)-1, step_size)
  np_h_init = np.zeros((1,hidden_size))
  for epoch in range(epochs):
    log.P("Running epoch {}".format(epoch))
    h_start = np_h_init.copy()
    #tbar = tqdm(text_range)
    for start_idx in text_range:
      x_seq = data[start_idx:start_idx+seq_len]
      x_seq += ' ' * (seq_len - len(x_seq))
      y_seq = data[start_idx+1: start_idx+seq_len+1]
      y_seq += ' ' * (seq_len - len(y_seq))
      x_seq_oh = str_to_oh(x_seq)
      y_seq_oh = str_to_oh(y_seq)
      _, loss, h_start = sess.run([tf_train_op, tf_loss, tf_h_out],
                                  feed_dict={
                                      tf_x_seq:x_seq_oh,
                                      tf_y_seq:y_seq_oh,
                                      tf_h_ini:h_start
                                      })
      #tbar.set_description("Pos:{} Loss:{:.2f}".format(start_idx, loss))
    log.P("Predicting with start: '{}'".format(start_text))
    final_output = start_text
    input_text = start_text
    for s in range(sampling_size):
      x_test = str_to_oh(input_text)
      preds, h_start = sess.run([tf_y_preds, tf_h_out], feed_dict={
          tf_x_seq: x_test,
          tf_h_ini: h_start
          })
      if random_sampling:
        out_text = rnd_p2s(preds, idx_to_chr)
      else:
        out_text = pred_to_str(preds)
      ch = out_text[-1]
      input_text = input_text[1:] + ch
      final_output += ch
    log.P("Predicted:\n{}".format(final_output))
      
        
    
    
    
  