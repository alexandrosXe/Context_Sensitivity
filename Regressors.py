import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
from tqdm import tqdm


METRICS = [
      keras.metrics.MeanSquaredError(name="MSE"),
      keras.metrics.MeanAbsoluteError(name="MAE"),
      keras.metrics.MeanSquaredLogarithmicError(name="MSLE"),
]


class BERT_MLP():

  def __init__(self,
                 bert_config = BertConfig(),
                 trainable_layers=3,
                 max_seq_length=128,
                 show_summary=False,
                 patience=3,
                 epochs=10,
                 save_predictions=False,
                 batch_size=32,
                 DATA_COLUMN="text",
                 TARGET_COLUMN="target",
                 DATA2_COLUMN=None,
                 lr=2e-05,
                 session=None,
                 dense_activation = None,
                 loss='MSE',
                 monitor_loss = loss,
                 monitor_mode = 'min'
                 ):
        self.bert_config = bert_config
        self.session = session
          # tf.compat.v1.set_random_seed(seed)
          # np.random.seed(seed)
        self.name = f'{"OOC1" if not DATA2_COLUMN else "OOC2"}-b{batch_size}.e{epochs}.len{max_seq_length}.bert'
          
          #bert_model = TFBertModel.from_pretrained("bert-base-cased")  # Automatically loads the config
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False,  max_length=max_seq_length,pad_to_max_length=True)

        self.lr = lr
        self.batch_size = batch_size
        self.DATA_COLUMN=DATA_COLUMN
        self.DATA2_COLUMN=DATA2_COLUMN
        self.TARGET_COLUMN=TARGET_COLUMN
        self.trainable_layers = trainable_layers
        self.max_seq_length = max_seq_length
        self.show_summary = show_summary
        self.patience=patience
        self.save_predictions = save_predictions
        self.epochs = epochs
        self.loss = loss
        self.monitor_loss = monitor_loss
        self.monitor_mode = monitor_mode
        self.dense_activation = dense_activation
        self.earlystop = tf.keras.callbacks.EarlyStopping(monitor='self.monitor_loss',
                                                            patience=self.patience,
                                                            verbose=1,
                                                            restore_best_weights=True,
                                                            mode=self.monitor_mode)
        self.BERT = TFBertModel.from_pretrained("bert-base-cased", output_attentions = True) #, config=self.bert_config)
        
  #prepare inputs for bert 
  def to_bert_input(self, sentences, text_to_encode='text'):
      input_ids, input_masks, input_segments = [],[],[]

      if self.DATA2_COLUMN==None:
        if text_to_encode == 'text':
          sentences = sentences.text
        elif text_to_encode == 'parent':
          sentences = sentences.parent
        else:
          print("Text to encode must be text (target) or parent")
        for sentence in tqdm(sentences):
            inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_seq_length, pad_to_max_length=True, 
                                                  return_attention_mask=True, return_token_type_ids=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])        
            
      elif self.DATA2_COLUMN=='parent': #use CA-SEP-BERT
        for i in range(sentences.shape[0]):
          inputs = self.tokenizer.encode_plus(sentences.iloc[i].parent,sentences.iloc[i].text, add_special_tokens=True, max_length=self.max_seq_length, pad_to_max_length=True, 
                                                  return_attention_mask=True, return_token_type_ids=True)
          input_ids.append(inputs['input_ids'])
          input_masks.append(inputs['attention_mask'])
          input_segments.append(inputs['token_type_ids'])
      return (np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32'))
      


  def unfreeze_last_k_layers(self,k=3):
    counter = 0
    for layer in self.BERT.layers:
      for l in layer.encoder.layer:
        if counter == len(layer.encoder.layer)-3:
          return
        else:
          counter+=1
          l.trainable = False

  def build(self, bias=0):
        #unfreeze last 3 layers of bert
        self.unfreeze_last_k_layers(3) 

        #take the 'CLS' token of the target
        in_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_ids", dtype='int32')
        in_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_masks", dtype='int32')
        in_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="segment_ids", dtype='int32')
        bert_inputs = [in_id, in_mask, in_segment]

        bert_output = self.BERT(bert_inputs).last_hidden_state
        bert_output = bert_output[:,0,:] #take only the embedding of the CLS token

        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='tanh')(bert_output)
        #x = tf.keras.layers.Dropout(0.1)(x)
        pred = tf.keras.layers.Dense(1, activation=self.dense_activation, bias_initializer=tf.keras.initializers.Constant(bias))(x)
        self.model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        self.model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=METRICS)
        if self.show_summary:
            self.model.summary()

  def fit(self, train, dev, bert_weights=None, class_weights={0: 1, 1: 1}, pretrained_embeddings=None):
      
        #encode target
        train_input = self.to_bert_input(train, text_to_encode="text")
        dev_input = self.to_bert_input(dev, text_to_encode="text")
        
        train_targets = train.target
        dev_targets = dev.target

        self.build()
        if bert_weights is not None:
            self.model.load_weights(bert_weights)
        self.model.fit(train_input,
                       train_targets,
                       validation_data=(dev_input, dev_targets),
                       epochs=self.epochs,
                       callbacks=[self.earlystop],
                       batch_size=self.batch_size,
                       class_weight=None 
                       )

  def predict(self, val_pd):

        #encode target
        val_input = self.to_bert_input(val_pd, text_to_encode="text")
        predictions = self.model.predict(val_input)
        
        print('Stopped epoch: ', self.earlystop.stopped_epoch)
        if self.save_predictions:
            self.save_evaluation_set(val_targets, predictions)
        return predictions

  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)
        
        
 #A Bert model encodes the parent post
class PcT_BERT():

  def __init__(self, bert_config = BertConfig(),
               trainable_layers=3,
               max_seq_length=128,
               show_summary=False,
               patience=3,
               epochs=10,
               save_predictions=False,
               batch_size=32,
               DATA_COLUMN="text",
               TARGET_COLUMN="target",
               DATA2_COLUMN=None,
               lr=2e-05,
               session=None,
               dense_activation = None,
               loss='MSE',
               monitor_loss = loss,
               monitor_mode = 'min'
               
               ):
        self.bert_config = bert_config
        self.session = session
        self.name = f'{"OOC1" if not DATA2_COLUMN else "OOC2"}-b{batch_size}.e{epochs}.len{max_seq_length}.bert'
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False,  max_length=max_seq_length,pad_to_max_length=True)
        self.lr = lr
        self.batch_size = batch_size
        self.DATA_COLUMN=DATA_COLUMN
        self.DATA2_COLUMN=DATA2_COLUMN
        self.TARGET_COLUMN=TARGET_COLUMN
        self.trainable_layers = trainable_layers
        self.max_seq_length = max_seq_length
        self.show_summary = show_summary
        self.patience=patience
        self.save_predictions = save_predictions
        self.epochs = epochs
        self.loss = loss
        self.monitor_loss = monitor_loss
        self.monitor_mode = monitor_mode
        self.dense_activation = dense_activation
        self.earlystop = tf.keras.callbacks.EarlyStopping(monitor=self.monitor_loss,
                                                            patience=self.patience,
                                                            verbose=1,
                                                            restore_best_weights=True,
                                                            mode=self.monitor_mode)
        self.BERT_parent = TFBertModel.from_pretrained("bert-base-cased", output_attentions = True) #, config=self.bert_config)
        self.BERT_target = TFBertModel.from_pretrained("bert-base-cased", output_attentions = True)
        
  #prepare inputs for bert 
  def to_bert_input(self, sentences, text_to_encode='text'):
      input_ids, input_masks, input_segments = [],[],[]

      if text_to_encode == 'text':
        sentences = sentences.text
      elif text_to_encode == 'parent':
        sentences = sentences.parent
      else:
        print("Text to encode must be text (target) or parent")
      for sentence in tqdm(sentences):
          inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_seq_length, pad_to_max_length=True, 
                                              return_attention_mask=True, return_token_type_ids=True)
          input_ids.append(inputs['input_ids'])
          input_masks.append(inputs['attention_mask'])
          input_segments.append(inputs['token_type_ids'])        
          
      return (np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32'))

  def unfreeze_last_k_layers(self, bert,k=3):
    counter = 0
    #print(len(self.BERT.trainable_layers),'layersss')
    for layer in bert.layers:
      for l in layer.encoder.layer:
        if counter == len(layer.encoder.layer)-3:
          return
        else:
          counter+=1
          l.trainable = False

  def build(self, bias=0):
        self.unfreeze_last_k_layers(self.BERT_parent,3)
        self.unfreeze_last_k_layers(self.BERT_target,3) 
 

        #take the 'CLS' token of the parent 

        parent_in_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="parent_input_ids", dtype='int32')
        parent_in_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="parent_input_masks", dtype='int32')
        parent_in_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="parent_segment_ids", dtype='int32')
        bert_parent_inputs = [parent_in_id, parent_in_mask, parent_in_segment]

      
        bert_parent_output = self.BERT_parent(bert_parent_inputs).last_hidden_state
        bert_parent_output = bert_parent_output[:,0,:] #take only the embedding of the CLS token

        #take the 'CLS' token of the target
        in_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_ids", dtype='int32')
        in_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_masks", dtype='int32')
        in_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="segment_ids", dtype='int32')
        bert_inputs = [in_id, in_mask, in_segment]

        bert_output = self.BERT_target(bert_inputs).last_hidden_state
        bert_output = bert_output[:,0,:] #take only the embedding of the CLS token

        #Concatenate the 'CLS' tokens
        x = tf.keras.layers.concatenate([bert_parent_output, bert_output])
        print(x.shape, "sizee")

        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='tanh')(x)
        #x = tf.keras.layers.Dropout(0.1)(x)
        pred = tf.keras.layers.Dense(1, activation=self.dense_activation, bias_initializer=tf.keras.initializers.Constant(bias))(x)
        self.model = tf.keras.models.Model(inputs=bert_parent_inputs + bert_inputs, outputs=pred)
        self.model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=METRICS)
        if self.show_summary:
            self.model.summary()

  def fit(self, train, dev, bert_weights=None, class_weights={0: 1, 1: 1}, pretrained_embeddings=None):
        #encode parent (ignore second return arg)
        parent_input = self.to_bert_input(train, text_to_encode="parent")
        parent_dev = self.to_bert_input(dev, text_to_encode="parent")

        #encode target
        train_input = self.to_bert_input(train, text_to_encode="text")
        dev_input = self.to_bert_input(dev, text_to_encode="text")
        
        train_targets = train.target
        dev_targets = dev.target

        self.build()
        if bert_weights is not None:
            self.model.load_weights(bert_weights)
        #self.initialise_vars() # instantiation needs to be right before fitting
        self.model.fit(list(parent_input) + list(train_input),
                       train_targets,
                       validation_data=(list(parent_dev) + list(dev_input) , dev_targets),
                       epochs=self.epochs,
                       callbacks=[self.earlystop],
                       batch_size=self.batch_size,
                       class_weight=None 
                       )

  def predict(self, val_pd):
        #with self.session.as_default():

        #encode parent 
        parent_val_input = self.to_bert_input(val_pd, text_to_encode="parent")

        #encode target
        val_input = self.to_bert_input(val_pd, text_to_encode="text")
        predictions = self.model.predict(list(parent_val_input) + list(val_input))
        print('Stopped epoch: ', self.earlystop.stopped_epoch)
        if self.save_predictions:
            self.save_evaluation_set(val_targets, predictions)
        return predictions

  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)
    
  

  
