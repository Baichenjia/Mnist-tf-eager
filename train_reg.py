# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
import os
tf.enable_eager_execution()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_data = (train_images / 255.).astype(np.float32)
test_data = (test_images / 255.).astype(np.float32)
train_labels, test_labels = train_labels.astype(np.int), test_labels.astype(np.int)

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu, name="Dense1")
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu, name="Dense2")
        self.logits = tf.keras.layers.Dense(10, activation=None, name="logits")

    def predict(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.logits(x)
        return y

    def loss_fn(self, X, y):
        """"""
        preds = self.predict(X)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=preds)
        return loss

    def loss_reg_fn(self):
        var = self.trainable_variables
        loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in var])
        return loss_reg

    def acc_fn(self, X, y):
        preds = self.predict(X).numpy()
        acc = np.sum(np.argmax(preds, axis=1) == y.numpy(), dtype=np.float32) / X.numpy().shape[0]
        return acc 


def train(model, dataset, test_data, test_labels, 
          checkpoint, checkpoint_prefix, optimizer, epoches=10):
    test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)
    # train
    for epoch in range(epoches):
        losses_ent, losses_reg = [], []
        for (batch, (inp, targ)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss_ent = model.loss_fn(inp, targ)
                loss_reg = model.loss_reg_fn() * 0.01
                loss = loss_ent + loss_reg
                # print("loss_ent:", loss_ent.numpy(), ", loss_reg:", loss_reg.numpy())

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.variables))
            losses_ent.append(loss_ent)
            losses_reg.append(loss_reg)
        
        print("Epoch :", epoch, ", train loss entropy:", np.mean(losses_ent), ", train loss reg:", np.mean(losses_reg))
        acc = model.acc_fn(test_data, test_labels)
        print("Epoch :", epoch, ", valid acc:", acc*100, "%")
        checkpoint.save(file_prefix=checkpoint_prefix)


# model 
learning_rate = tf.Variable(1e-3, name="learning_rate")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
model = CNN()

# dataset
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(60000)
dataset = dataset.batch(32, drop_remainder=True)

# checkpoint
checkpoint_dir = 'checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, learning_rate=learning_rate, model=model)

# train
train(model, dataset, test_data, test_labels, checkpoint, checkpoint_prefix, optimizer, epoches=30)
