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
    
    def _vars(self, scope):
        res = []
        var = self.trainable_variables
        for v in var:
            if v.name.startswith(scope):
                res.append(v)
        assert len(res) > 0
        return res
    
    def loss_fn(self, X, y):
        """"""
        preds = self.predict(X)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=preds)
        return loss

    def acc_fn(self, X, y):
        preds = self.predict(X).numpy()
        acc = np.sum(np.argmax(preds, axis=1) == y.numpy(), dtype=np.float32) / X.numpy().shape[0]
        return acc 


def train(model, dataset, test_data, test_labels, 
          checkpoint, checkpoint_prefix, optimizer, epoches=10):
    test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)
    # train
    for epoch in range(epoches):
        losses = []
        for (batch, (inp, targ)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = model.loss_fn(inp, targ)
            
            # 对所有变量计算梯度
            # gradients = tape.gradient(loss, model.trainable_variables)
            # optimizer.apply_gradients(zip(gradients, model.variables))
            
            # 对部分变量计算梯度
            gradients = tape.gradient(loss, model._vars("Softmax"))
            optimizer.apply_gradients(zip(gradients, model._vars("Softmax")))
            # print(gradients, "\n\n")

            # print("loss: ", loss.numpy(), ",\tacc: ", model.acc_fn(inp, targ)*100, "%")
            # gradients, _ = tf.clip_by_global_norm(gradients, 1.)      # clip梯度

            losses.append(loss)
        
        print("Epoch :", epoch, ", train loss :", np.mean(losses))
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


# y = model.predict(tf.convert_to_tensor(test_data))
# print(model.trainable_variables)
# print("\n\n")
# print(model._vars("Dense1"))
