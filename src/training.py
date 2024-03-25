from configs import configs


class Training:
	def __init__(self):
		self.configs = configs['S5_pendulum']
	

	def train(self, metrics, dataloader, learner):
		for steps in range(learner.training_steps):
			batch_inputs, batch_labels, mask = dataloader.batch_data(mode='train')
			loss, lr = learner.train_batch(batch_inputs, batch_labels, steps, mask)
			metrics.add_train_metrics(loss, lr, steps)
			if (steps + 1) % 20 == 0:
				batch_inputs, batch_labels, _ = dataloader.batch_data(mode='val')
				loss = learner.validate_batch(batch_inputs, batch_labels)
				metrics.add_val_loss(loss, steps)
			print(f"Step: {steps+1}/{learner.training_steps}", end='\r')
		metrics.close_writer()

