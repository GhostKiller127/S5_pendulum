from metrics import Metrics
from dataloader import PendulumDataloader
from learner import Learner
from training import Training


training = Training()
metrics = Metrics(training)
dataloader = PendulumDataloader(training)
learner = Learner(training)

training.train(metrics, dataloader, learner)