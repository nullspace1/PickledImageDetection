from model.CrossCorrelation import CrossCorrelation
from model.FeatureExtractor import FeatureExtractor
from model.DetectionHead import DetectionHead
from model.Loss import Loss

featureExtractor = FeatureExtractor()
crossCorrelation = CrossCorrelation()
detectionHead = DetectionHead()
loss = Loss(0.5,0.5)

featureExtractor.test()
crossCorrelation.test()
detectionHead.test()
loss.test()