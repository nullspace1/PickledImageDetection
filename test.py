from model.CrossCorrelation import CrossCorrelation
from model.FeatureExtractor import FeatureExtractor
from model.DetectionHead import DetectionHead
from model.Loss import Loss

featureExtractor = FeatureExtractor(out_channels=(4,8,16))
crossCorrelation = CrossCorrelation(feature_map_size=(64,64))
detectionHead = DetectionHead(feature_extractor_out_channels=(4,8,16), conv_out_channels=16, pool_size=(16,16))
loss = Loss(0.5,0.5)

featureExtractor.test()
crossCorrelation.test()
detectionHead.test()
loss.test()